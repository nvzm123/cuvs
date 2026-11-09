#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for case-scoped Lucene execution-path log evidence."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import _lucene_log_capture as log_capture

from _lucene_log_capture import (
    CPU_HNSW_FALLBACK_WARNING,
    case_used_cpu_hnsw_fallback,
    require_observable_lucene_warnings,
)


def test_cpu_fallback_warning_is_attributed_only_to_its_named_case() -> None:
    fallback_case = (
        "test_lucene_integration.py::"
        "test_accelerated_hnsw_builds_on_gpu_and_searches_on_cpu"
    )
    gpu_cagra_case = (
        "test_lucene_integration.py::"
        "test_cagra_build_and_search_verify_persisted_index_and_search_behavior"
        "[gpu-cagra-aligned-dimensions]"
    )
    combined_log = "\n".join(
        (
            f"CUVS_BENCH_LUCENE_CASE_START {fallback_case}",
            "Sep 21, 2026 12:00:00 PM "
            "com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat "
            "fieldsWriter",
            f"WARNING: {CPU_HNSW_FALLBACK_WARNING}",
            f"CUVS_BENCH_LUCENE_CASE_END {fallback_case}",
            f"CUVS_BENCH_LUCENE_CASE_START {gpu_cagra_case}",
            "Sep 21, 2026 12:00:01 PM "
            "com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter mergeOneField",
            "INFO: Built CAGRA index",
            f"CUVS_BENCH_LUCENE_CASE_END {gpu_cagra_case}",
        )
    )

    assert case_used_cpu_hnsw_fallback(combined_log, fallback_case)
    assert not case_used_cpu_hnsw_fallback(combined_log, gpu_cagra_case)


def test_cpu_fallback_detection_fails_closed_when_case_markers_are_missing() -> (
    None
):
    with pytest.raises(ValueError, match="contains no complete case"):
        case_used_cpu_hnsw_fallback(
            f"WARNING: {CPU_HNSW_FALLBACK_WARNING}",
            "test_lucene_integration.py::test_unmarked_accelerated_hnsw_case",
        )


@pytest.mark.parametrize(
    "fake_report, fake_returncode",
    [
        pytest.param("<testsuite/>", 0, id="no-controls-executed"),
        pytest.param(
            "<testsuite><testcase/></testsuite>", 0, id="incomplete-selection"
        ),
        pytest.param(
            "<testsuite><testcase/><testcase/><testcase/><testcase><skipped/></testcase></testsuite>",
            0,
            id="skipped-control",
        ),
        pytest.param(
            "<testsuite><testcase><failure/></testcase></testsuite>",
            1,
            id="failed-control",
        ),
        pytest.param("i-am-a-malformed-fake-report", 0, id="malformed-report"),
        pytest.param(None, 0, id="missing-report-cannot-reuse-stale-success"),
    ],
)
def test_capacity_warning_gate_rejects_unproven_controls(
    tmp_path, monkeypatch, fake_report, fake_returncode
):
    report = tmp_path / "warning-controls.xml"
    report.write_text("<testsuite>" + "<testcase/>" * 4 + "</testsuite>")

    def fake_pytest_process(command, **kwargs):
        assert "--run-lucene-e2e" in command
        assert kwargs["timeout"] > 0
        if fake_report is not None:
            report.write_text(fake_report)
        return subprocess.CompletedProcess(
            args=command,
            returncode=fake_returncode,
            stdout="i-am-fake-pytest-output",
            stderr="i-am-fake-pytest-stderr",
        )

    monkeypatch.setattr(
        log_capture, "_run_warning_control_process", fake_pytest_process
    )

    with pytest.raises(RuntimeError, match="do not trust capacity GPU-writer"):
        require_observable_lucene_warnings(tmp_path)


def test_capacity_warning_gate_rejects_a_hung_control_process(
    tmp_path, monkeypatch
):
    def fake_hung_pytest_process(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(
        log_capture, "_run_warning_control_process", fake_hung_pytest_process
    )

    with pytest.raises(RuntimeError, match="warning controls failed"):
        require_observable_lucene_warnings(tmp_path)


def test_capacity_warning_gate_accepts_four_reported_passes(
    tmp_path, monkeypatch
):
    """Test report handling only; fabricated output does not establish live logging."""

    def fake_successful_pytest_process(command, **kwargs):
        report = tmp_path / "warning-controls.xml"
        report.write_text("<testsuite>" + "<testcase/>" * 4 + "</testsuite>")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="i-am-fake-pytest-output",
            stderr="",
        )

    monkeypatch.setattr(
        log_capture,
        "_run_warning_control_process",
        fake_successful_pytest_process,
    )

    require_observable_lucene_warnings(tmp_path)


@pytest.mark.skipif(
    not Path("/proc/self/stat").is_file(),
    reason="Linux process-state observation",
)
def test_warning_control_timeout_stops_its_nested_process(tmp_path):
    """Real processes prove cleanup reaches the grandchild, not only outer pytest."""
    descendant_pid_file = tmp_path / "nested-process.pid"
    parent_script = """
import subprocess
import sys
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(str(child.pid))
child.wait()
"""
    command = [sys.executable, "-c", parent_script, str(descendant_pid_file)]
    descendant_pid = None

    def descendant_is_running():
        try:
            state = (
                Path(f"/proc/{descendant_pid}/stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()[0]
            )
        except FileNotFoundError:
            return False
        # A killed grandchild may await the OS reaper; it no longer holds JVM/GPU resources.
        return state != "Z"

    try:
        with pytest.raises(subprocess.TimeoutExpired):
            log_capture._run_warning_control_process(command, timeout=2)
        assert descendant_pid_file.is_file(), (
            "Nested probe did not start before the timeout"
        )
        descendant_pid = int(descendant_pid_file.read_text())
        deadline = time.monotonic() + 5
        while descendant_is_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not descendant_is_running(), (
            "Warning-control timeout left a descendant running"
        )
    finally:
        # Also clean up when a regression makes the assertion fail.
        if descendant_pid is None and descendant_pid_file.is_file():
            descendant_pid = int(descendant_pid_file.read_text())
        if descendant_pid is not None and descendant_is_running():
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
