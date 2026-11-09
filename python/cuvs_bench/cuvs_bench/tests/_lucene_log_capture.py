#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Case-scoped log evidence for live Lucene integration tests."""

from __future__ import annotations

import sys
import os
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree


CPU_HNSW_FALLBACK_WARNING = (
    "GPU based indexing not supported, falling back to using the "
    "Lucene99HnswVectorsWriter"
)
GRAPH_CLAMP_WARNINGS = (
    "Intermediate graph degree cannot be larger",
    "cannot be larger than intermediate graph degree",
    "for nn-descent needs to match cagra intermediate graph degree",
)

_CASE_START = "CUVS_BENCH_LUCENE_CASE_START "
_CASE_END = "CUVS_BENCH_LUCENE_CASE_END "


def _validate_case_name(case_name: str) -> None:
    if not case_name or "\n" in case_name or "\r" in case_name:
        raise ValueError("Lucene log case names must be nonempty single lines")


@contextmanager
def lucene_log_case(case_name: str) -> Iterator[None]:
    """Delimit one serial test case on the JVM logger's stderr stream."""
    _validate_case_name(case_name)
    print(f"{_CASE_START}{case_name}", file=sys.stderr, flush=True)
    try:
        yield
    finally:
        print(f"{_CASE_END}{case_name}", file=sys.stderr, flush=True)


def lucene_case_output(output: str, case_name: str) -> str:
    """Return only output enclosed by markers for ``case_name``."""
    _validate_case_name(case_name)
    sections: dict[str, list[list[str]]] = {}
    active_case: str | None = None
    active_lines: list[str] | None = None

    for line in output.splitlines():
        if line.startswith(_CASE_START):
            started_case = line.removeprefix(_CASE_START)
            _validate_case_name(started_case)
            if active_case is not None:
                raise ValueError(
                    "Nested Lucene log case markers are not supported: "
                    f"{active_case!r}, {started_case!r}"
                )
            active_case = started_case
            active_lines = []
            continue
        if line.startswith(_CASE_END):
            ended_case = line.removeprefix(_CASE_END)
            _validate_case_name(ended_case)
            if active_case != ended_case or active_lines is None:
                raise ValueError(
                    "Mismatched Lucene log case marker: "
                    f"expected {active_case!r}, found {ended_case!r}"
                )
            sections.setdefault(ended_case, []).append(active_lines)
            active_case = None
            active_lines = None
            continue
        if active_lines is not None:
            active_lines.append(line)

    if active_case is not None:
        raise ValueError(
            f"Unterminated Lucene log case marker: {active_case!r}"
        )
    try:
        case_sections = sections[case_name]
    except KeyError as error:
        raise ValueError(
            f"Lucene log contains no complete case named {case_name!r}"
        ) from error
    return "\n".join(line for section in case_sections for line in section)


def case_used_cpu_hnsw_fallback(output: str, case_name: str) -> bool:
    """Report whether one accelerated-HNSW case logged its CPU fallback."""
    return CPU_HNSW_FALLBACK_WARNING in lucene_case_output(output, case_name)


def _run_warning_control_process(
    command: list[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Own the outer pytest and its nested GPU-hidden JVM as one process group."""
    if os.name != "posix":
        raise RuntimeError(
            "Lucene capacity warning controls require POSIX process groups"
        )
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as cleanup_error:
            error.add_note(
                f"Could not terminate warning-control process group: {cleanup_error}"
            )
        try:
            cleanup_timeout_seconds = 10
            process.communicate(timeout=cleanup_timeout_seconds)
        except BaseException as cleanup_error:
            error.add_note(
                f"Could not drain/reap warning-control process: {cleanup_error}"
            )
        finally:
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except OSError as cleanup_error:
                    error.add_note(
                        f"Could not close warning-control pipe: {cleanup_error}"
                    )
        raise
    return subprocess.CompletedProcess(
        command, process.returncode, stdout, stderr
    )


def require_observable_lucene_warnings(report_directory: Path) -> None:
    """Run real positive controls before trusting a capacity run's empty logs.

    A subprocess preserves the capacity process's ability to start its own
    larger JVM heap. It inherits the same artifact and native-library settings.
    Failure here invalidates GPU-writer/clamp-warning absence evidence: repair
    the runtime, warning emitter, logging or capture, then rerun qualification.
    """
    module = Path(__file__).with_name("test_lucene_integration.py")
    report = report_directory / "warning-controls.xml"
    report.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        "--run-lucene-e2e",
        f"--junitxml={report}",
        f"{module}::test_native_graph_clamp_warning_reaches_pytest",
        f"{module}::test_accelerated_hnsw_logs_cpu_fallback_when_gpu_is_hidden",
    ]
    failure_context = (
        "Lucene warning controls failed; do not trust capacity GPU-writer or "
        "clamp-warning absence checks. Inspect the runtime, warning emitters, "
        "logging and capture, then rerun the capacity suite."
    )
    control_timeout_seconds = 600
    try:
        completed = _run_warning_control_process(
            command, timeout=control_timeout_seconds
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"{failure_context} {error}") from error
    report_directory.joinpath("warning-controls.stdout.log").write_text(
        completed.stdout
    )
    report_directory.joinpath("warning-controls.stderr.log").write_text(
        completed.stderr
    )
    if completed.returncode:
        raise RuntimeError(
            f"{failure_context}\n{completed.stdout}\n{completed.stderr}"
        )
    try:
        cases = list(ElementTree.parse(report).iter("testcase"))
    except (OSError, ElementTree.ParseError) as error:
        raise RuntimeError(
            f"{failure_context} Missing or malformed control report: {report}"
        ) from error
    expected_control_count = (
        len(GRAPH_CLAMP_WARNINGS) + 1
    )  # One GPU-hidden fallback control.
    if len(cases) != expected_control_count or any(
        case.find(status) is not None
        for case in cases
        for status in ("failure", "error", "skipped")
    ):
        raise RuntimeError(
            f"{failure_context} Expected {expected_control_count} passing "
            f"controls and no skips; inspect {report}."
        )
