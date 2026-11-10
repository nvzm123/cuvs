#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public CLI and installed-plugin checks for the Lucene backend."""

from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from cuvs_bench.backends.base import BuildResult
from cuvs_bench.backends.lucene import (
    CAGRA_ALGORITHM,
    register as register_lucene,
)
from cuvs_bench.orchestrator import BenchmarkOrchestrator
from cuvs_bench.run.__main__ import main as run_main


_LUCENE_ENTRY_POINT = "cuvs_bench.backends.lucene:register"


def _write_lucene_backend_config(tmp_path: Path) -> Path:
    backend_config = tmp_path / "lucene-backend.yaml"
    backend_config.write_text("backend: lucene\n", encoding="utf-8")
    return backend_config


def _verify_installed_lucene_plugin() -> None:
    """Exercise installed metadata and cold plugin discovery."""
    for group in ("cuvs_bench.backends", "cuvs_bench.config_loaders"):
        assert any(
            item.name == "lucene" and item.value == _LUCENE_ENTRY_POINT
            for item in entry_points(group=group)
        ), f"installed cuvs-bench is missing the {group!r} Lucene entry point"

    orchestrator = BenchmarkOrchestrator("lucene")
    assert orchestrator.backend_type == "lucene"


def _invoke_run(tmp_path: Path, *extra_args: str, input_text: str = ""):
    register_lucene()
    captured = {}

    class RecordingOrchestrator:
        def __init__(self, backend_type):
            captured["backend_type"] = backend_type

        def run_benchmark(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return []

    arguments = [
        "--dataset",
        "test-data",
        "--dataset-path",
        str(tmp_path),
        "--batch-size",
        "10",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
        "--dry-run",
        *extra_args,
    ]
    with patch(
        "cuvs_bench.run.__main__.BenchmarkOrchestrator",
        RecordingOrchestrator,
    ):
        result = CliRunner().invoke(run_main, arguments, input=input_text)
    return result, captured


def test_backend_config_selects_the_lucene_default(tmp_path: Path) -> None:
    backend_config = _write_lucene_backend_config(tmp_path)

    result, captured = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        input_text="\n",
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "lucene"
    assert captured["run_kwargs"]["algorithms"] == CAGRA_ALGORITHM


def test_backend_config_preserves_an_explicit_cpu_algorithm(
    tmp_path: Path,
) -> None:
    backend_config = _write_lucene_backend_config(tmp_path)

    result, captured = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        "--algorithms",
        "lucene_cpu_hnsw",
    )

    assert result.exit_code == 0, result.output
    assert captured["run_kwargs"]["algorithms"] == "lucene_cpu_hnsw"


def test_cli_records_failed_lucene_result_before_exiting(tmp_path: Path):
    backend_config = _write_lucene_backend_config(tmp_path)
    failed = BuildResult(
        index_path="",
        build_time_seconds=0.0,
        index_size_bytes=0,
        algorithm=CAGRA_ALGORITHM,
        build_params={},
        success=False,
        error_message="GPU path was unavailable",
    )
    arguments = [
        "--dataset",
        "test-data",
        "--dataset-path",
        str(tmp_path),
        "--batch-size",
        "10",
        "-k",
        "10",
        "--groups",
        "test",
        "-m",
        "latency",
        "--backend-config",
        str(backend_config),
        "--algorithms",
        CAGRA_ALGORITHM,
    ]
    register_lucene()

    with (
        patch.object(
            BenchmarkOrchestrator,
            "run_benchmark",
            return_value=[failed],
        ),
        patch("cuvs_bench.run.__main__.write_results_to_csv") as write_results,
    ):
        result = CliRunner().invoke(run_main, arguments)

    assert result.exit_code == 1
    assert "GPU path was unavailable" in result.output
    write_results.assert_called_once()


if __name__ == "__main__":
    _verify_installed_lucene_plugin()
