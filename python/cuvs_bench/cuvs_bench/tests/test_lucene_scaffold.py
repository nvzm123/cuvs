# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opt-in Lucene backend scaffold."""

import os
import subprocess
import tomllib
from importlib import resources
from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from cuvs_bench.backends.base import BuildResult
from cuvs_bench.backends.lucene import (
    CAGRA_ALGORITHM,
    LuceneBackend,
    register as register_lucene,
)
from cuvs_bench.orchestrator import BenchmarkOrchestrator
from cuvs_bench.run.__main__ import main as run_main


_PROJECT_ROOT = Path(__file__).parents[2]
_REPOSITORY_ROOT = Path(__file__).parents[4]
_LUCENE_ENTRY_POINT = "cuvs_bench.backends.lucene:register"
_PYLUCENE_RECIPE_DIRECTORY = (
    _REPOSITORY_ROOT / "conda" / "recipes" / "cuvs-bench"
)
_PYLUCENE_BUILD_HELPER = (
    _PYLUCENE_RECIPE_DIRECTORY / "build_pylucene_10_2.sh"
)
_PYLUCENE_PATCH = _PYLUCENE_RECIPE_DIRECTORY / "pylucene-10.2.0.patch"


def _verify_installed_lucene_plugin() -> None:
    """Exercise the installed entry points without importing PyLucene."""
    for group in ("cuvs_bench.backends", "cuvs_bench.config_loaders"):
        assert any(
            item.name == "lucene" and item.value == _LUCENE_ENTRY_POINT
            for item in entry_points(group=group)
        ), f"installed cuvs-bench is missing the {group!r} Lucene entry point"

    orchestrator = BenchmarkOrchestrator("lucene")
    assert orchestrator.backend_type == "lucene"


def _invoke_run(tmp_path, *extra_args, input_text=""):
    register_lucene()
    captured = {}

    class RecordingOrchestrator:
        def __init__(self, backend_type):
            captured["backend_type"] = backend_type

        def run_benchmark(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return []

    args = [
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
        result = CliRunner().invoke(run_main, args, input=input_text)
    return result, captured


def test_ordinary_cli_keeps_cpp_backend_and_cagra_default(tmp_path):
    result, captured = _invoke_run(tmp_path, input_text="\n")

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "cpp_gbench"
    assert captured["run_kwargs"]["algorithms"] == "cuvs_cagra"


def test_lucene_backend_selects_cagra_default(tmp_path):
    result, captured = _invoke_run(
        tmp_path, "--backend", "lucene", input_text="\n"
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "lucene"
    assert captured["run_kwargs"]["algorithms"] == "lucene_cuvs_cagra"


def test_registered_backend_class_controls_the_prompt_default(tmp_path):
    class PluginBackend:
        default_algorithm = "plugin_default"

    with patch(
        "cuvs_bench.run.__main__.get_backend_class",
        return_value=PluginBackend,
    ):
        result, captured = _invoke_run(
            tmp_path,
            "--backend",
            "third_party",
            input_text="\n",
        )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "third_party"
    assert captured["run_kwargs"]["algorithms"] == "plugin_default"


def test_plugin_without_a_default_keeps_the_cli_default(tmp_path):
    class PluginBackend:
        default_algorithm = None

    class DefaultBackend:
        default_algorithm = "cuvs_cagra"

    def backend_class(name):
        return PluginBackend if name == "third_party" else DefaultBackend

    with patch(
        "cuvs_bench.run.__main__.get_backend_class",
        side_effect=backend_class,
    ):
        result, captured = _invoke_run(
            tmp_path,
            "--backend",
            "third_party",
            input_text="\n",
        )

    assert result.exit_code == 0, result.output
    assert captured["run_kwargs"]["algorithms"] == "cuvs_cagra"


def test_lucene_backend_preserves_explicit_cpu_algorithm(tmp_path):
    result, captured = _invoke_run(
        tmp_path,
        "--backend",
        "lucene",
        "--algorithms",
        "lucene_cpu_hnsw",
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "lucene"
    assert captured["run_kwargs"]["algorithms"] == "lucene_cpu_hnsw"


def test_lucene_backend_does_not_rewrite_explicit_algorithm(tmp_path):
    result, captured = _invoke_run(
        tmp_path,
        "--backend",
        "lucene",
        "--algorithms",
        "cuvs_cagra",
    )

    assert result.exit_code == 0, result.output
    assert captured["run_kwargs"]["algorithms"] == "cuvs_cagra"


def test_lucene_backend_preserves_an_algorithm_typed_at_the_prompt(tmp_path):
    result, captured = _invoke_run(
        tmp_path,
        "--backend",
        "lucene",
        input_text="cuvs_cagra\n",
    )

    assert result.exit_code == 0, result.output
    assert captured["run_kwargs"]["algorithms"] == "cuvs_cagra"


def test_lucene_backend_config_sets_the_prompt_default(tmp_path):
    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("backend: lucene\n", encoding="utf-8")

    result, captured = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        input_text="\n",
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "lucene"
    assert captured["run_kwargs"]["algorithms"] == "lucene_cuvs_cagra"


def test_backend_config_keeps_selecting_backend_without_new_option(tmp_path):
    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("backend: opensearch\nhost: search.example\n")

    result, captured = _invoke_run(
        tmp_path,
        "--backend-config",
        str(backend_config),
        "--algorithms",
        "opensearch_faiss_hnsw",
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_type"] == "opensearch"
    assert captured["run_kwargs"]["host"] == "search.example"


def test_explicit_backend_must_match_backend_config(tmp_path):
    backend_config = tmp_path / "backend.yaml"
    backend_config.write_text("backend: cpp_gbench\n")

    result, captured = _invoke_run(
        tmp_path,
        "--backend",
        "lucene",
        "--backend-config",
        str(backend_config),
        "--algorithms",
        "lucene_cuvs_cagra",
    )

    assert result.exit_code != 0
    assert "must match" in str(result.exception)
    assert captured == {}


def test_lucene_plugin_uses_lazy_entry_points():
    pyproject = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text())
    entry_points = pyproject["project"]["entry-points"]

    assert (
        entry_points["cuvs_bench.backends"]["lucene"]
        == _LUCENE_ENTRY_POINT
    )
    assert (
        entry_points["cuvs_bench.config_loaders"]["lucene"]
        == _LUCENE_ENTRY_POINT
    )


def test_lucene_backend_makes_failed_results_fatal() -> None:
    successful = BuildResult(
        index_path="",
        build_time_seconds=0.0,
        index_size_bytes=0,
        algorithm=CAGRA_ALGORITHM,
        build_params={},
    )
    failed = BuildResult(
        index_path="",
        build_time_seconds=0.0,
        index_size_bytes=0,
        algorithm=CAGRA_ALGORITHM,
        build_params={},
        success=False,
        error_message="GPU path was unavailable",
    )

    assert LuceneBackend.result_failure_message([successful]) is None
    assert LuceneBackend.result_failure_message([successful, failed]) == (
        "GPU path was unavailable"
    )


def test_lucene_cli_fails_after_recording_failed_results(tmp_path: Path):
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
        "--backend",
        "lucene",
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


def test_lucene_algorithm_configs_are_packaged_resources():
    expected_codecs = {
        "lucene_accelerated_hnsw": "Lucene101AcceleratedHNSWCodec",
        "lucene_cpu_hnsw": "Lucene101",
        "lucene_cuvs_cagra": "CuVS2510GPUSearchCodec",
    }
    algorithm_resources = resources.files("cuvs_bench.config.algos")

    for algorithm, codec in expected_codecs.items():
        config = yaml.safe_load(
            algorithm_resources.joinpath(f"{algorithm}.yaml").read_text()
        )
        assert config["name"] == algorithm
        assert set(config["groups"]) == {"base", "test"}
        for group in config["groups"].values():
            assert group == {
                "build": {"codec": [codec]},
                "search": {},
            }


def test_pylucene_source_builder_is_self_describing() -> None:
    completed = subprocess.run(
        ["bash", str(_PYLUCENE_BUILD_HELPER), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert os.access(_PYLUCENE_BUILD_HELPER, os.X_OK)
    assert _PYLUCENE_PATCH.is_file()
    assert completed.returncode == 0, completed.stderr
    assert "Build an isolated PyLucene 10.2.0 environment" in completed.stdout


def test_pylucene_source_builder_rejects_unsafe_build_roots(
    tmp_path: Path,
) -> None:
    for unsafe_character in (" ", ":", ";", "$", "`", "&", "#", "|"):
        build_root = tmp_path / f"unsafe{unsafe_character}root"
        completed = subprocess.run(
            [
                "bash",
                str(_PYLUCENE_BUILD_HELPER),
                "--build-root",
                str(build_root),
                "--prepare-only",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode != 0
        assert "--build-root may contain only" in completed.stderr
        assert not build_root.exists()


def test_pylucene_source_builder_rejects_unsafe_resolved_build_root(
    tmp_path: Path,
) -> None:
    unsafe_target = tmp_path / "unsafe;target"
    unsafe_target.mkdir()
    build_root = tmp_path / "safe-link"
    build_root.symlink_to(unsafe_target, target_is_directory=True)

    completed = subprocess.run(
        [
            "bash",
            str(_PYLUCENE_BUILD_HELPER),
            "--build-root",
            str(build_root),
            "--prepare-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--build-root may contain only" in completed.stderr
    assert not (unsafe_target / ".build.lock").exists()


if __name__ == "__main__":
    _verify_installed_lucene_plugin()
