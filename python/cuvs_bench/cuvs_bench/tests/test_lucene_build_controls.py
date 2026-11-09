# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-control contracts without a JVM; live tests establish actual topology."""

from dataclasses import replace

import pytest

from _lucene_test_support import RecordingRuntime, _backend_and_index, _dataset
from cuvs_bench.backends import lucene as backend_module
from cuvs_bench.backends.lucene import (
    ACCELERATED_HNSW_ALGORITHM,
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    LuceneBackend,
    LuceneConfigLoader,
    _build_controls,
    _build_parameters_for,
    _controlled_index_suffix,
)


class FakeControlledRuntime(RecordingRuntime):
    """Record build arguments, without claiming native parameter execution."""

    def __init__(self):
        super().__init__(pylucene_version="i-am-fake-data")
        self.received_controls = []

    def build_index(self, path, vectors, codec, build_parameters=None):
        self.received_controls.append(build_parameters)
        return super().build_index(path, vectors, codec)


@pytest.mark.parametrize(
    "algorithm",
    [ACCELERATED_HNSW_ALGORITHM, CAGRA_ALGORITHM, CPU_HNSW_ALGORITHM],
)
def test_default_build_does_not_invent_controls_or_change_index_names(
    algorithm,
):
    assert _build_controls(algorithm, {}) == {}
    assert _controlled_index_suffix({}) == ""


@pytest.mark.parametrize(
    "parameters, error",
    [
        pytest.param(
            {"m": 16}, "both m and beam_width", id="missing-beam-width"
        ),
        pytest.param(
            {"m": True, "beam_width": 100},
            "m must be an integer",
            id="boolean-is-not-m",
        ),
        pytest.param(
            {"m": 16, "beam_width": 0},
            "beam_width must be an integer",
            id="zero-beam-width",
        ),
        pytest.param(
            {"m": 513, "beam_width": 100},
            "m must be an integer",
            id="m-above-api-limit",
        ),
    ],
)
def test_invalid_build_controls_fail_before_creating_a_runtime(
    parameters, error
):
    with pytest.raises(ValueError, match=error):
        _build_parameters_for(ACCELERATED_HNSW_ALGORITHM, parameters)


def test_stock_cpu_hnsw_does_not_silently_accept_accelerated_build_controls():
    with pytest.raises(
        ValueError, match="Unsupported Lucene build parameters"
    ):
        _build_parameters_for(CPU_HNSW_ALGORITHM, {"m": 16, "beam_width": 100})


def test_different_build_parameters_cannot_reuse_the_same_index(tmp_path):
    fake_runtime = FakeControlledRuntime()
    backend, index, _ = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    original = replace(
        index, build_param={**index.build_param, "m": 16, "beam_width": 100}
    )
    assert backend.build(_dataset(), [original]).success

    changed = replace(original, build_param={**original.build_param, "m": 32})
    result = backend.build(_dataset(), [changed])

    assert not result.success
    assert "different build parameters" in result.error_message
    assert len(fake_runtime.build_calls) == 1


def test_configured_saved_index_remains_usable_when_writer_default_changes(
    tmp_path, monkeypatch
):
    """Fake codec names test routing only, not Lucene version compatibility."""
    fake_runtime = FakeControlledRuntime()
    backend, index, factory = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    controls = {"m": 16, "beam_width": 100}
    index = replace(index, build_param={**index.build_param, **controls})
    assert backend.build(_dataset(), [index]).success
    saved_codec = backend.codec
    fake_new_codec = "i-am-a-fake-new-writer"
    monkeypatch.setitem(
        backend_module._CODEC_BY_ALGORITHM,
        ACCELERATED_HNSW_ALGORITHM,
        fake_new_codec,
    )
    backend = LuceneBackend(
        {**backend.config, "codec": fake_new_codec}, runtime_factory=factory
    )
    index = replace(index, build_param={"codec": fake_new_codec, **controls})

    reused = backend.build(_dataset(), [index])
    searched = backend.search(_dataset(), [index], k=2)[0]

    assert reused.success, reused.error_message
    assert searched.success, searched.error_message
    assert (
        reused.metadata["codec"] == searched.metadata["codec"] == saved_codec
    )
    assert len(fake_runtime.build_calls) == 1


def test_build_parameter_sweep_assigns_distinct_index_files(tmp_path):
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text("- name: tiny-l2\n  distance: euclidean\n  dims: 2\n")
    algorithms = tmp_path / "algorithms.yaml"
    algorithms.write_text(
        f"name: {ACCELERATED_HNSW_ALGORITHM}\n"
        "groups:\n  custom:\n    build:\n      m: [16, 32]\n"
        "      beam_width: [100]\n    search:\n      num_candidates: [10]\n"
    )

    _, configurations = LuceneConfigLoader().load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(datasets),
        algorithm_configuration=str(algorithms),
        algorithms=ACCELERATED_HNSW_ALGORITHM,
        groups="custom",
    )

    indexes = [
        index
        for configuration in configurations
        for index in configuration.indexes
    ]
    assert [index.build_param["m"] for index in indexes] == [16, 32]
    assert len({index.file for index in indexes}) == 2


def test_controlled_base_build_cannot_replace_an_ordinary_named_group(
    tmp_path,
):
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text("- name: tiny-l2\n  distance: euclidean\n  dims: 2\n")
    algorithms = tmp_path / "algorithms.yaml"
    algorithms.write_text(
        f"name: {ACCELERATED_HNSW_ALGORITHM}\n"
        "groups:\n  base:\n    build:\n      m: [16]\n      beam_width: [100]\n"
        "  m16_bw100:\n    build: {}\n"
    )
    _, configurations = LuceneConfigLoader().load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(datasets),
        algorithm_configuration=str(algorithms),
        algorithms=ACCELERATED_HNSW_ALGORITHM,
        groups="base,m16_bw100",
    )
    controlled, ordinary = [
        configuration.indexes[0] for configuration in configurations
    ]
    assert ordinary.name == f"{ACCELERATED_HNSW_ALGORITHM}_m16_bw100"
    assert controlled.file != ordinary.file

    fake_runtime = FakeControlledRuntime()
    fake_environment_backend, _, fake_factory = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    backends = [
        LuceneBackend(
            {
                **fake_environment_backend.config,
                **configuration.backend_config,
            },
            runtime_factory=fake_factory,
        )
        for configuration in configurations
    ]
    # Build both before reusing either: a shared path would overwrite or reject one.
    for backend, index in zip(backends, (controlled, ordinary)):
        result = backend.build(_dataset(), [index])
        assert result.success, result.error_message
    for backend, index in zip(backends, (controlled, ordinary)):
        result = backend.build(_dataset(), [index])
        assert result.success, result.error_message
    assert len(fake_runtime.build_calls) == 2
