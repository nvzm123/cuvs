# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and persistence contracts; doubles do not exercise a GPU."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from _lucene_test_support import RecordingRuntime, _backend_and_index, _dataset
from cuvs_bench.backends.lucene import (
    ACCELERATED_HNSW_ALGORITHM,
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    LuceneConfigLoader,
    _build_controls,
    _controlled_index_suffix,
    _read_manifest,
    _search_parameters,
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
    "parameters,expected",
    [
        (
            {"graph_degree": 32},
            {"graph_degree": 32, "build_strategy": "custom"},
        ),
        (
            {"nn_descent_n_iters": 7},
            {"nn_descent_n_iters": 7, "build_strategy": "custom"},
        ),
        ({"build_strategy": "custom"}, {"build_strategy": "custom"}),
        ({"build_quality": 3}, {"build_quality": 3}),
        ({"writer_threads": 2}, {"writer_threads": 2}),
        (
            {"build_strategy": "heuristic", "graph_degree": 32},
            {"build_strategy": "heuristic", "graph_degree": 32},
        ),
    ],
)
def test_cagra_build_strategy_does_not_invent_or_drop_settings(
    parameters, expected
):
    assert _build_controls(CAGRA_ALGORITHM, parameters) == expected


def test_hnsw_layer_control_preserves_heuristic_mode_and_legacy_identity():
    parameters = {"m": 16, "beam_width": 100}
    assert (
        _build_controls(ACCELERATED_HNSW_ALGORITHM, parameters) == parameters
    )
    assert _controlled_index_suffix(parameters) == "~m16_bw100"
    heuristic = {
        **parameters,
        "hnsw_layers": 3,
    }
    assert _build_controls(ACCELERATED_HNSW_ALGORITHM, heuristic) == heuristic


@pytest.mark.parametrize(
    "algorithm",
    [CPU_HNSW_ALGORITHM, ACCELERATED_HNSW_ALGORITHM, CAGRA_ALGORITHM],
)
@pytest.mark.parametrize(
    "heuristic", ["same_graph_footprint", "similar_search_performance"]
)
def test_backend_rejects_hnsw_heuristic_selection(algorithm, heuristic):
    with pytest.raises(
        ValueError, match="Unsupported Lucene build parameters: hnsw_heuristic"
    ):
        _build_controls(algorithm, {"hnsw_heuristic": heuristic})


@pytest.mark.parametrize(
    "controls,schema,suffix",
    [
        ({}, 2, ""),
        ({"m": 16, "beam_width": 100}, 3, "~m16_bw100"),
    ],
)
def test_legacy_manifests_keep_their_original_schema_fields_and_reuse(
    tmp_path, controls, schema, suffix
):
    fake_runtime = FakeControlledRuntime()
    backend, index, _ = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    index = replace(index, build_param={**index.build_param, **controls})
    assert backend.build(_dataset(), [index]).success

    manifest = _read_manifest(Path(index.file))
    expected_fields = {
        "schema_version",
        "algorithm",
        "codec",
        "dataset",
        "segment_count",
        "build_runtime_artifacts",
    }
    if controls:
        expected_fields.update({"build_parameters", "runtime_build_topology"})
        assert manifest["build_parameters"] == controls
        assert manifest["runtime_build_topology"] is None
    assert set(manifest) == expected_fields
    assert manifest["schema_version"] == schema
    assert _controlled_index_suffix(controls) == suffix
    reused = backend.build(_dataset(), [index])
    assert reused.success, reused.error_message
    assert len(fake_runtime.build_calls) == 1


@pytest.mark.parametrize(
    "parameters,error",
    [
        (
            {"build_strategy": "heuristic", "intermediate_graph_degree": 64},
            "conflicts",
        ),
        ({"build_strategy": "custom", "build_quality": 3}, "conflicts"),
        ({"graph_degree": True}, "graph_degree must be an integer"),
        ({"graph_degree": 513}, "graph_degree must be an integer"),
        ({"graph_degree": 64, "intermediate_graph_degree": 32}, "must be >="),
        ({"hnsw_layers": 3}, "Unsupported"),
    ],
)
def test_invalid_cagra_build_combinations_are_rejected(parameters, error):
    with pytest.raises(ValueError, match=error):
        _build_controls(CAGRA_ALGORITHM, parameters)


@pytest.mark.parametrize(
    "parameters,error",
    [
        ({"m": 16, "beam_width": 100, "graph_degree": 33}, r"<= 2 \* m"),
        ({"build_quality": 2}, "Unsupported"),
        ({"build_strategy": "heuristic", "graph_degree": 32}, "conflicts"),
    ],
)
def test_hnsw_build_rejects_ignored_and_incompatible_controls(
    parameters, error
):
    with pytest.raises(ValueError, match=error):
        _build_controls(ACCELERATED_HNSW_ALGORITHM, parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"graph_degree": 32},
        {"writer_threads": 2},
        {"build_strategy": "custom"},
    ],
)
def test_cpu_build_does_not_accept_gpu_controls(parameters):
    with pytest.raises(ValueError, match="Unsupported"):
        _build_controls(CPU_HNSW_ALGORITHM, parameters)


def test_new_build_identity_is_bounded_stable_and_includes_every_control():
    controls = _build_controls(
        CAGRA_ALGORITHM, {"graph_degree": 32, "writer_threads": 2}
    )
    suffix = _controlled_index_suffix(controls)
    assert len(suffix) == 29
    assert suffix.startswith("~cfg-")
    assert (
        _controlled_index_suffix(dict(reversed(list(controls.items()))))
        == suffix
    )
    assert (
        _controlled_index_suffix({**controls, "writer_threads": 3}) != suffix
    )
    assert _controlled_index_suffix({**controls, "graph_degree": 64}) != suffix
    assert (
        _controlled_index_suffix({**controls, "m": 16, "beam_width": 100})
        != suffix
    )


@pytest.mark.parametrize(
    "corruption",
    [
        {"graph_degree": 513},
        {"build_strategy": "heuristic"},
        {"search_width": 2},
    ],
)
def test_saved_new_controls_are_validated_before_reuse(tmp_path, corruption):
    fake_runtime = FakeControlledRuntime()
    backend, index, _ = _backend_and_index(
        tmp_path, CAGRA_ALGORITHM, fake_runtime
    )
    index = replace(
        index, build_param={**index.build_param, "graph_degree": 32}
    )
    assert backend.build(_dataset(), [index]).success
    path = Path(index.file) / ".cuvs-bench-lucene.json"
    manifest = json.loads(path.read_text())
    if corruption == {"build_strategy": "heuristic"}:
        # Removing the inferred strategy is noncanonical even when valid as input.
        manifest["build_parameters"].pop("build_strategy")
    else:
        manifest["build_parameters"].update(corruption)
    path.write_text(json.dumps(manifest))

    reused = backend.build(_dataset(), [index])

    assert not reused.success
    assert "Invalid Lucene manifest build parameters" in reused.error_message
    assert len(fake_runtime.build_calls) == 1


def test_cagra_search_preserves_all_supplied_controls():
    parameters = {
        "itopk": 128,
        "search_width": 3,
        "max_iterations": 7,
        "algo": "multi_cta",
        "thread_block_size": 256,
    }
    assert _search_parameters(CAGRA_ALGORITHM, parameters, 10) == {
        "num_candidates": 10,
        **parameters,
    }


@pytest.mark.parametrize(
    "parameters,k",
    [
        ({"itopk": 9}, 10),
        ({"itopk": True}, 1),
        ({"itopk": 2**31}, 1),
        ({"search_width": 0}, 1),
        ({"search_width": 4_194_304}, 1),
        ({"max_iterations": -1}, 1),
        ({"max_iterations": 2**31}, 1),
        ({"thread_block_size": 32}, 1),
        ({"algo": "AUTO"}, 1),
        ({"algo": "single_cta", "itopk": 513}, 1),
        ({"algo": "single_cta"}, 513),
        ({"num_candidates": 100}, 10),
        ({"ef_search": 100}, 10),
        ({"k": 10}, 10),
    ],
)
def test_cagra_search_rejects_invalid_or_ignored_controls(parameters, k):
    with pytest.raises(ValueError):
        _search_parameters(CAGRA_ALGORITHM, parameters, k)


@pytest.mark.parametrize(
    "parameters,k",
    [
        ({"algo": "single_cta"}, 512),
        ({"algo": "multi_cta", "itopk": 2**31 - 1}, 1024),
        (
            {
                "search_width": 4_194_303,
                "max_iterations": 0,
                "thread_block_size": 0,
            },
            1,
        ),
    ],
)
def test_cagra_search_accepts_documented_parameter_boundaries(parameters, k):
    assert _search_parameters(CAGRA_ALGORITHM, parameters, k) == {
        "num_candidates": k,
        **parameters,
    }


@pytest.mark.parametrize(
    "algorithm", [CPU_HNSW_ALGORITHM, ACCELERATED_HNSW_ALGORITHM]
)
def test_ef_search_is_a_candidate_budget_alias_not_an_independent_control(
    algorithm,
):
    assert _search_parameters(algorithm, {"ef_search": 32}, 10) == {
        "num_candidates": 32,
        "ef_search": 32,
    }
    assert _search_parameters(
        algorithm, {"num_candidates": 32, "ef_search": 32}, 10
    ) == {"num_candidates": 32, "ef_search": 32}
    with pytest.raises(ValueError, match="must equal"):
        _search_parameters(
            algorithm, {"num_candidates": 32, "ef_search": 64}, 10
        )


def test_build_sweeps_have_distinct_paths_while_search_sweeps_share_an_index(
    tmp_path,
):
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text("- name: tiny-l2\n  distance: euclidean\n  dims: 2\n")
    algorithms = tmp_path / "algorithms.yaml"
    algorithms.write_text(
        f"name: {CAGRA_ALGORITHM}\ngroups:\n  custom:\n    build:\n      graph_degree: [16, 32]\n    search:\n      itopk: [32, 64]\n"
    )

    _, configurations = LuceneConfigLoader().load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(datasets),
        algorithm_configuration=str(algorithms),
        algorithms=CAGRA_ALGORITHM,
        groups="custom",
    )

    indexes = [config.indexes[0] for config in configurations]
    assert len(indexes) == 2
    assert len({index.file for index in indexes}) == 2
    assert [index.search_params for index in indexes] == [
        [{"itopk": 32}, {"itopk": 64}]
    ] * 2
    assert all("itopk" not in index.build_param for index in indexes)
