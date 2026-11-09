# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and persistence contracts; doubles do not exercise a GPU."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from _lucene_test_support import RecordingRuntime, _backend_and_index, _dataset
from cuvs_bench.backends.base import Dataset
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


_IVF_PARAMETERS = {
    "graph_degree": 32,
    "intermediate_graph_degree": 64,
    "graph_build_algo": "ivf_pq",
    "writer_threads": 2,
    "n_lists": 2,
    "n_probes": 1,
    "kmeans_n_iters": 3,
    "kmeans_trainset_fraction": 0.75,
    "pq_dim": 2,
    "pq_bits": 4,
    "refinement_rate": 2.5,
    "max_train_points_per_pq_code": 32,
    "codebook_kind": "per_cluster",
    "force_random_rotation": True,
    "lut_dtype": "uint8",
    "internal_distance_dtype": "float16",
}


class FakeControlledRuntime(RecordingRuntime):
    """Record build arguments, without claiming native parameter execution."""

    def __init__(self):
        super().__init__(pylucene_version="i-am-fake-data")
        self.received_controls = []

    def build_index(self, path, vectors, codec, build_parameters=None):
        self.received_controls.append(build_parameters)
        return super().build_index(path, vectors, codec)


@pytest.mark.parametrize(
    "algorithm", [ACCELERATED_HNSW_ALGORITHM, CAGRA_ALGORITHM]
)
def test_all_ivf_build_controls_reach_runtime_and_survive_saved_index_reuse(
    tmp_path, algorithm
):
    fake_runtime = FakeControlledRuntime()
    backend, index, _ = _backend_and_index(tmp_path, algorithm, fake_runtime)
    index = replace(
        index, build_param={**index.build_param, **_IVF_PARAMETERS}
    )
    expected = {**_IVF_PARAMETERS, "build_strategy": "custom"}

    built = backend.build(_dataset(), [index])
    reused = backend.build(_dataset(), [index])

    assert built.success, built.error_message
    assert reused.success, reused.error_message
    assert fake_runtime.received_controls == [expected]
    manifest = _read_manifest(Path(index.file))
    assert manifest["schema_version"] == 3
    assert manifest["build_parameters"] == expected
    assert manifest["runtime_build_topology"] is None
    assert reused.metadata["skipped"] is True


@pytest.mark.parametrize(
    "extra",
    [
        {},  # The fixed 1024-list default exceeds this four-vector fixture.
        {"n_lists": 5, "n_probes": 1},
        {
            "n_lists": 3,
            "n_probes": 1,
            "premerge_segment_count": 2,
            "force_merge_segment_count": 0,
            "ram_per_thread_hard_limit_mb": 512,
        },
    ],
)
def test_cagra_ivf_list_count_cannot_exceed_known_segment_capacity(
    tmp_path, extra
):
    fake_runtime = FakeControlledRuntime()
    backend, index, factory = _backend_and_index(
        tmp_path, CAGRA_ALGORITHM, fake_runtime
    )
    index = replace(
        index,
        build_param={
            **index.build_param,
            "graph_build_algo": "ivf_pq",
            **extra,
        },
    )

    result = backend.build(_dataset(), [index])

    assert not result.success
    assert "n_lists must not exceed" in result.error_message
    assert fake_runtime.received_controls == []
    assert factory.calls == []


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


@pytest.mark.parametrize(
    "algorithm", [ACCELERATED_HNSW_ALGORITHM, CAGRA_ALGORITHM]
)
@pytest.mark.parametrize(
    "supplied",
    [{}, {"n_lists": 64}, {"n_probes": 1024}, {"n_lists": 10, "n_probes": 10}],
    ids=[
        "all-defaults",
        "lists-only",
        "probe-all-default-lists",
        "explicit-pair",
    ],
)
def test_ivf_normalization_records_java_builder_defaults_and_explicit_overrides(
    algorithm, supplied
):
    request = {"graph_build_algo": "ivf_pq", **supplied}
    assert _build_controls(algorithm, request) == {
        "n_lists": 1024,
        "n_probes": 20,
        "pq_dim": 0,
        "kmeans_n_iters": 20,
        "kmeans_trainset_fraction": 0.5,
        **request,
        "build_strategy": "custom",
    }


@pytest.mark.parametrize(
    "old_implicit_default",
    [
        "n_lists",
        "n_probes",
        "pq_dim",
        "kmeans_n_iters",
        "kmeans_trainset_fraction",
    ],
)
def test_restored_ivf_defaults_are_persisted_and_old_implicit_defaults_are_not_reused(
    tmp_path,
    old_implicit_default,
):
    fake_runtime = FakeControlledRuntime()
    backend, index, _ = _backend_and_index(
        tmp_path, CAGRA_ALGORITHM, fake_runtime
    )
    index = replace(index, build_param={"graph_build_algo": "ivf_pq"})
    expected = {
        "graph_build_algo": "ivf_pq",
        "build_strategy": "custom",
        "n_lists": 1024,
        "n_probes": 20,
        "pq_dim": 0,
        "kmeans_n_iters": 20,
        "kmeans_trainset_fraction": 0.5,
    }
    # Only configuration/persistence is exercised: these fake vectors never reach cuVS.
    dataset = Dataset(
        name="fake-ivf-defaults",
        training_vectors=np.zeros((1024, 2), dtype=np.float32),
        distance_metric="euclidean",
    )

    built = backend.build(dataset, [index])
    reused = backend.build(dataset, [index])

    assert built.success, built.error_message
    assert reused.success, reused.error_message
    assert reused.metadata["skipped"] is True
    assert fake_runtime.received_controls == [expected]
    assert _read_manifest(Path(index.file))["build_parameters"] == expected

    # Older experimental manifests omitted these controls and used different defaults.
    manifest_path = Path(index.file) / ".cuvs-bench-lucene.json"
    old_manifest = json.loads(manifest_path.read_text())
    old_manifest["build_parameters"].pop(old_implicit_default)
    manifest_path.write_text(json.dumps(old_manifest))

    old_index_reuse = backend.build(dataset, [index])

    assert not old_index_reuse.success
    assert (
        "Invalid Lucene manifest build parameters"
        in old_index_reuse.error_message
    )
    assert fake_runtime.received_controls == [expected]


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
        (
            {"graph_build_algo": "ivf_pq", "nn_descent_n_iters": 3},
            "requires graph_build_algo=nn_descent",
        ),
        ({"n_lists": 128}, "require graph_build_algo=ivf_pq"),
        (
            {"graph_build_algo": "ivf_pq", "n_lists": 19},
            "n_probes must not exceed",
        ),
        (
            {"graph_build_algo": "ivf_pq", "n_probes": 1025},
            "n_probes must not exceed",
        ),
        (
            {"graph_build_algo": "ivf_pq", "n_lists": 10, "n_probes": 11},
            "n_probes must not exceed",
        ),
        (
            {"graph_build_algo": "ivf_pq", "pq_dim": 3, "pq_bits": 5},
            "divisible by 8",
        ),
        (
            {"graph_build_algo": "ivf_pq", "pq_bits": 3},
            "pq_bits must be an integer",
        ),
        (
            {"graph_build_algo": "ivf_pq", "force_random_rotation": 1},
            "must be a boolean",
        ),
        (
            {
                "graph_build_algo": "ivf_pq",
                "kmeans_trainset_fraction": float("nan"),
            },
            "must be finite",
        ),
        (
            {"graph_build_algo": "ivf_pq", "kmeans_trainset_fraction": 1.1},
            "must be <= 1",
        ),
        (
            {"graph_build_algo": "ivf_pq", "refinement_rate": 1},
            "must be finite",
        ),
        (
            {"graph_build_algo": "ivf_pq", "refinement_rate": 1 + 1e-9},
            "float32",
        ),
        ({"graph_build_algo": "ivf_pq", "refinement_rate": 1e100}, "float32"),
        (
            {"graph_build_algo": "ivf_pq", "internal_distance_dtype": "uint8"},
            "must be one of",
        ),
        ({"hnsw_layers": 3}, "Unsupported"),
    ],
)
def test_invalid_cagra_build_combinations_are_rejected(parameters, error):
    with pytest.raises(ValueError, match=error):
        _build_controls(CAGRA_ALGORITHM, parameters)


@pytest.mark.parametrize(
    "algorithm", [ACCELERATED_HNSW_ALGORITHM, CAGRA_ALGORITHM]
)
@pytest.mark.parametrize(
    "lut_settings",
    [{}, {"lut_dtype": "float32"}],
    ids=["default-lut", "explicit-lut"],
)
def test_ivf_rejects_half_precision_distances_with_float32_lookup_table(
    algorithm, lut_settings
):
    controls = {
        "graph_build_algo": "ivf_pq",
        "internal_distance_dtype": "float16",
        **lut_settings,
    }
    with pytest.raises(
        ValueError, match="float16 requires lut_dtype=float16 or uint8"
    ):
        _build_controls(algorithm, controls)


@pytest.mark.parametrize(
    "algorithm", [ACCELERATED_HNSW_ALGORITHM, CAGRA_ALGORITHM]
)
@pytest.mark.parametrize(
    "distance_dtype,lut_dtype",
    [
        ("float16", "float16"),
        ("float16", "uint8"),
        ("float32", "float16"),
        ("float32", "uint8"),
        ("float32", "float32"),
    ],
)
def test_ivf_preserves_compatible_distance_and_lookup_table_precisions(
    algorithm, distance_dtype, lut_dtype
):
    controls = {
        "graph_build_algo": "ivf_pq",
        "internal_distance_dtype": distance_dtype,
        "lut_dtype": lut_dtype,
    }
    assert _build_controls(algorithm, controls) == {
        "build_strategy": "custom",
        "n_lists": 1024,
        "n_probes": 20,
        "pq_dim": 0,
        "kmeans_n_iters": 20,
        "kmeans_trainset_fraction": 0.5,
        **controls,
    }


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
    controls = _build_controls(CAGRA_ALGORITHM, _IVF_PARAMETERS)
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
    assert _controlled_index_suffix({**controls, "n_probes": 2}) != suffix
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
