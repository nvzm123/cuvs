#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Result, parameter, and configuration contracts for the Lucene backend."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from _lucene_test_support import (
    ALGORITHM_CASES,
    RecordingRuntime,
    _backend_and_index,
    _dataset,
)
from cuvs_bench.backends._lucene_runtime import (
    ACCELERATED_HNSW_CODEC,
    CAGRA_CODEC,
    CPU_HNSW_CODEC,
)
from cuvs_bench.backends.lucene import (
    ACCELERATED_HNSW_ALGORITHM,
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    LuceneBackend,
    LuceneConfigLoader,
    _build_parameter_metadata,
    _build_parameters_for,
    _codec_for,
    _score_to_squared_euclidean,
    _search_parameters,
)
from cuvs_bench.run.data_export import write_results_to_csv


@pytest.mark.parametrize(("algorithm", "codec", "_path"), ALGORITHM_CASES)
def test_backend_accepts_only_the_codec_owned_by_each_algorithm(
    tmp_path: Path, algorithm: str, codec: str, _path: str
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(tmp_path, algorithm, runtime)

    assert backend.algorithm == algorithm
    assert backend.codec == codec
    assert _codec_for(index.algo, index.build_param) == codec


@pytest.mark.parametrize(
    ("algorithm", "codec"),
    (
        pytest.param(CPU_HNSW_ALGORITHM, CAGRA_CODEC, id="cpu-with-cagra"),
        pytest.param(CAGRA_ALGORITHM, CPU_HNSW_CODEC, id="cagra-with-cpu"),
        pytest.param(
            ACCELERATED_HNSW_ALGORITHM,
            CPU_HNSW_CODEC,
            id="accelerated-with-cpu",
        ),
    ),
)
def test_backend_rejects_mismatched_algorithm_and_codec(
    tmp_path: Path, algorithm: str, codec: str
) -> None:
    with pytest.raises(
        ValueError, match="Invalid Lucene algorithm/codec pair"
    ):
        LuceneBackend(
            {
                "name": "invalid",
                "algo": algorithm,
                "codec": codec,
                "index_root": str(tmp_path),
            }
        )


@pytest.mark.parametrize(("algorithm", "codec", "_path"), ALGORITHM_CASES)
def test_build_parameters_accept_only_the_fixed_codec(
    algorithm: str, codec: str, _path: str
) -> None:
    assert _codec_for(algorithm, {}) == codec
    assert _codec_for(algorithm, {"codec": codec}) == codec

    with pytest.raises(
        ValueError, match="Unsupported Lucene build parameters"
    ):
        _codec_for(algorithm, {"codec": codec, "graph_degree": 32})


def test_accelerated_hnsw_build_parameters_are_canonical() -> None:
    assert _build_parameters_for(ACCELERATED_HNSW_ALGORITHM, {}) == {
        "codec": ACCELERATED_HNSW_CODEC,
        "m": 32,
        "beam_width": 32,
    }
    assert _build_parameters_for(
        ACCELERATED_HNSW_ALGORITHM,
        {
            "codec": ACCELERATED_HNSW_CODEC,
            "m": 16,
            "beam_width": 80,
        },
    ) == {
        "codec": ACCELERATED_HNSW_CODEC,
        "m": 16,
        "beam_width": 80,
    }


def test_accelerated_hnsw_metadata_records_the_heuristic_derivation() -> None:
    assert _build_parameter_metadata(
        ACCELERATED_HNSW_ALGORITHM,
        {
            "codec": ACCELERATED_HNSW_CODEC,
            "m": 16,
            "beam_width": 80,
        },
    ) == {
        "hnsw_m": 16,
        "hnsw_beam_width": 80,
        "hnsw_heuristic": "SAME_GRAPH_FOOTPRINT",
        "graph_degree": 32,
        "intermediate_graph_degree": 48,
    }


def test_accelerated_hnsw_segment_topology_is_canonical_and_reported() -> None:
    parameters = _build_parameters_for(
        ACCELERATED_HNSW_ALGORITHM,
        {
            "ram_per_thread_hard_limit_mb": 61440,
            "force_merge_segment_count": 1,
            "beam_width": 80,
            "codec": ACCELERATED_HNSW_CODEC,
            "premerge_segment_count": 4,
            "m": 16,
        },
    )

    assert list(parameters) == [
        "codec",
        "m",
        "beam_width",
        "premerge_segment_count",
        "force_merge_segment_count",
        "ram_per_thread_hard_limit_mb",
    ]
    assert parameters == {
        "codec": ACCELERATED_HNSW_CODEC,
        "m": 16,
        "beam_width": 80,
        "premerge_segment_count": 4,
        "force_merge_segment_count": 1,
        "ram_per_thread_hard_limit_mb": 61440,
    }
    assert _build_parameter_metadata(
        ACCELERATED_HNSW_ALGORITHM, parameters
    ) == {
        "hnsw_m": 16,
        "hnsw_beam_width": 80,
        "hnsw_heuristic": "SAME_GRAPH_FOOTPRINT",
        "graph_degree": 32,
        "intermediate_graph_degree": 48,
        "requested_premerge_segment_count": 4,
        "requested_force_merge_segment_count": 1,
        "ram_per_thread_hard_limit_mb": 61440,
    }


@pytest.mark.parametrize("force_merge_segment_count", (0, 1))
def test_accelerated_hnsw_partitioned_topology_is_canonical_and_reported(
    force_merge_segment_count: int,
) -> None:
    parameters = _build_parameters_for(
        ACCELERATED_HNSW_ALGORITHM,
        {
            "codec": ACCELERATED_HNSW_CODEC,
            "m": 16,
            "beam_width": 80,
            "num_indexing_threads": 4,
            "force_merge_segment_count": force_merge_segment_count,
            "ram_per_thread_hard_limit_mb": 61440,
        },
    )

    assert parameters == {
        "codec": ACCELERATED_HNSW_CODEC,
        "m": 16,
        "beam_width": 80,
        "num_indexing_threads": 4,
        "force_merge_segment_count": force_merge_segment_count,
        "ram_per_thread_hard_limit_mb": 61440,
    }
    assert _build_parameter_metadata(
        ACCELERATED_HNSW_ALGORITHM, parameters
    ) == {
        "hnsw_m": 16,
        "hnsw_beam_width": 80,
        "hnsw_heuristic": "SAME_GRAPH_FOOTPRINT",
        "graph_degree": 32,
        "intermediate_graph_degree": 48,
        "requested_num_indexing_threads": 4,
        "requested_force_merge_segment_count": force_merge_segment_count,
        "ram_per_thread_hard_limit_mb": 61440,
    }


def test_accelerated_hnsw_rejects_mixed_topology_models() -> None:
    with pytest.raises(ValueError, match="cannot combine"):
        _build_parameters_for(
            ACCELERATED_HNSW_ALGORITHM,
            {
                "codec": ACCELERATED_HNSW_CODEC,
                "premerge_segment_count": 4,
                "num_indexing_threads": 4,
                "force_merge_segment_count": 1,
                "ram_per_thread_hard_limit_mb": 61440,
            },
        )


@pytest.mark.parametrize(
    "missing",
    (
        "premerge_segment_count",
        "force_merge_segment_count",
        "ram_per_thread_hard_limit_mb",
    ),
)
def test_accelerated_hnsw_segment_topology_must_be_all_or_none(
    missing: str,
) -> None:
    topology = {
        "premerge_segment_count": 4,
        "force_merge_segment_count": 1,
        "ram_per_thread_hard_limit_mb": 61440,
    }
    topology.pop(missing)

    with pytest.raises(ValueError, match="segment topology requires.*missing"):
        _build_parameters_for(
            ACCELERATED_HNSW_ALGORITHM,
            {"codec": ACCELERATED_HNSW_CODEC, **topology},
        )


@pytest.mark.parametrize(
    "missing",
    (
        "num_indexing_threads",
        "force_merge_segment_count",
        "ram_per_thread_hard_limit_mb",
    ),
)
def test_accelerated_hnsw_partitioned_topology_must_be_all_or_none(
    missing: str,
) -> None:
    topology = {
        "num_indexing_threads": 4,
        "force_merge_segment_count": 1,
        "ram_per_thread_hard_limit_mb": 61440,
    }
    topology.pop(missing)

    with pytest.raises(ValueError, match="segment topology requires.*missing"):
        _build_parameters_for(
            ACCELERATED_HNSW_ALGORITHM,
            {"codec": ACCELERATED_HNSW_CODEC, **topology},
        )


@pytest.mark.parametrize(
    "value",
    (0, -1, True, 1.5, "4", None),
    ids=("zero", "negative", "boolean", "float", "string", "none"),
)
def test_accelerated_hnsw_rejects_invalid_num_indexing_threads(
    value: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="Lucene num_indexing_threads must be a positive integer",
    ):
        _build_parameters_for(
            ACCELERATED_HNSW_ALGORITHM,
            {
                "codec": ACCELERATED_HNSW_CODEC,
                "num_indexing_threads": value,
                "force_merge_segment_count": 1,
                "ram_per_thread_hard_limit_mb": 61440,
            },
        )


@pytest.mark.parametrize(
    "name",
    (
        "premerge_segment_count",
        "ram_per_thread_hard_limit_mb",
    ),
)
@pytest.mark.parametrize(
    "value",
    (0, -1, True, 1.5, "4", None),
    ids=("zero", "negative", "boolean", "float", "string", "none"),
)
def test_accelerated_hnsw_rejects_invalid_segment_topology_values(
    name: str, value: Any
) -> None:
    topology: dict[str, Any] = {
        "premerge_segment_count": 4,
        "force_merge_segment_count": 1,
        "ram_per_thread_hard_limit_mb": 61440,
    }
    topology[name] = value

    with pytest.raises(
        ValueError, match=rf"Lucene {name} must be a positive integer"
    ):
        _build_parameters_for(
            ACCELERATED_HNSW_ALGORITHM,
            {"codec": ACCELERATED_HNSW_CODEC, **topology},
        )


@pytest.mark.parametrize(
    "value",
    (-1, 2, True, 1.5, "0", None),
    ids=("negative", "above-one", "boolean", "float", "string", "none"),
)
def test_accelerated_hnsw_force_merge_must_be_zero_or_one(
    value: Any,
) -> None:
    with pytest.raises(ValueError, match=r"must be 0 \(disabled\) or 1"):
        _build_parameters_for(
            ACCELERATED_HNSW_ALGORITHM,
            {
                "codec": ACCELERATED_HNSW_CODEC,
                "premerge_segment_count": 2,
                "force_merge_segment_count": value,
                "ram_per_thread_hard_limit_mb": 61440,
            },
        )


@pytest.mark.parametrize("force_merge_segment_count", (0, 1))
def test_accelerated_hnsw_accepts_supported_force_merge_modes(
    force_merge_segment_count: int,
) -> None:
    parameters = _build_parameters_for(
        ACCELERATED_HNSW_ALGORITHM,
        {
            "codec": ACCELERATED_HNSW_CODEC,
            "premerge_segment_count": 4,
            "force_merge_segment_count": force_merge_segment_count,
            "ram_per_thread_hard_limit_mb": 61440,
        },
    )

    assert parameters["force_merge_segment_count"] == force_merge_segment_count


@pytest.mark.parametrize("name", ("m", "beam_width"))
@pytest.mark.parametrize(
    "value",
    (0, 513, True, 1.5, "16", None),
    ids=("below-min", "above-max", "boolean", "float", "string", "none"),
)
def test_accelerated_hnsw_rejects_invalid_build_parameters(
    name: str, value: Any
) -> None:
    with pytest.raises(ValueError, match=rf"Lucene {name} must be an integer"):
        _build_parameters_for(
            ACCELERATED_HNSW_ALGORITHM,
            {
                "codec": ACCELERATED_HNSW_CODEC,
                "m": 16,
                "beam_width": 80,
                name: value,
            },
        )


def test_accelerated_hnsw_rejects_removed_writer_threads_parameter() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unsupported Lucene build parameters: cuvs_writer_threads",
    ):
        _build_parameters_for(
            ACCELERATED_HNSW_ALGORITHM,
            {
                "codec": ACCELERATED_HNSW_CODEC,
                "m": 16,
                "beam_width": 80,
                "cuvs_writer_threads": 16,
            },
        )


@pytest.mark.parametrize(
    "parameters",
    (
        {"m": 16},
        {"beam_width": 80},
    ),
    ids=("missing-beam-width", "missing-m"),
)
def test_accelerated_hnsw_requires_the_parameter_pair(
    parameters: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="require both m and beam_width"):
        _build_parameters_for(
            ACCELERATED_HNSW_ALGORITHM,
            {"codec": ACCELERATED_HNSW_CODEC, **parameters},
        )


@pytest.mark.parametrize(
    ("algorithm", "codec"),
    (
        (CPU_HNSW_ALGORITHM, CPU_HNSW_CODEC),
        (CAGRA_ALGORITHM, CAGRA_CODEC),
    ),
)
@pytest.mark.parametrize(
    "extra_parameters",
    (
        {"m": 16, "beam_width": 80},
        {
            "premerge_segment_count": 4,
            "force_merge_segment_count": 1,
            "ram_per_thread_hard_limit_mb": 61440,
        },
        {
            "num_indexing_threads": 4,
            "force_merge_segment_count": 1,
            "ram_per_thread_hard_limit_mb": 61440,
        },
        {"cuvs_writer_threads": 16},
    ),
    ids=(
        "hnsw-quality",
        "legacy-segment-topology",
        "concurrent-segment-topology",
        "cuvs-writer-threads",
    ),
)
def test_nonaccelerated_algorithms_reject_hnsw_build_parameters(
    algorithm: str, codec: str, extra_parameters: dict[str, int]
) -> None:
    with pytest.raises(
        ValueError, match="Unsupported Lucene build parameters"
    ):
        _build_parameters_for(
            algorithm,
            {"codec": codec, **extra_parameters},
        )


@pytest.mark.parametrize(
    ("parameters", "k", "expected"),
    (
        pytest.param({}, 3, {"num_candidates": 3}, id="default-to-k"),
        pytest.param(
            {"num_candidates": 7},
            3,
            {"num_candidates": 7},
            id="explicit-candidates",
        ),
    ),
)
@pytest.mark.parametrize(
    "algorithm",
    (CPU_HNSW_ALGORITHM, ACCELERATED_HNSW_ALGORITHM),
)
def test_hnsw_search_parameter_validation_accepts_candidate_budgets(
    algorithm: str,
    parameters: dict[str, int],
    k: int,
    expected: dict[str, int],
) -> None:
    assert _search_parameters(algorithm, parameters, k) == expected


@pytest.mark.parametrize(
    "parameters",
    (
        pytest.param({"num_candidates": 2}, id="below-k"),
        pytest.param({"num_candidates": True}, id="boolean"),
        pytest.param({"search_width": 16}, id="unsupported-key"),
    ),
)
@pytest.mark.parametrize(
    "algorithm",
    (CPU_HNSW_ALGORITHM, ACCELERATED_HNSW_ALGORITHM),
)
def test_hnsw_search_parameter_validation_rejects_invalid_budgets(
    algorithm: str,
    parameters: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _search_parameters(algorithm, parameters, 3)


def test_cagra_search_accepts_only_fixed_parameters_within_its_route_limit() -> (
    None
):
    assert _search_parameters(CAGRA_ALGORITHM, {}, 1024) == {
        "num_candidates": 1024
    }

    with pytest.raises(ValueError, match="accepts no search parameters"):
        _search_parameters(CAGRA_ALGORITHM, {"search_width": 16}, 10)
    with pytest.raises(ValueError, match=r"supports k <= 1024"):
        _search_parameters(CAGRA_ALGORITHM, {}, 1025)


@pytest.mark.parametrize(
    ("score", "expected_distance"),
    (
        pytest.param(1.0, 0.0, id="zero-distance"),
        pytest.param(0.5, 1.0, id="unit-distance"),
        pytest.param(0.2, 4.0, id="distance-four"),
    ),
)
def test_lucene_scores_are_inverted_to_squared_euclidean_distance(
    score: float, expected_distance: float
) -> None:
    assert _score_to_squared_euclidean(score) == pytest.approx(
        expected_distance
    )


@pytest.mark.parametrize(
    "score",
    (
        0.0,
        -1.0,
        1.0 + 2.0 * float(np.spacing(np.float32(1.0))),
        np.nan,
        np.inf,
    ),
)
def test_score_inversion_rejects_values_outside_lucenes_score_domain(
    score: float,
) -> None:
    with pytest.raises(RuntimeError, match="invalid Euclidean score"):
        _score_to_squared_euclidean(score)


def test_score_inversion_tolerates_float32_roundoff_above_one() -> None:
    score = float(np.nextafter(np.float32(1.0), np.float32(2.0)))

    assert _score_to_squared_euclidean(score) == 0.0


def test_result_metadata_is_exported_instead_of_silently_dropped(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = _dataset()
    build = backend.build(dataset, [index])
    search = backend.search(dataset, [index], k=2, batch_size=2)[0]

    write_results_to_csv(
        [build, search], dataset.name, str(tmp_path), count=2, batch_size=2
    )

    result_root = tmp_path / dataset.name / "result"
    build_csv = result_root / "build" / "lucene_cpu_hnsw,test.csv"
    search_csv = result_root / "search" / "lucene_cpu_hnsw,test,k2,bs2,raw.csv"
    assert build_csv.is_file()
    assert search_csv.is_file()
    with search_csv.open(newline="", encoding="utf-8") as stream:
        [row] = csv.DictReader(stream)
    assert row["index_name"] == CPU_HNSW_ALGORITHM
    assert float(row["build time"]) == pytest.approx(build.build_time_seconds)
    assert row["timing_contract_version"] == "1"
    assert row["latency_scope"] == "client_query"
    assert row["throughput_scope"] == "query_corpus_wall"
    assert row["execution_model"] == "serial_single_query"
    assert row["requested_batch_size"] == "2"
    assert row["effective_search_batch_size"] == "1"
    assert row["search_dispatch_kind"] == "direct_pylucene"
    assert row["client_query_count"] == "2"
    assert float(row["client_query_mean_ms"]) == pytest.approx(2.5)
    assert float(row["pylucene_search_dispatch_mean_ms"]) == pytest.approx(2.0)
    assert int(row["index_prewarm_bytes"]) > 0


def test_dry_and_failed_exports_preserve_existing_measurements(
    tmp_path: Path,
) -> None:
    runtime = RecordingRuntime()
    backend, index, _factory = _backend_and_index(
        tmp_path, CPU_HNSW_ALGORITHM, runtime
    )
    dataset = _dataset()
    build = backend.build(dataset, [index])
    search = backend.search(dataset, [index], k=2, batch_size=2)[0]
    assert build.success and search.success
    write_results_to_csv(
        [build, search], dataset.name, str(tmp_path), count=2, batch_size=2
    )

    result_root = tmp_path / dataset.name / "result"
    exported_files = (
        result_root / "build" / "lucene_cpu_hnsw,test.csv",
        result_root / "search" / "lucene_cpu_hnsw,test,k2,bs2,raw.csv",
        result_root / "search" / "lucene_cpu_hnsw,test,k2,bs2,throughput.csv",
        result_root / "search" / "lucene_cpu_hnsw,test,k2,bs2,latency.csv",
    )
    original_contents = {path: path.read_bytes() for path in exported_files}

    dry_build = backend.build(dataset, [index], dry_run=True)
    dry_search = backend.search(dataset, [index], k=2, dry_run=True)[0]
    runtime.build_error = RuntimeError("replacement build failed")
    failed_build = backend.build(dataset, [index], force=True)
    runtime.search_error = RuntimeError("replacement search failed")
    failed_search = backend.search(dataset, [index], k=2)[0]
    assert not failed_build.success
    assert not failed_search.success

    write_results_to_csv(
        [dry_build, dry_search, failed_build, failed_search],
        dataset.name,
        str(tmp_path),
        count=2,
        batch_size=2,
    )

    assert {path: path.read_bytes() for path in exported_files} == (
        original_contents
    )


def test_config_loader_maps_each_algorithm_to_its_codec_and_requirement(
    tmp_path: Path,
) -> None:
    dataset_configuration = tmp_path / "datasets.yaml"
    dataset_configuration.write_text(
        "- name: tiny-l2\n  distance: euclidean\n  dims: 2\n",
        encoding="utf-8",
    )
    loader = LuceneConfigLoader()

    _dataset_config, configurations = loader.load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(dataset_configuration),
        algorithms=(
            f"{CPU_HNSW_ALGORITHM},{ACCELERATED_HNSW_ALGORITHM},"
            f"{CAGRA_ALGORITHM}"
        ),
        groups="test",
    )
    by_algorithm = {
        configuration.indexes[0].algo: configuration
        for configuration in configurations
    }

    assert set(by_algorithm) == {
        CPU_HNSW_ALGORITHM,
        ACCELERATED_HNSW_ALGORITHM,
        CAGRA_ALGORITHM,
    }
    assert by_algorithm[CPU_HNSW_ALGORITHM].indexes[0].build_param == {
        "codec": CPU_HNSW_CODEC
    }
    assert by_algorithm[CAGRA_ALGORITHM].indexes[0].build_param == {
        "codec": CAGRA_CODEC
    }
    assert by_algorithm[ACCELERATED_HNSW_ALGORITHM].indexes[0].build_param == {
        "codec": ACCELERATED_HNSW_CODEC,
        "m": 32,
        "beam_width": 32,
    }
    assert by_algorithm[ACCELERATED_HNSW_ALGORITHM].index_name == (
        f"{ACCELERATED_HNSW_ALGORITHM}_test.m32.beam_width32"
    )
    assert (
        by_algorithm[CPU_HNSW_ALGORITHM].backend_config["requires_cuvs"]
        is False
    )
    assert (
        by_algorithm[ACCELERATED_HNSW_ALGORITHM].backend_config[
            "requires_cuvs"
        ]
        is True
    )
    assert (
        by_algorithm[CAGRA_ALGORITHM].backend_config["requires_cuvs"] is True
    )
    assert (
        by_algorithm[CPU_HNSW_ALGORITHM].backend_config["include_cuvs"] is True
    )
    assert (
        by_algorithm[ACCELERATED_HNSW_ALGORITHM].backend_config["include_cuvs"]
        is True
    )
    assert by_algorithm[CAGRA_ALGORITHM].backend_config["include_cuvs"] is True
    assert by_algorithm[CPU_HNSW_ALGORITHM].backend_config["group"] == "test"
    assert (
        by_algorithm[ACCELERATED_HNSW_ALGORITHM].backend_config["group"]
        == "test"
    )
    assert by_algorithm[CAGRA_ALGORITHM].backend_config["group"] == "test"


@pytest.mark.parametrize("force_merge_segment_count", (0, 1))
def test_config_loader_uses_canonical_segment_topology_label_order(
    tmp_path: Path,
    force_merge_segment_count: int,
) -> None:
    dataset_configuration = tmp_path / "datasets.yaml"
    dataset_configuration.write_text(
        "- name: tiny-l2\n  distance: euclidean\n  dims: 2\n",
        encoding="utf-8",
    )
    algorithm_configuration = tmp_path / "accelerated.yaml"
    algorithm_configuration.write_text(
        f"""\
name: {ACCELERATED_HNSW_ALGORITHM}
groups:
  segmented:
    build:
      ram_per_thread_hard_limit_mb: [61440]
      force_merge_segment_count: [{force_merge_segment_count}]
      beam_width: [80]
      codec: ["{ACCELERATED_HNSW_CODEC}"]
      premerge_segment_count: [4]
      m: [16]
    search: {{}}
""",
        encoding="utf-8",
    )

    _dataset_config, [configuration] = LuceneConfigLoader().load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(dataset_configuration),
        algorithm_configuration=str(algorithm_configuration),
        algorithms=ACCELERATED_HNSW_ALGORITHM,
        groups="segmented",
    )

    expected_parameters = {
        "codec": ACCELERATED_HNSW_CODEC,
        "m": 16,
        "beam_width": 80,
        "premerge_segment_count": 4,
        "force_merge_segment_count": force_merge_segment_count,
        "ram_per_thread_hard_limit_mb": 61440,
    }
    expected_name = (
        f"{ACCELERATED_HNSW_ALGORITHM}_segmented"
        ".m16.beam_width80.premerge_segment_count4"
        f".force_merge_segment_count{force_merge_segment_count}"
        ".ram_per_thread_hard_limit_mb61440"
    )
    assert configuration.indexes[0].build_param == expected_parameters
    assert configuration.index_name == expected_name
    assert configuration.index_path == (
        tmp_path / "tiny-l2" / "index" / expected_name
    )


@pytest.mark.parametrize("force_merge_segment_count", (0, 1))
def test_config_loader_uses_canonical_partitioned_topology_label_order(
    tmp_path: Path,
    force_merge_segment_count: int,
) -> None:
    dataset_configuration = tmp_path / "datasets.yaml"
    dataset_configuration.write_text(
        "- name: tiny-l2\n  distance: euclidean\n  dims: 2\n",
        encoding="utf-8",
    )
    algorithm_configuration = tmp_path / "accelerated.yaml"
    algorithm_configuration.write_text(
        f"""\
name: {ACCELERATED_HNSW_ALGORITHM}
groups:
  partitioned:
    build:
      ram_per_thread_hard_limit_mb: [61440]
      force_merge_segment_count: [{force_merge_segment_count}]
      beam_width: [80]
      codec: ["{ACCELERATED_HNSW_CODEC}"]
      num_indexing_threads: [4]
      m: [16]
    search: {{}}
""",
        encoding="utf-8",
    )

    _dataset_config, [configuration] = LuceneConfigLoader().load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(dataset_configuration),
        algorithm_configuration=str(algorithm_configuration),
        algorithms=ACCELERATED_HNSW_ALGORITHM,
        groups="partitioned",
    )

    expected_parameters = {
        "codec": ACCELERATED_HNSW_CODEC,
        "m": 16,
        "beam_width": 80,
        "num_indexing_threads": 4,
        "force_merge_segment_count": force_merge_segment_count,
        "ram_per_thread_hard_limit_mb": 61440,
    }
    expected_name = (
        f"{ACCELERATED_HNSW_ALGORITHM}_partitioned"
        ".m16.beam_width80.num_indexing_threads4"
        f".force_merge_segment_count{force_merge_segment_count}"
        ".ram_per_thread_hard_limit_mb61440"
    )
    assert configuration.indexes[0].build_param == expected_parameters
    assert configuration.index_name == expected_name
    assert configuration.index_path == (
        tmp_path / "tiny-l2" / "index" / expected_name
    )


def test_config_loader_rejects_dataset_names_that_can_escape_the_root(
    tmp_path: Path,
) -> None:
    dataset_configuration = tmp_path / "datasets.yaml"
    dataset_configuration.write_text(
        "- name: ../escape\n  distance: euclidean\n  dims: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsafe Lucene dataset"):
        LuceneConfigLoader().load(
            dataset="../escape",
            dataset_path=str(tmp_path),
            dataset_configuration=str(dataset_configuration),
            algorithms=CPU_HNSW_ALGORITHM,
            groups="test",
        )


def test_cpu_only_config_does_not_include_optional_cuvs_artifacts(
    tmp_path: Path,
) -> None:
    dataset_configuration = tmp_path / "datasets.yaml"
    dataset_configuration.write_text(
        "- name: tiny-l2\n  distance: euclidean\n  dims: 2\n",
        encoding="utf-8",
    )

    _dataset_config, [configuration] = LuceneConfigLoader().load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(dataset_configuration),
        algorithms=CPU_HNSW_ALGORITHM,
        groups="test",
    )

    assert configuration.backend_config["requires_cuvs"] is False
    assert configuration.backend_config["include_cuvs"] is False
