#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Explicitly selected 3 GiB tests for PyLucene document ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from _lucene_log_capture import (
    GRAPH_CLAMP_WARNINGS,
    case_used_cpu_hnsw_fallback,
    lucene_case_output,
    lucene_log_case,
    require_observable_lucene_warnings,
)
from cuvs_bench.backends._lucene_compat import LUCENE_COMPATIBILITY
from cuvs_bench.backends._lucene_runtime import (
    ACCELERATED_HNSW_CODEC,
    CAGRA_CODEC,
)
from cuvs_bench.backends._lucene_runtime_config import maven_artifact_version
from cuvs_bench.backends.base import BuildResult, Dataset, SearchResult
from cuvs_bench.backends.lucene import (
    ACCELERATED_HNSW_ALGORITHM,
    CAGRA_ALGORITHM,
    LuceneBackend,
)
from cuvs_bench.orchestrator.config_loaders import IndexConfig

pytestmark = [
    pytest.mark.lucene_e2e,
    pytest.mark.lucene_large_segment_e2e,
    pytest.mark.filterwarnings(
        "ignore:builtin type .* has no __module__ attribute:DeprecationWarning"
    ),
]

_DIMENSIONS = 1_024
_INTRINSIC_DIMENSIONS = 8
_COORDINATE_BOUND = 0.125
_GIB = 1024**3
_PAYLOAD_BYTES = 3 * _GIB
_VECTOR_COUNT = _PAYLOAD_BYTES // (_DIMENSIONS * np.dtype(np.float32).itemsize)
_CHUNK_ROWS = 2_048
_TOP_K = 10
_HEADER_BYTES = 8
_MINIMUM_RECALL = 0.75
_MINIMUM_FREE_DISK_BYTES = 16 * _GIB
_MINIMUM_AVAILABLE_MEMORY_BYTES = 16 * _GIB
_ARTIFACT_VERSION = maven_artifact_version()
_MANIFEST_FILE = ".cuvs-bench-lucene.json"
_JVM_ARGS = ("-Xmx12g",)
_EXAMPLE_HNSW_M = 16
_EXAMPLE_HNSW_BEAM_WIDTH = 80
_EXAMPLE_UNSUPPORTED_RAM_LIMIT_MB = 6144

assert (
    _VECTOR_COUNT * _DIMENSIONS * np.dtype(np.float32).itemsize
    == _PAYLOAD_BYTES
)
assert _DIMENSIONS % _INTRINSIC_DIMENSIONS == 0


@dataclass(frozen=True)
class _LargeFbinCase:
    path: Path
    query: np.ndarray
    query_ids: np.ndarray
    neighbors: np.ndarray
    squared_distances: np.ndarray
    payload_sha256: str


def _available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _require_generation_resources(directory: Path) -> None:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail(
            "The large Lucene suite must run without pytest-xdist so its "
            "3 GiB fixture is generated only once"
        )
    free_disk = shutil.disk_usage(directory).free
    if free_disk < _MINIMUM_FREE_DISK_BYTES:
        pytest.fail(
            "The explicitly selected large Lucene suite needs at least "
            f"{_MINIMUM_FREE_DISK_BYTES // _GIB} GiB of free temporary "
            f"storage; found {free_disk / _GIB:.1f} GiB. Select a larger "
            "local filesystem with pytest --basetemp."
        )
    available_memory = _available_memory_bytes()
    if (
        available_memory is not None
        and available_memory < _MINIMUM_AVAILABLE_MEMORY_BYTES
    ):
        pytest.fail(
            "The explicitly selected PyLucene document-ingest suite needs "
            "at least "
            f"{_MINIMUM_AVAILABLE_MEMORY_BYTES // _GIB} GiB of available "
            f"host memory; found {available_memory / _GIB:.1f} GiB"
        )


def _nearest_neighbors(
    best_ids: np.ndarray,
    best_distances: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.concatenate((best_ids, candidate_ids))
    distances = np.concatenate((best_distances, candidate_distances))
    order = np.lexsort((ids, distances))[:_TOP_K]
    return ids[order], distances[order]


def _capacity_query_ids() -> np.ndarray:
    """Query both sides of the payload address boundary and the final vector."""
    bytes_per_vector = _DIMENSIONS * np.dtype(np.float32).itemsize
    first_row_at_two_gib = 2 * _GIB // bytes_per_vector
    return np.asarray(
        [
            first_row_at_two_gib - 1,
            first_row_at_two_gib,
            first_row_at_two_gib + 1,
            _VECTOR_COUNT - 1,
        ],
        dtype=np.int64,
    )


def _write_large_fbin(path: Path) -> _LargeFbinCase:
    rng = np.random.default_rng(2624)
    # Tile an eight-dimensional cloud to materialize the full 3 GiB payload.
    # This tests physical capacity, not high-intrinsic-dimensional ANN quality.
    intrinsic_vectors = rng.uniform(
        -_COORDINATE_BOUND,
        _COORDINATE_BOUND,
        size=(_VECTOR_COUNT, _INTRINSIC_DIMENSIONS),
    ).astype(np.float32)
    coordinate_repetitions = _DIMENSIONS // _INTRINSIC_DIMENSIONS
    query_ids = _capacity_query_ids()
    queries = np.tile(
        intrinsic_vectors[query_ids], (1, coordinate_repetitions)
    )
    best_ids = [np.empty(0, dtype=np.int64) for _ in queries]
    best_distances = [np.empty(0, dtype=np.float32) for _ in queries]
    digest = hashlib.sha256()

    with path.open("wb") as stream:
        stream.write(
            np.asarray([_VECTOR_COUNT, _DIMENSIONS], dtype="<u4").tobytes()
        )
        for start in range(0, _VECTOR_COUNT, _CHUNK_ROWS):
            row_count = min(_CHUNK_ROWS, _VECTOR_COUNT - start)
            expanded_chunk = np.tile(
                intrinsic_vectors[start : start + row_count],
                (1, coordinate_repetitions),
            )
            payload = expanded_chunk.view(np.uint8)
            stream.write(payload)
            digest.update(payload)

            candidate_ids = np.arange(start, start + row_count, dtype=np.int64)
            for query_number, query in enumerate(queries):
                delta = expanded_chunk - query
                squared_distances = np.einsum(
                    "ij,ij->i", delta, delta, dtype=np.float32
                )
                best_ids[query_number], best_distances[query_number] = (
                    _nearest_neighbors(
                        best_ids[query_number],
                        best_distances[query_number],
                        candidate_ids,
                        squared_distances,
                    )
                )

    neighbors = np.asarray(best_ids, dtype=np.int32)
    np.testing.assert_array_equal(neighbors[:, 0], query_ids)
    assert path.stat().st_size == _HEADER_BYTES + _PAYLOAD_BYTES
    return _LargeFbinCase(
        path=path,
        query=queries,
        query_ids=query_ids,
        neighbors=neighbors,
        squared_distances=np.asarray(best_distances, dtype=np.float32),
        payload_sha256=digest.hexdigest(),
    )


def _backend_and_index(
    root: Path,
    algorithm: str,
    codec: str,
    search_params: list[dict[str, int]],
) -> tuple[LuceneBackend, IndexConfig]:
    index_path = root / algorithm / "index"
    return (
        LuceneBackend(
            {
                "name": algorithm,
                "algo": algorithm,
                "codec": codec,
                "group": "large-python-ingest-test",
                "index_root": str(index_path.parent),
                "requires_cuvs": True,
                "include_cuvs": True,
                "jvm_args": list(_JVM_ARGS),
            }
        ),
        IndexConfig(
            name=algorithm,
            algo=algorithm,
            build_param={"codec": codec},
            search_params=search_params,
            file=str(index_path),
        ),
    )


@pytest.fixture(scope="module")
def _large_suite_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("lucene-large-python-suite")
    _require_generation_resources(root)
    return root


@pytest.fixture(scope="module")
def _verified_warning_controls(_large_suite_root: Path) -> None:
    # This dependency cannot be omitted by selecting only one capacity case.
    # A failed control stops the 3 GiB allocation and invalidates absence evidence.
    require_observable_lucene_warnings(_large_suite_root)


@pytest.fixture(scope="module")
def _verified_gpu_runtime(
    _large_suite_root: Path, _verified_warning_controls
) -> None:
    """Fail before allocating 3 GiB when the live cuVS path is unavailable."""
    root = _large_suite_root / "runtime-check"
    root.mkdir()
    backend, index = _backend_and_index(
        root, CAGRA_ALGORITHM, CAGRA_CODEC, [{}]
    )
    rng = np.random.default_rng(174)
    vectors = rng.standard_normal((512, 128)).astype(np.float32)
    dataset = Dataset(
        name="lucene-large-runtime-check",
        training_vectors=vectors,
        query_vectors=vectors[:1].copy(),
        distance_metric="euclidean",
    )
    index.build_param.update(
        {
            "premerge_segment_count": 1,
            "force_merge_segment_count": 0,
            "ram_per_thread_hard_limit_mb": _EXAMPLE_UNSUPPORTED_RAM_LIMIT_MB,
            "allow_unsupported_lucene_ram_limit": True,
        }
    )
    try:
        build = backend.build(dataset, [index], force=True)
        assert build.success, build.error_message
        assert build.metadata["persisted_index_kind"] == "gpu_cagra_only"
        assert build.metadata["applied_ram_per_thread_hard_limit_mb"] == (
            _EXAMPLE_UNSUPPORTED_RAM_LIMIT_MB
        )
        assert build.metadata["ram_per_thread_hard_limit_application"] == (
            "unsupported_field_override"
        )
    finally:
        if Path(index.file).is_dir():
            shutil.rmtree(Path(index.file))


@pytest.fixture(scope="module")
def large_fbin_case(
    _large_suite_root: Path,
    _verified_gpu_runtime,
) -> Iterator[_LargeFbinCase]:
    root = _large_suite_root / "dataset"
    root.mkdir()
    path = root / "base.fbin"
    partial = root / "base.fbin.partial"
    try:
        generated = _write_large_fbin(partial)
        partial.replace(path)
        yield _LargeFbinCase(
            path=path,
            query=generated.query,
            query_ids=generated.query_ids,
            neighbors=generated.neighbors,
            squared_distances=generated.squared_distances,
            payload_sha256=generated.payload_sha256,
        )
    finally:
        partial.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def _assert_artifact_provenance(metadata: dict, role: str) -> None:
    expected = {
        "cuvs_java": f"com.nvidia.cuvs:cuvs-java:{_ARTIFACT_VERSION}",
        "cuvs_lucene": (
            f"com.nvidia.cuvs.lucene:{LUCENE_COMPATIBILITY.artifact_id}:{_ARTIFACT_VERSION}"
        ),
    }
    for artifact, coordinates in expected.items():
        assert metadata[f"{role}_{artifact}_coordinates"] == coordinates
        jar_path = Path(metadata[f"{role}_{artifact}_jar_path"])
        assert jar_path.is_file(), f"Missing {role} {artifact} JAR: {jar_path}"
        digest = metadata[f"{role}_{artifact}_jar_sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", digest), (
            f"Invalid {role} {artifact} SHA-256: {digest!r}"
        )


def _recall(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.mean(
            [
                len(set(row).intersection(truth)) / expected.shape[1]
                for row, truth in zip(actual, expected)
            ]
        )
    )


def _squared_distances_for_hits(
    source: Path, query: np.ndarray, document_ids: np.ndarray
) -> np.ndarray:
    distances = []
    with source.open("rb") as stream:
        for document_id in document_ids:
            stream.seek(
                _HEADER_BYTES
                + int(document_id)
                * _DIMENSIONS
                * np.dtype(np.float32).itemsize
            )
            vector = np.frombuffer(
                stream.read(_DIMENSIONS * np.dtype(np.float32).itemsize),
                dtype=np.float32,
            )
            if vector.size != _DIMENSIONS:
                raise AssertionError(
                    f"FBIN row {int(document_id)} is unexpectedly truncated"
                )
            delta = vector - query
            distances.append(float(np.dot(delta, delta)))
    return np.asarray([distances], dtype=np.float32)


def _assert_single_segment_build_without_merging(
    build: BuildResult, algorithm: str, codec: str
) -> None:
    assert build.index_size_bytes > 2 * _GIB
    assert build.metadata["codec"] == codec
    assert build.metadata["persisted_index_kind"] == (
        "hnsw" if algorithm == ACCELERATED_HNSW_ALGORITHM else "gpu_cagra_only"
    )
    assert build.metadata["build_route_policy"] == (
        "gpu_cagra_or_cpu_hnsw_fallback"
        if algorithm == ACCELERATED_HNSW_ALGORITHM
        else "gpu_cagra"
    )
    assert build.metadata["segment_count"] == 1
    assert build.metadata["field_count"] == 1
    assert build.metadata["vector_count"] == _VECTOR_COUNT
    assert build.metadata["dimensions"] == _DIMENSIONS
    assert build.metadata["requested_premerge_segment_count"] == 1
    assert build.metadata["observed_premerge_segment_count"] == 1
    assert build.metadata["requested_force_merge_segment_count"] == 0
    assert build.metadata["premerge_segment_vector_counts"] == (
        f"[{_VECTOR_COUNT}]"
    )
    assert build.metadata["max_buffered_docs"] == _VECTOR_COUNT + 1
    assert build.metadata["runtime_force_merge_seconds"] == 0.0
    assert build.metadata["ingest_merge_policy"] == "NoMergePolicy"
    assert build.metadata["final_merge_policy"] is None
    assert build.metadata["applied_ram_per_thread_hard_limit_mb"] == (
        _EXAMPLE_UNSUPPORTED_RAM_LIMIT_MB
    )
    assert build.metadata["ram_per_thread_hard_limit_application"] == (
        "unsupported_field_override"
    )
    _assert_artifact_provenance(build.metadata, "build_runtime")


def _assert_persisted_dataset_identity(
    index_path: Path, dataset_case: _LargeFbinCase
) -> None:
    assert dataset_case.path.stat().st_size == _HEADER_BYTES + _PAYLOAD_BYTES
    manifest = json.loads(
        (index_path / _MANIFEST_FILE).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 3
    assert manifest["dataset"]["vector_count"] == _VECTOR_COUNT
    assert manifest["dataset"]["dimensions"] == _DIMENSIONS
    assert manifest["dataset"]["sha256"] == dataset_case.payload_sha256
    assert manifest["dataset"]["source"]["size"] == (
        _HEADER_BYTES + _PAYLOAD_BYTES
    )
    assert manifest["segment_count"] == 1


def _assert_search_matches_large_dataset(
    result: SearchResult,
    dataset_case: _LargeFbinCase,
    expected_search_route: str,
) -> None:
    assert result.success, result.error_message
    assert result.metadata["expected_search_route"] == expected_search_route
    assert result.neighbors.shape == (len(dataset_case.query_ids), _TOP_K)
    # Aggregate recall must not excuse a missing boundary or tail self-match.
    np.testing.assert_array_equal(
        result.neighbors[:, 0], dataset_case.query_ids
    )
    assert all(len(set(row)) == _TOP_K for row in result.neighbors.tolist())
    assert np.all(result.neighbors >= 0)
    assert np.all(result.neighbors < _VECTOR_COUNT)
    assert _recall(result.neighbors, dataset_case.neighbors) >= _MINIMUM_RECALL
    exact_distances = np.concatenate(
        [
            _squared_distances_for_hits(dataset_case.path, query, hits)
            for query, hits in zip(dataset_case.query, result.neighbors)
        ]
    )
    np.testing.assert_allclose(
        result.distances, exact_distances, rtol=1e-4, atol=1e-4
    )
    _assert_artifact_provenance(result.metadata, "build_runtime")
    _assert_artifact_provenance(result.metadata, "search_runtime")


@pytest.mark.parametrize(
    ("algorithm", "codec", "search_params", "expected_search_route"),
    (
        pytest.param(
            ACCELERATED_HNSW_ALGORITHM,
            ACCELERATED_HNSW_CODEC,
            [{"num_candidates": 256}],
            "cpu_hnsw",
            id="cagra-hnsw-single-3gib-segment",
        ),
        pytest.param(
            CAGRA_ALGORITHM,
            CAGRA_CODEC,
            [{}],
            "gpu_cagra",
            id="cagra-gpu-single-3gib-segment",
        ),
    ),
)
def test_pylucene_document_ingest_builds_one_segment_above_two_gibibytes(
    tmp_path: Path,
    capfd,
    request,
    large_fbin_case: _LargeFbinCase,
    algorithm: str,
    codec: str,
    search_params: list[dict[str, int]],
    expected_search_route: str,
) -> None:
    mapping = np.memmap(
        large_fbin_case.path,
        mode="r",
        dtype=np.float32,
        offset=_HEADER_BYTES,
        shape=(_VECTOR_COUNT, _DIMENSIONS),
    )
    array_view = np.asarray(mapping)
    assert mapping.flags.c_contiguous
    assert np.shares_memory(mapping, array_view)
    assert np.shares_memory(mapping, np.ascontiguousarray(array_view))
    dataset = Dataset(
        name=f"lucene-large-python-ingest-{algorithm}",
        training_vectors=mapping,
        query_vectors=large_fbin_case.query,
        groundtruth_neighbors=large_fbin_case.neighbors,
        groundtruth_distances=large_fbin_case.squared_distances,
        distance_metric="euclidean",
        base_file=str(large_fbin_case.path),
    )
    backend, index = _backend_and_index(
        tmp_path, algorithm, codec, search_params
    )
    if algorithm == ACCELERATED_HNSW_ALGORITHM:
        index.build_param.update(
            {
                "m": _EXAMPLE_HNSW_M,
                "beam_width": _EXAMPLE_HNSW_BEAM_WIDTH,
            }
        )
    index.build_param.update(
        {
            "premerge_segment_count": 1,
            "force_merge_segment_count": 0,
            "ram_per_thread_hard_limit_mb": _EXAMPLE_UNSUPPORTED_RAM_LIMIT_MB,
            "allow_unsupported_lucene_ram_limit": True,
        }
    )
    case_name = request.node.nodeid
    capfd.readouterr()
    mapping_is_open = True

    try:
        with lucene_log_case(case_name):
            build = backend.build(dataset, [index], force=True)
            assert build.success, build.error_message
            dataset.training_vectors = None
            mapping._mmap.close()
            mapping_is_open = False
            [result] = backend.search(dataset, [index], k=_TOP_K, batch_size=1)

        captured = capfd.readouterr()
        combined_output = captured.out + captured.err
        if algorithm == ACCELERATED_HNSW_ALGORITHM:
            assert not case_used_cpu_hnsw_fallback(captured.err, case_name), (
                f"{case_name} used Lucene's CPU HNSW writer fallback:\n"
                f"{lucene_case_output(captured.err, case_name)}"
            )
        # Persisted CAGRA-only verification rejects brute-force payloads without debug logging.
        for warning in GRAPH_CLAMP_WARNINGS:
            assert warning.casefold() not in combined_output.casefold()

        _assert_single_segment_build_without_merging(build, algorithm, codec)
        _assert_persisted_dataset_identity(Path(index.file), large_fbin_case)
        _assert_search_matches_large_dataset(
            result, large_fbin_case, expected_search_route
        )
    finally:
        dataset.training_vectors = None
        if mapping_is_open:
            mapping._mmap.close()
        if Path(index.file).is_dir():
            shutil.rmtree(Path(index.file))
