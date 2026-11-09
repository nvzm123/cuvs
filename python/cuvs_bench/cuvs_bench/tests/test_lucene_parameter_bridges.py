#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Python/JCC request wiring; real Java construction is covered separately."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cuvs_bench.backends._lucene_runtime import (
    ACCELERATED_HNSW_CODEC,
    CAGRA_CODEC,
    CPU_HNSW_CODEC,
    LuceneRuntime,
)
from _lucene_test_support import RecordingRuntime, _backend_and_index, _dataset
from cuvs_bench.backends.lucene import (
    ACCELERATED_HNSW_ALGORITHM,
    CAGRA_ALGORITHM,
)


class FakeJavaMap(dict):
    def put(self, key, value):
        self[key] = value


def _runtime_with_fake_scalar_boxing():
    runtime = object.__new__(LuceneRuntime)
    runtime.HashMap = FakeJavaMap
    for name in ("Integer", "Double", "Boolean"):
        # Tagged tuples are deliberately fake, not real Java boxed objects.
        setattr(
            runtime,
            name,
            SimpleNamespace(valueOf=lambda value, tag=name: (tag, value)),
        )
    return runtime


def test_parameter_requests_preserve_boolean_integer_and_fraction_types():
    runtime = _runtime_with_fake_scalar_boxing()

    request = runtime._parameter_request(
        {
            "writer_threads": 2,
            "kmeans_trainset_fraction": 0.25,
            "force_random_rotation": True,
            "graph_build_algo": "ivf_pq",
        }
    )

    assert request == {
        "writer_threads": ("Integer", 2),
        "kmeans_trainset_fraction": ("Double", 0.25),
        "force_random_rotation": ("Boolean", True),
        "graph_build_algo": "ivf_pq",
    }


@pytest.mark.parametrize(
    "codec", [CPU_HNSW_CODEC, ACCELERATED_HNSW_CODEC, CAGRA_CODEC]
)
def test_topology_only_build_does_not_require_a_parameter_factory(codec):
    runtime = object.__new__(LuceneRuntime)
    fake_default_codec = object()
    runtime.resolve_codec = Mock(return_value=fake_default_codec)
    runtime.resolve_configured_hnsw_codec = Mock()
    runtime.resolve_configured_cagra_codec = Mock()

    selected = runtime._resolve_build_codec(
        codec, {"premerge_segment_count": 1, "force_merge_segment_count": 0}
    )

    assert selected is fake_default_codec
    runtime.resolve_codec.assert_called_once_with(codec)
    runtime.resolve_configured_hnsw_codec.assert_not_called()
    runtime.resolve_configured_cagra_codec.assert_not_called()


def test_hnsw_codec_request_keeps_topology_out_of_algorithm_controls():
    runtime = object.__new__(LuceneRuntime)
    fake_configured_codec = object()
    runtime.resolve_configured_hnsw_codec = Mock(
        return_value=fake_configured_codec
    )

    selected = runtime._resolve_build_codec(
        ACCELERATED_HNSW_CODEC,
        {
            "m": 16,
            "beam_width": 100,
            "hnsw_layers": 3,
            "writer_threads": 2,
            "premerge_segment_count": 4,
        },
    )

    assert selected is fake_configured_codec
    runtime.resolve_configured_hnsw_codec.assert_called_once_with(
        16, 100, {"hnsw_layers": 3, "writer_threads": 2}
    )


def test_custom_cagra_build_reaches_the_gpu_codec_factory():
    runtime = object.__new__(LuceneRuntime)
    fake_configured_codec = object()
    runtime.resolve_configured_cagra_codec = Mock(
        return_value=fake_configured_codec
    )
    algorithm_controls = {
        "graph_build_algo": "ivf_pq",
        "kmeans_trainset_fraction": 0.25,
    }

    selected = runtime._resolve_build_codec(CAGRA_CODEC, algorithm_controls)

    assert selected is fake_configured_codec
    runtime.resolve_configured_cagra_codec.assert_called_once_with(
        algorithm_controls
    )


def test_configured_query_forwards_the_java_vector_and_every_search_control():
    runtime = _runtime_with_fake_scalar_boxing()
    fake_java_vector = object()
    fake_java_query = object()
    runtime._java_cagra_query_factory = Mock()
    runtime._java_cagra_query_factory.apply.return_value = fake_java_query
    runtime.Query = SimpleNamespace(cast_=lambda value: value)

    query = runtime._configured_cagra_query(
        fake_java_vector,
        10,
        {
            "itopk": 64,
            "search_width": 16,
            "max_iterations": 20,
            "algo": "multi_cta",
            "thread_block_size": 128,
        },
    )

    assert query is fake_java_query
    sent_request = runtime._java_cagra_query_factory.apply.call_args.args[0]
    assert sent_request == {
        "field": "vector",
        "target": fake_java_vector,
        "k": ("Integer", 10),
        "itopk": ("Integer", 64),
        "search_width": ("Integer", 16),
        "max_iterations": ("Integer", 20),
        "algo": "multi_cta",
        "thread_block_size": ("Integer", 128),
    }


@pytest.mark.parametrize(
    "gpu_controls,num_entry_points",
    [
        pytest.param(None, 0, id="stock-hnsw"),
        pytest.param({}, 0, id="gpu-cagra-defaults"),
        pytest.param({"search_width": 32}, 0, id="gpu-cagra"),
        pytest.param(None, 42, id="random-entry-hnsw"),
    ],
)
def test_query_execution_uses_only_the_requested_search_factory(
    gpu_controls,
    num_entry_points,
):
    runtime = object.__new__(LuceneRuntime)
    fake_java_vector = object()
    fake_default_query = object()
    fake_gpu_query = object()
    fake_random_entry_query = object()
    runtime._java_vector = Mock(return_value=fake_java_vector)
    runtime.KnnFloatVectorQuery = Mock(return_value=fake_default_query)
    runtime._configured_cagra_query = Mock(return_value=fake_gpu_query)
    runtime._base_layer_hnsw_query = Mock(return_value=fake_random_entry_query)
    runtime._search = Mock(return_value=(SimpleNamespace(scoreDocs=[]), 1, 1))
    runtime._materialize_hits = Mock(return_value=[])
    fake_searcher, fake_reader = object(), object()

    runtime._search_one(
        fake_searcher,
        fake_reader,
        np.zeros(2, dtype=np.float32),
        10,
        64,
        gpu_controls,
        num_entry_points=num_entry_points,
    )

    if num_entry_points:
        runtime._base_layer_hnsw_query.assert_called_once_with(
            fake_java_vector, 64, 42
        )
        runtime.KnnFloatVectorQuery.assert_not_called()
        runtime._configured_cagra_query.assert_not_called()
        expected_query = fake_random_entry_query
    elif gpu_controls is None:
        runtime.KnnFloatVectorQuery.assert_called_once_with(
            "vector", fake_java_vector, 64
        )
        runtime._configured_cagra_query.assert_not_called()
        expected_query = fake_default_query
    else:
        runtime._configured_cagra_query.assert_called_once_with(
            fake_java_vector, 10, gpu_controls
        )
        runtime.KnnFloatVectorQuery.assert_not_called()
        expected_query = fake_gpu_query
    if not num_entry_points:
        runtime._base_layer_hnsw_query.assert_not_called()
    runtime._search.assert_called_once_with(fake_searcher, expected_query, 10)


def test_random_entry_factory_receives_the_java_vector_without_conversion():
    runtime = _runtime_with_fake_scalar_boxing()
    fake_java_vector = object()
    fake_java_query = object()
    runtime._java_base_layer_query_factory = Mock()
    runtime._java_base_layer_query_factory.apply.return_value = fake_java_query
    runtime.Query = SimpleNamespace(cast_=lambda value: value)

    query = runtime._base_layer_hnsw_query(fake_java_vector, 100, 42)

    assert query is fake_java_query
    sent_request = runtime._java_base_layer_query_factory.apply.call_args.args[
        0
    ]
    assert sent_request == {
        "field": "vector",
        "target": fake_java_vector,
        "k": ("Integer", 100),
        "num_entry_points": ("Integer", 42),
    }


def test_random_entry_sweep_reaches_search_without_rebuilding_the_index(
    tmp_path,
):
    fake_runtime = RecordingRuntime()
    backend, index, _ = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    index.search_params = [
        {"num_candidates": 4},
        {"num_candidates": 4, "random_entry_points": True},
        {
            "num_candidates": 4,
            "random_entry_points": True,
            "num_entry_points": 7,
        },
    ]
    dataset = _dataset()
    build = backend.build(dataset, [index])
    assert build.success, build.error_message

    results = backend.search(dataset, [index], k=2)

    assert all(result.success for result in results), [
        result.error_message for result in results
    ]
    assert [
        call.get("num_entry_points", 0) for call in fake_runtime.search_calls
    ] == [0, 42, 7]
    assert results[1].search_params == [
        {
            "num_candidates": 4,
            "random_entry_points": True,
            "num_entry_points": 42,
        }
    ]
    assert results[2].search_params == [index.search_params[2]]
    assert len(fake_runtime.build_calls) == 1


def test_requested_factory_failure_is_not_replaced_by_default_search():
    runtime = object.__new__(LuceneRuntime)
    runtime.Class = SimpleNamespace(
        forName=Mock(side_effect=RuntimeError("missing requested class"))
    )

    with pytest.raises(
        RuntimeError, match="require an updated thin cuVS-Lucene JAR"
    ) as failure:
        runtime._load_parameter_factory("fake.missing.Factory")

    assert "missing requested class" in str(failure.value)


def test_cagra_query_sweep_reaches_runtime_and_result_metadata_without_rebuilding(
    tmp_path,
):
    fake_runtime = RecordingRuntime(pylucene_version="i-am-fake-data")
    backend, index, _factory = _backend_and_index(
        tmp_path, CAGRA_ALGORITHM, fake_runtime
    )
    query_plans = [
        {"itopk": 32, "search_width": 16},
        {"itopk": 64, "search_width": 32, "algo": "multi_cta"},
    ]
    index.search_params = query_plans
    dataset = _dataset()
    build = backend.build(dataset, [index])
    assert build.success, build.error_message

    results = backend.search(dataset, [index], k=2)

    assert all(result.success for result in results), [
        result.error_message for result in results
    ]
    assert [
        call["cagra_search_parameters"] for call in fake_runtime.search_calls
    ] == query_plans
    assert [result.search_params for result in results] == [
        [plan] for plan in query_plans
    ]
    assert len(fake_runtime.build_calls) == 1
