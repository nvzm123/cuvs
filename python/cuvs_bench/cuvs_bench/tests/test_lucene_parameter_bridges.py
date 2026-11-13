#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Python/JCC request wiring; real Java construction is covered separately."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cuvs_bench.backends._lucene_runtime import LuceneRuntime
from _lucene_test_support import RecordingRuntime, _backend_and_index, _dataset
from cuvs_bench.backends.lucene import CAGRA_ALGORITHM


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
            "fake_integer": 2,
            "fake_fraction": 0.25,
            "fake_boolean": True,
            "fake_string": "i-am-fake-data",
        }
    )

    assert request == {
        "fake_integer": ("Integer", 2),
        "fake_fraction": ("Double", 0.25),
        "fake_boolean": ("Boolean", True),
        "fake_string": "i-am-fake-data",
    }


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
    "gpu_controls",
    [
        pytest.param(None, id="stock-hnsw"),
        pytest.param({}, id="gpu-cagra-defaults"),
        pytest.param({"search_width": 32}, id="gpu-cagra"),
    ],
)
def test_query_execution_uses_only_the_requested_search_factory(gpu_controls):
    runtime = object.__new__(LuceneRuntime)
    fake_java_vector = object()
    fake_default_query = object()
    fake_gpu_query = object()
    runtime._java_vector = Mock(return_value=fake_java_vector)
    runtime.KnnFloatVectorQuery = Mock(return_value=fake_default_query)
    runtime._configured_cagra_query = Mock(return_value=fake_gpu_query)
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
    )

    if gpu_controls is None:
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
    runtime._search.assert_called_once_with(fake_searcher, expected_query, 10)


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
