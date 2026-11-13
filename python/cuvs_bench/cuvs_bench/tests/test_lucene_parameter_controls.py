# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and persistence contracts; doubles do not exercise a GPU."""

import pytest

from cuvs_bench.backends.lucene import (
    ACCELERATED_HNSW_ALGORITHM,
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    _search_parameters,
)


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
