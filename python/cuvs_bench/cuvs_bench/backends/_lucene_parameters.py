# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure validation for the optional cuVS Lucene build and search controls."""

from __future__ import annotations

from typing import Any, Mapping

_JAVA_INT_MAX = 2_147_483_647


# Mirrors GPUKnnFloatVectorQuery.MAX_SEARCH_WIDTH, its result-buffer indexing limit.
_MAX_CAGRA_SEARCH_WIDTH = 4_194_303
_CUSTOM_BUILD_KEYS = frozenset(
    (
        "graph_degree",
        "intermediate_graph_degree",
        "graph_build_algo",
        "nn_descent_n_iters",
    )
)
CUVS_BUILD_KEYS = frozenset(
    (
        *_CUSTOM_BUILD_KEYS,
        "build_strategy",
        "hnsw_layers",
        "writer_threads",
        "build_quality",
    )
)
CAGRA_SEARCH_KEYS = frozenset(
    ("itopk", "search_width", "max_iterations", "algo", "thread_block_size")
)


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be an integer in [{minimum}, {maximum}], got {value!r}"
        )
    return value


def _choice(value: Any, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


def normalize_cuvs_build_controls(
    parameters: Mapping[str, Any], *, hnsw: bool
) -> dict[str, Any]:
    """Normalize new controls, retaining absence of legacy default settings."""
    controls = {
        key: value
        for key, value in parameters.items()
        if key in CUVS_BUILD_KEYS
    }
    if not controls:
        return {}
    unsupported = (
        {"build_quality"} if hnsw else {"hnsw_layers"}
    ) & controls.keys()
    if unsupported:
        raise ValueError(
            "Unsupported Lucene build parameters: "
            + ", ".join(sorted(unsupported))
        )
    custom = _CUSTOM_BUILD_KEYS & controls.keys()
    strategy = _choice(
        controls.get("build_strategy", "custom" if custom else "heuristic"),
        "build_strategy",
        ("heuristic", "custom"),
    )
    if custom or "build_strategy" in controls:
        controls["build_strategy"] = strategy
    # The dataset heuristic consumes graph_degree directly, unlike the HNSW
    # heuristic. Every other custom-only setting would otherwise be ignored.
    ignored_custom = custom - ({"graph_degree"} if not hnsw else set())
    if strategy == "heuristic" and ignored_custom:
        raise ValueError(
            "build_strategy=heuristic conflicts with custom build parameters: "
            + ", ".join(sorted(ignored_custom))
        )
    ignored_heuristic = {"build_quality"} & controls.keys()
    if strategy == "custom" and ignored_heuristic:
        raise ValueError(
            "build_strategy=custom conflicts with heuristic build parameters: "
            + ", ".join(sorted(ignored_heuristic))
        )
    ranges = {
        "graph_degree": (1, 512),
        "intermediate_graph_degree": (2, 512),
        "nn_descent_n_iters": (1, 100),
        "hnsw_layers": (1, 99),
        "writer_threads": (1, 512),
        "build_quality": (0, _JAVA_INT_MAX),
    }
    for key, (minimum, maximum) in ranges.items():
        if key in controls:
            controls[key] = _integer(controls[key], key, minimum, maximum)
    choices = {
        "graph_build_algo": ("nn_descent",),
    }
    for key, allowed in choices.items():
        if key in controls:
            controls[key] = _choice(controls[key], key, allowed)
    if (
        "nn_descent_n_iters" in controls
        and controls.get("graph_build_algo", "nn_descent") != "nn_descent"
    ):
        raise ValueError(
            "nn_descent_n_iters requires graph_build_algo=nn_descent"
        )
    if strategy == "custom":
        max_connections = parameters.get("m", 32)
        graph_degree = controls.get(
            "graph_degree", min(64, 2 * max_connections) if hnsw else 64
        )
        intermediate_degree = controls.get(
            "intermediate_graph_degree", max(128, graph_degree)
        )
        if intermediate_degree < graph_degree:
            raise ValueError(
                "intermediate_graph_degree must be >= graph_degree"
            )
        if hnsw and graph_degree > 2 * max_connections:
            raise ValueError(
                "graph_degree must be <= 2 * m for accelerated HNSW"
            )
    return controls


def normalize_cagra_search_controls(
    parameters: Mapping[str, Any], k: int
) -> dict[str, Any]:
    """Keep supplied CAGRA controls distinct from the result count."""
    _integer(k, "k", 1, _JAVA_INT_MAX)
    unsupported = set(parameters) - CAGRA_SEARCH_KEYS
    if unsupported:
        raise ValueError(
            "Unsupported Lucene search parameters: "
            + ", ".join(sorted(str(name) for name in unsupported))
        )
    controls = dict(parameters)
    for key, minimum, maximum in (
        ("itopk", k, _JAVA_INT_MAX),
        ("search_width", 1, _MAX_CAGRA_SEARCH_WIDTH),
        ("max_iterations", 0, _JAVA_INT_MAX),
    ):
        if key in controls:
            controls[key] = _integer(controls[key], key, minimum, maximum)
    if "algo" in controls:
        controls["algo"] = _choice(
            controls["algo"], "algo", ("auto", "single_cta", "multi_cta")
        )
    if "thread_block_size" in controls:
        block = _integer(
            controls["thread_block_size"], "thread_block_size", 0, 1024
        )
        if block not in (0, 64, 128, 256, 512, 1024):
            raise ValueError(
                "thread_block_size must be 0, 64, 128, 256, 512, or 1024"
            )
    if controls.get("algo") == "single_cta" and controls.get("itopk", k) > 512:
        raise ValueError("single_cta requires effective itopk <= 512")
    return controls
