# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure validation for the optional cuVS Lucene build and search controls."""

from __future__ import annotations

from typing import Any, Mapping

_JAVA_INT_MAX = 2_147_483_647


# Mirrors GPUKnnFloatVectorQuery.MAX_SEARCH_WIDTH, its result-buffer indexing limit.
_MAX_CAGRA_SEARCH_WIDTH = 4_194_303
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
