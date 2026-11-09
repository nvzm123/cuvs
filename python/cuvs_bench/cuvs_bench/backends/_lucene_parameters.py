# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure validation for the optional cuVS Lucene build and search controls."""

from __future__ import annotations

import math
import struct
import sys
from typing import Any, Mapping

_JAVA_INT_MAX = 2_147_483_647
# These Java-builder defaults are explicit in saved build identities. Keeping
# them here also permits validation without starting a JVM.
_IVF_DEFAULTS = {
    "n_lists": 1024,
    "n_probes": 20,
    "pq_dim": 0,
    "kmeans_n_iters": 20,
    "kmeans_trainset_fraction": 0.5,
}


# Mirrors GPUKnnFloatVectorQuery.MAX_SEARCH_WIDTH, its result-buffer indexing limit.
_MAX_CAGRA_SEARCH_WIDTH = 4_194_303
IVF_PQ_BUILD_KEYS = frozenset(
    (
        "n_lists",
        "n_probes",
        "kmeans_n_iters",
        "kmeans_trainset_fraction",
        "pq_dim",
        "pq_bits",
        "refinement_rate",
        "max_train_points_per_pq_code",
        "codebook_kind",
        "force_random_rotation",
        "lut_dtype",
        "internal_distance_dtype",
    )
)
_CUSTOM_BUILD_KEYS = frozenset(
    (
        "graph_degree",
        "intermediate_graph_degree",
        "graph_build_algo",
        "nn_descent_n_iters",
        *IVF_PQ_BUILD_KEYS,
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


def _positive_real(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if (
        type(value) not in (int, float)
        or abs(value) > sys.float_info.max
        or not math.isfinite(value)
        or value <= minimum
    ):
        raise ValueError(
            f"{name} must be finite and > {minimum}, got {value!r}"
        )
    return float(value)


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
        "n_lists": (1, _JAVA_INT_MAX),
        "n_probes": (1, _JAVA_INT_MAX),
        "kmeans_n_iters": (1, _JAVA_INT_MAX),
        "pq_dim": (0, _JAVA_INT_MAX),
        "pq_bits": (4, 8),
        "max_train_points_per_pq_code": (1, _JAVA_INT_MAX),
    }
    for key, (minimum, maximum) in ranges.items():
        if key in controls:
            controls[key] = _integer(controls[key], key, minimum, maximum)
    choices = {
        "graph_build_algo": ("nn_descent", "ivf_pq"),
        "codebook_kind": ("per_subspace", "per_cluster"),
        "lut_dtype": ("float32", "float16", "uint8"),
        "internal_distance_dtype": ("float32", "float16"),
    }
    for key, allowed in choices.items():
        if key in controls:
            controls[key] = _choice(controls[key], key, allowed)
    if (
        controls.get("internal_distance_dtype", "float32") == "float16"
        and controls.get("lut_dtype", "float32") == "float32"
    ):
        raise ValueError(
            "internal_distance_dtype=float16 requires lut_dtype=float16 or uint8"
        )
    if (
        "force_random_rotation" in controls
        and type(controls["force_random_rotation"]) is not bool
    ):
        raise ValueError("force_random_rotation must be a boolean")
    if "kmeans_trainset_fraction" in controls:
        fraction = _positive_real(
            controls["kmeans_trainset_fraction"], "kmeans_trainset_fraction"
        )
        if fraction > 1:
            raise ValueError("kmeans_trainset_fraction must be <= 1")
        controls["kmeans_trainset_fraction"] = fraction
    if "refinement_rate" in controls:
        rate = _positive_real(
            controls["refinement_rate"], "refinement_rate", minimum=1.0
        )
        # The Java builder stores a float. Check its actual representation so
        # rounding to 1.0 cannot silently disable graph refinement.
        try:
            native_rate = struct.unpack("f", struct.pack("f", rate))[0]
        except OverflowError as error:
            raise ValueError(
                "refinement_rate must fit a finite float32"
            ) from error
        if not math.isfinite(native_rate) or native_rate <= 1.0:
            raise ValueError(
                "refinement_rate must remain finite and > 1 in float32"
            )
        controls["refinement_rate"] = rate
    if (
        IVF_PQ_BUILD_KEYS & controls.keys()
        and controls.get("graph_build_algo") != "ivf_pq"
    ):
        raise ValueError(
            "IVF-PQ build parameters require graph_build_algo=ivf_pq"
        )
    if (
        "nn_descent_n_iters" in controls
        and controls.get("graph_build_algo", "nn_descent") != "nn_descent"
    ):
        raise ValueError(
            "nn_descent_n_iters requires graph_build_algo=nn_descent"
        )
    if controls.get("graph_build_algo") == "ivf_pq":
        # Record all restored defaults so an old experimental index cannot be
        # reused under a different omitted-parameter policy.
        controls = {**_IVF_DEFAULTS, **controls}
    if (
        "n_probes" in controls
        and "n_lists" in controls
        and controls["n_probes"] > controls["n_lists"]
    ):
        raise ValueError("n_probes must not exceed n_lists")
    if controls.get("pq_dim", 0) * controls.get("pq_bits", 8) % 8:
        raise ValueError("pq_dim * pq_bits must be divisible by 8")
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


def validate_cagra_build_dataset(
    controls: Mapping[str, Any], vector_count: int
) -> None:
    """Reject impossible IVF list counts, not promise automatic segment sizes."""
    if controls.get("graph_build_algo") != "ivf_pq":
        return
    # Explicit topology guarantees equal partition sizes. With automatic
    # segmentation, the corpus size is only an upper bound on segment size;
    # smaller real segments still need the native writer's own checks.
    segment_upper_bound = vector_count // controls.get(
        "premerge_segment_count", 1
    )
    if "n_lists" in controls and controls["n_lists"] > segment_upper_bound:
        raise ValueError(
            "n_lists must not exceed the vector count per requested CAGRA build segment"
        )


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
