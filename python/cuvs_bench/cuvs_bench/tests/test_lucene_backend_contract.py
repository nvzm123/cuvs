#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registration and packaged-configuration contracts for the Lucene backend."""

from importlib import resources

import yaml

from cuvs_bench.backends.base import BuildResult
from cuvs_bench.backends.lucene import (
    CAGRA_ALGORITHM,
    LuceneBackend,
    LuceneConfigLoader,
    register,
)
from cuvs_bench.backends.registry import get_backend_class, get_config_loader


def test_registration_exposes_the_backend_and_config_loader() -> None:
    register()

    backend_class = get_backend_class("lucene")
    assert backend_class is LuceneBackend
    assert backend_class.default_algorithm == CAGRA_ALGORITHM
    assert get_config_loader("lucene") is LuceneConfigLoader


def test_algorithm_configs_define_each_supported_codec() -> None:
    expected_codecs = {
        "lucene_accelerated_hnsw": "Lucene101AcceleratedHNSWCodec",
        "lucene_cpu_hnsw": "Lucene101",
        "lucene_cuvs_cagra": "CuVS2510GPUSearchCodec",
    }
    algorithm_resources = resources.files("cuvs_bench.config.algos")

    for algorithm, codec in expected_codecs.items():
        config = yaml.safe_load(
            algorithm_resources.joinpath(f"{algorithm}.yaml").read_text()
        )
        assert config["name"] == algorithm
        assert set(config["groups"]) == {"base", "test"}
        for group in config["groups"].values():
            assert group == {
                "build": {"codec": [codec]},
                "search": {},
            }


def test_failed_results_are_fatal_for_the_lucene_backend() -> None:
    successful = BuildResult(
        index_path="",
        build_time_seconds=0.0,
        index_size_bytes=0,
        algorithm=CAGRA_ALGORITHM,
        build_params={},
    )
    failed = BuildResult(
        index_path="",
        build_time_seconds=0.0,
        index_size_bytes=0,
        algorithm=CAGRA_ALGORITHM,
        build_params={},
        success=False,
        error_message="GPU path was unavailable",
    )

    assert LuceneBackend.result_failure_message([successful]) is None
    assert LuceneBackend.result_failure_message([successful, failed]) == (
        "GPU path was unavailable"
    )
