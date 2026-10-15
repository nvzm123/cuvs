#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Pytest configuration for cuvs_bench tests."""

import pytest


def pytest_addoption(parser):
    """Add explicit opt-ins for ordinary and large Lucene integration tests."""
    parser.addoption(
        "--run-lucene-e2e",
        action="store_true",
        default=False,
        help=(
            "run live Lucene tests; all cases require PyLucene/Java, and "
            "GPU-intended cases additionally require cuVS/CUDA/GPU"
        ),
    )
    parser.addoption(
        "--run-lucene-large-segment-e2e",
        action="store_true",
        default=False,
        help=(
            "run only the resource-intensive Lucene cases that build a "
            "single segment from more than 2 GiB of vector data"
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Select ordinary and large Lucene cases through independent opt-ins."""
    run_ordinary = config.getoption("--run-lucene-e2e")
    run_large = config.getoption("--run-lucene-large-segment-e2e")
    for item in items:
        if "lucene_large_segment_e2e" in item.keywords:
            if not run_large:
                item.add_marker(
                    pytest.mark.skip(
                        reason="requires --run-lucene-large-segment-e2e"
                    )
                )
        elif "lucene_e2e" in item.keywords and not run_ordinary:
            item.add_marker(
                pytest.mark.skip(reason="requires --run-lucene-e2e")
            )


def pytest_configure(config):
    """Register elastic plugin when elasticsearch is available.

    Ensures elastic tests run when elasticsearch is installed, even if
    cuvs-bench-elastic was not installed via pip (e.g. using PYTHONPATH).
    """
    try:
        import elasticsearch  # noqa: F401
    except ImportError:
        return

    try:
        from cuvs_bench_elastic import register
    except ImportError:
        from cuvs_bench.backends.elasticsearch import register

    register()
