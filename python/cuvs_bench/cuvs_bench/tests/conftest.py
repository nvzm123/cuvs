#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Pytest configuration for cuvs_bench tests."""

import pytest


def pytest_addoption(parser):
    """Add the explicit opt-in for live Lucene integration tests."""
    parser.addoption(
        "--run-lucene-large-segment-e2e",
        action="store_true",
        default=False,
        help="run the separate 3 GiB Lucene capacity suite in a fresh serial process",
    )
    parser.addoption(
        "--run-lucene-e2e",
        action="store_true",
        default=False,
        help=(
            "run live Lucene tests; all cases require PyLucene/Java, and "
            "GPU-intended cases additionally require cuVS/CUDA/GPU"
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Skip live Lucene cases unless the suite was selected explicitly."""
    for item in items:
        if "lucene_large_segment_e2e" in item.keywords:
            if not config.getoption("--run-lucene-large-segment-e2e"):
                item.add_marker(
                    pytest.mark.skip(
                        reason="requires --run-lucene-large-segment-e2e in a fresh process"
                    )
                )
        elif "lucene_e2e" in item.keywords and not config.getoption(
            "--run-lucene-e2e"
        ):
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
