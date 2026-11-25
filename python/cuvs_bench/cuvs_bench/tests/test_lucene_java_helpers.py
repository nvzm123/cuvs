# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for optional standalone CPU search-helper preparation."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from cuvs_bench.backends import _lucene_java


@pytest.mark.parametrize("failure_type", [FileNotFoundError, PermissionError])
def test_source_read_failure_retains_cpu_fallback(monkeypatch, failure_type):
    source = Path(_lucene_java.__file__).parent / "FloatQueryVector.java"
    original_read_bytes = Path.read_bytes

    def read_bytes(path):
        if path == source:
            raise failure_type("helper source unavailable for test")
        return original_read_bytes(path)

    compiler = Mock(side_effect=AssertionError("compiler must not be reached"))
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(_lucene_java.subprocess, "run", compiler)

    with pytest.warns(
        RuntimeWarning,
        match="CPU search helpers unavailable.*helper source unavailable for test",
    ) as warnings:
        assert _lucene_java.standalone_search_jar("unused-classpath") is None

    assert len(warnings) == 1
    compiler.assert_not_called()
