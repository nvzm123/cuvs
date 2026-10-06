#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pinned PyLucene source builder."""

import os
import subprocess
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).parents[4]
_RECIPE_DIRECTORY = _REPOSITORY_ROOT / "conda" / "recipes" / "cuvs-bench"
_BUILD_HELPER = _RECIPE_DIRECTORY / "build_pylucene_10_2.sh"
_COMPATIBILITY_PATCH = _RECIPE_DIRECTORY / "pylucene-10.2.0.patch"


def test_source_builder_is_self_describing() -> None:
    completed = subprocess.run(
        ["bash", str(_BUILD_HELPER), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert os.access(_BUILD_HELPER, os.X_OK)
    assert _COMPATIBILITY_PATCH.is_file()
    assert completed.returncode == 0, completed.stderr
    assert "Build an isolated PyLucene 10.2.0 environment" in completed.stdout


def test_source_builder_rejects_unsafe_build_roots(tmp_path: Path) -> None:
    for unsafe_character in (" ", ":", ";", "$", "`", "&", "#", "|"):
        build_root = tmp_path / f"unsafe{unsafe_character}root"
        completed = subprocess.run(
            [
                "bash",
                str(_BUILD_HELPER),
                "--build-root",
                str(build_root),
                "--prepare-only",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode != 0
        assert "--build-root may contain only" in completed.stderr
        assert not build_root.exists()


def test_source_builder_rejects_unsafe_resolved_build_root(
    tmp_path: Path,
) -> None:
    unsafe_target = tmp_path / "unsafe;target"
    unsafe_target.mkdir()
    build_root = tmp_path / "safe-link"
    build_root.symlink_to(unsafe_target, target_is_directory=True)

    completed = subprocess.run(
        [
            "bash",
            str(_BUILD_HELPER),
            "--build-root",
            str(build_root),
            "--prepare-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--build-root may contain only" in completed.stderr
    assert not (unsafe_target / ".build.lock").exists()
