# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Build the CPU-only client bridges against the installed PyLucene classes."""

import hashlib
import os
import subprocess
import tempfile
import warnings
import zipfile
from pathlib import Path

from .._lucene_compat import LUCENE_COMPATIBILITY


def standalone_search_jar(classpath: str) -> Path | None:
    """Cache the same three Java helpers used by the accelerated client."""
    try:
        sources = Path(__file__).parent
        version = LUCENE_COMPATIBILITY.pylucene_version
        digest = hashlib.sha256(
            f"pylucene-{version}-javac-release-22".encode()
        )
        java_files = [
            sources / name
            for name in (
                "FloatQueryVector.java",
                "IndexSearcherTimingBridge.java",
                "RankedSearchHits.java",
            )
        ]
        for source in java_files:
            digest.update(source.read_bytes())
        cache_home = os.environ.get("XDG_CACHE_HOME")
        cache = Path(cache_home or "")
        if not cache.is_absolute():
            cache = Path.home() / ".cache"
        cache = cache / "cuvs-bench" / "pylucene"
        jar = cache / f"search-helpers-{digest.hexdigest()}.jar"
        java_home = os.environ.get("JAVA_HOME")
        javac = (
            str(Path(java_home) / "bin" / "javac") if java_home else "javac"
        )
        if jar.is_file():
            return jar
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            work = Path(temp)
            subprocess.run(
                [
                    javac,
                    "--release",
                    "22",
                    "-proc:none",
                    "-classpath",
                    classpath,
                    "-d",
                    str(work),
                    *map(str, java_files),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with zipfile.ZipFile(work / "helpers.jar", "w") as archive:
                for compiled in sorted(work.rglob("*.class")):
                    archive.write(compiled, compiled.relative_to(work))
            # Readers only see a complete JAR, including concurrent first runs.
            os.replace(work / "helpers.jar", jar)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        warnings.warn(
            f"PyLucene CPU search helpers unavailable ({detail}). "
            "Using per-hit transfers and Python dispatch timing.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return jar
