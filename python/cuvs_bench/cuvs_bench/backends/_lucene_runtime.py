#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""PyLucene lifecycle, index I/O, and fail-closed CAGRA verification."""

from __future__ import annotations

import hashlib
import importlib
import os
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._lucene_runtime_config import maven_artifact_version

CPU_HNSW_CODEC = "Lucene101"
ACCELERATED_HNSW_CODEC = "Lucene101AcceleratedHNSWCodec"
CAGRA_CODEC = "CuVS2510GPUSearchCodec"
CONFIGURED_ACCELERATED_HNSW_CODEC = (
    "com.nvidia.cuvs.lucene.Lucene101ConfiguredHNSWCodec"
)
RAM_PER_THREAD_HARD_LIMIT_BRIDGE = (
    "com.nvidia.cuvs.lucene.IndexWriterConfigPerThreadHardLimitBridge"
)
FBIN_INDEXING_BRIDGE = "com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge"
_WRITER_CONFIG_REQUEST_KEY = "config"
_PER_THREAD_HARD_LIMIT_REQUEST_KEY = "per_thread_hard_limit_mb"
HNSW_MAX_CONN_PROPERTY = "com.nvidia.cuvs.lucene.hnsw.maxConn"
HNSW_BEAM_WIDTH_PROPERTY = "com.nvidia.cuvs.lucene.hnsw.beamWidth"
MAX_CAGRA_TOP_K = 1024
REQUIRED_PYLUCENE_VERSION = "10.2.0"
_PYLUCENE_SETUP_GUIDANCE = (
    "Setup documentation: fern/pages/cuvs_bench/lucene_backend.md "
    "(published at https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide/"
    "cu-vs-bench-tool/lucene-backend). Source-checkout helper: "
    "python/cuvs_bench/tools/pylucene/build_pylucene_10_2.sh "
    "(run it with --build-root <stable-absolute-path>, then source "
    "<stable-absolute-path>/activate.sh)."
)

_ID_FIELD = "id"
_VECTOR_FIELD = "vector"
_MAX_DIMENSIONS = 4096
_CAGRA_META_EXTENSION = ".vemc"
_CAGRA_META_CODEC_NAME = "Lucene102CuVSVectorsFormatMeta"
_CAGRA_DATA_EXTENSION = ".vcag"
_CAGRA_DATA_CODEC_NAME = "Lucene102CuVSVectorsFormatIndex"
_CAGRA_FORMAT_VERSION = 0
_FLOAT32_ENCODING_ORDINAL = 1
_EUCLIDEAN_SIMILARITY_ORDINAL = 0
_INDEX_SEARCHER_TIMING_BRIDGE = (
    "com.nvidia.cuvs.lucene.IndexSearcherTimingBridge"
)
_SEARCHER_REQUEST_KEY = "searcher"
_QUERY_REQUEST_KEY = "query"
_TOP_K_REQUEST_KEY = "top_k"
_TOP_DOCS_RESPONSE_KEY = "top_docs"
_ELAPSED_NANOS_RESPONSE_KEY = "elapsed_nanos"
DIRECT_PYLUCENE_DISPATCH = "direct_pylucene"
TIMED_BRIDGE_PYLUCENE_DISPATCH = "thin_jar_timing_bridge"

_JVM_LOCK = threading.Lock()
_CONFIGURED_CODEC_LOCK = threading.Lock()
_INITIALIZED_CLASSPATH: str | None = None
_INITIALIZED_VMARGS: tuple[str, ...] | None = None
_INITIALIZED_ARTIFACT_PROVENANCE: dict[str, str] | None = None
_INITIALIZED_ARTIFACT_TOKENS: dict[str, tuple[int, ...]] | None = None


class _CleanupStack:
    """Close resources in reverse order without hiding an earlier failure."""

    def __init__(self) -> None:
        self._cleanups: list[tuple[str, Callable[[], None]]] = []

    def __enter__(self) -> "_CleanupStack":
        return self

    def add(self, description: str, cleanup: Callable[[], None]) -> None:
        self._cleanups.append((description, cleanup))

    def __exit__(
        self, _kind: Any, error: BaseException | None, _tb: Any
    ) -> bool:
        pending = error
        for description, cleanup in reversed(self._cleanups):
            try:
                cleanup()
            except BaseException as cleanup_error:
                if pending is None:
                    pending = cleanup_error
                elif isinstance(
                    cleanup_error, (KeyboardInterrupt, SystemExit)
                ) and isinstance(pending, Exception):
                    cleanup_error.add_note(
                        "Resource handling first failed: "
                        f"{type(pending).__name__}: {pending}"
                    )
                    pending = cleanup_error
                elif isinstance(pending, Exception):
                    pending.add_note(
                        f"Failed to {description}: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
        if pending is not None and pending is not error:
            raise pending
        return False


def _rollback_writer(writer: Any, error: BaseException) -> None:
    """Roll back a failed writer without swallowing process-control errors."""
    try:
        writer.rollback()
    except BaseException as rollback_error:
        if isinstance(
            rollback_error, (KeyboardInterrupt, SystemExit)
        ) and isinstance(error, Exception):
            rollback_error.add_note(
                f"Lucene writer first failed: {type(error).__name__}: {error}"
            )
            raise
        error.add_note(
            "Failed to roll back Lucene writer: "
            f"{type(rollback_error).__name__}: {rollback_error}"
        )


def _read_jar(path: Path, label: str) -> tuple[set[str], dict[str, bytes]]:
    inspected = {
        "META-INF/MANIFEST.MF",
        "META-INF/maven/com.nvidia.cuvs/cuvs-java/pom.properties",
        "META-INF/maven/com.nvidia.cuvs.lucene/cuvs-lucene/pom.properties",
        "META-INF/services/org.apache.lucene.codecs.Codec",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            entries = set(archive.namelist())
            contents = {
                name: archive.read(name)
                for name in inspected
                if name in entries
            }
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"{label} is not a readable JAR: {path}") from error
    return entries, contents


def _maven_coordinates(
    contents: Mapping[str, bytes], descriptor: str
) -> tuple[str, str, str]:
    try:
        text = contents[descriptor].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"Java artifact is missing valid Maven coordinates at {descriptor}"
        ) from error
    properties = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and not key.lstrip().startswith(("#", "!")):
            properties[key.strip()] = value.strip()
    try:
        return (
            properties["groupId"],
            properties["artifactId"],
            properties["version"],
        )
    except KeyError as error:
        raise RuntimeError(
            f"Incomplete Maven coordinates at {descriptor}"
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_stat_token(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _verify_artifact_tokens(
    tokens: Mapping[str, tuple[int, ...]],
    provenance: Mapping[str, str],
) -> None:
    expected_hashes = {
        provenance.get("cuvs_java_jar_path"): provenance.get(
            "cuvs_java_jar_sha256"
        ),
        provenance.get("cuvs_lucene_jar_path"): provenance.get(
            "cuvs_lucene_jar_sha256"
        ),
    }
    for raw_path, expected in tokens.items():
        path = Path(raw_path)
        try:
            before = _artifact_stat_token(path)
        except OSError as error:
            raise RuntimeError(
                f"Initialized Java artifact is no longer available: {path}"
            ) from error
        expected_hash = expected_hashes.get(raw_path)
        if (
            before != expected
            or not isinstance(expected_hash, str)
            or _sha256(path) != expected_hash
            or _artifact_stat_token(path) != expected
        ):
            raise RuntimeError(
                "A Java artifact changed after the process-wide JVM was "
                f"initialized: {path}. Start a new process."
            )


def _verify_artifact_stat_tokens(
    tokens: Mapping[str, tuple[int, ...]],
) -> None:
    """Catch ordinary artifact replacement between operation boundaries."""
    for raw_path, expected in tokens.items():
        path = Path(raw_path)
        try:
            actual = _artifact_stat_token(path)
        except OSError as error:
            raise RuntimeError(
                f"Initialized Java artifact is no longer available: {path}"
            ) from error
        if actual != expected:
            raise RuntimeError(
                "A Java artifact changed after the process-wide JVM was "
                f"initialized: {path}. Start a new process."
            )


def _validate_artifacts(
    java_jar: Path, lucene_jar: Path
) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
    before = {
        str(java_jar): _artifact_stat_token(java_jar),
        str(lucene_jar): _artifact_stat_token(lucene_jar),
    }
    java_entries, java_contents = _read_jar(java_jar, "cuvs_java_jar")
    java_required = {
        "com/nvidia/cuvs/CagraIndex.class",
        "com/nvidia/cuvs/CuVSResources.class",
        "META-INF/versions/22/com/nvidia/cuvs/spi/JDKProvider.class",
    }
    missing = sorted(java_required - java_entries)
    if missing:
        raise RuntimeError(
            "cuvs_java_jar is not the base cuvs-java artifact; missing: "
            + ", ".join(missing)
        )
    manifest = java_contents.get("META-INF/MANIFEST.MF", b"").decode(
        "utf-8", errors="replace"
    )
    if "multi-release: true" not in manifest.casefold():
        raise RuntimeError("cuvs_java_jar must declare Multi-Release: true")
    if any(
        entry.rpartition("/")[2] in {"libcuvs.so", "libcuvs_c.so"}
        for entry in java_entries
    ):
        raise RuntimeError(
            "cuvs_java_jar embeds native libraries; use the base JAR, not a "
            "native-classifier artifact"
        )

    lucene_entries, lucene_contents = _read_jar(lucene_jar, "cuvs_lucene_jar")
    lucene_required = {
        "com/nvidia/cuvs/lucene/CuVS2510GPUVectorsFormat.class",
        "com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.class",
        "com/nvidia/cuvs/lucene/CuvsBenchFbinIndexingBridge.class",
        "com/nvidia/cuvs/lucene/IndexSearcherTimingBridge.class",
        "com/nvidia/cuvs/lucene/IndexWriterConfigPerThreadHardLimitBridge.class",
        "com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.class",
        "com/nvidia/cuvs/lucene/Lucene101ConfiguredHNSWCodec.class",
        "META-INF/services/org.apache.lucene.codecs.Codec",
    }
    missing = sorted(lucene_required - lucene_entries)
    if missing:
        raise RuntimeError(
            "cuvs_lucene_jar is not the standard cuvs-lucene artifact; missing: "
            + ", ".join(missing)
        )
    bundled_lucene = next(
        (
            entry
            for entry in lucene_entries
            if entry.endswith(".class")
            and (
                entry.startswith("org/apache/lucene/")
                or "/org/apache/lucene/" in entry
            )
        ),
        None,
    )
    if bundled_lucene:
        raise RuntimeError(
            "cuvs_lucene_jar bundles Lucene classes; use the dependency-thin "
            f"artifact. First bundled class: {bundled_lucene}"
        )
    providers = {
        line.partition("#")[0].strip()
        for line in lucene_contents[
            "META-INF/services/org.apache.lucene.codecs.Codec"
        ]
        .decode("utf-8")
        .splitlines()
        if line.partition("#")[0].strip()
    }
    if "com.nvidia.cuvs.lucene.CuVS2510GPUSearchCodec" not in providers:
        raise RuntimeError(
            "cuvs_lucene_jar does not advertise CuVS2510GPUSearchCodec"
        )
    if "com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec" not in providers:
        raise RuntimeError(
            "cuvs_lucene_jar does not advertise Lucene101AcceleratedHNSWCodec"
        )

    java_coordinates = _maven_coordinates(
        java_contents,
        "META-INF/maven/com.nvidia.cuvs/cuvs-java/pom.properties",
    )
    lucene_coordinates = _maven_coordinates(
        lucene_contents,
        "META-INF/maven/com.nvidia.cuvs.lucene/cuvs-lucene/pom.properties",
    )
    if java_coordinates[:2] != ("com.nvidia.cuvs", "cuvs-java"):
        raise RuntimeError(
            f"Unexpected cuvs-java coordinates: {java_coordinates[:2]}"
        )
    if lucene_coordinates[:2] != (
        "com.nvidia.cuvs.lucene",
        "cuvs-lucene",
    ):
        raise RuntimeError(
            f"Unexpected cuvs-lucene coordinates: {lucene_coordinates[:2]}"
        )
    expected_version = maven_artifact_version()
    if java_coordinates[2] != lucene_coordinates[2]:
        raise RuntimeError(
            "cuvs-java and cuvs-lucene JAR versions differ: "
            f"{java_coordinates[2]} != {lucene_coordinates[2]}"
        )
    if java_coordinates[2] != expected_version:
        raise RuntimeError(
            "Java artifacts do not match this cuVS Bench release: expected "
            f"{expected_version}, found {java_coordinates[2]}"
        )
    provenance = {
        "cuvs_java_coordinates": ":".join(java_coordinates),
        "cuvs_java_jar_path": str(java_jar),
        "cuvs_java_jar_sha256": _sha256(java_jar),
        "cuvs_lucene_coordinates": ":".join(lucene_coordinates),
        "cuvs_lucene_jar_path": str(lucene_jar),
        "cuvs_lucene_jar_sha256": _sha256(lucene_jar),
    }
    after = {
        str(java_jar): _artifact_stat_token(java_jar),
        str(lucene_jar): _artifact_stat_token(lucene_jar),
    }
    if before != after:
        raise RuntimeError(
            "Java artifacts changed while their identities were being validated"
        )
    return provenance, after


def _load_pylucene() -> Any:
    try:
        return importlib.import_module("lucene")
    except ImportError as error:
        raise ImportError(
            "The Lucene backend requires the custom PyLucene 10.2.0 runtime, "
            "which is not included in cuVS Bench packages. "
            f"{_PYLUCENE_SETUP_GUIDANCE} PyLucene import failed: {error}"
        ) from error


def _classpath(
    config: Mapping[str, Any], lucene: Any, *, validate_artifacts: bool
) -> tuple[str, dict[str, str], dict[str, tuple[int, ...]]]:
    entries = []
    provenance = {}
    artifact_tokens = {}
    java_value = config.get("cuvs_java_jar")
    lucene_value = config.get("cuvs_lucene_jar")
    if bool(java_value) != bool(lucene_value):
        raise RuntimeError(
            "Both cuvs_java_jar and cuvs_lucene_jar are required together"
        )
    if java_value:
        java_jar = Path(os.fspath(java_value)).resolve()
        lucene_jar = Path(os.fspath(lucene_value)).resolve()
        if validate_artifacts:
            provenance, artifact_tokens = _validate_artifacts(
                java_jar, lucene_jar
            )
        entries.extend((str(java_jar), str(lucene_jar)))
    entries.append(str(lucene.CLASSPATH))
    return os.pathsep.join(entries), provenance, artifact_tokens


def _vmargs(config: Mapping[str, Any]) -> list[str]:
    arguments = [
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules=jdk.incubator.vector",
    ]
    if library_path := config.get("java_library_path"):
        arguments.append(f"-Djava.library.path={os.fspath(library_path)}")
    extra = config.get("jvm_args", ())
    if isinstance(extra, (str, bytes)) or not isinstance(extra, (list, tuple)):
        raise TypeError("jvm_args must be a list or tuple of strings")
    if not all(isinstance(argument, str) for argument in extra):
        raise TypeError("Every jvm_args entry must be a string")
    arguments.extend(extra)
    return arguments


def initialize_pylucene(
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, str], dict[str, tuple[int, ...]]]:
    """Start PyLucene once with an immutable, validated JVM configuration."""
    global _INITIALIZED_ARTIFACT_PROVENANCE
    global _INITIALIZED_ARTIFACT_TOKENS
    global _INITIALIZED_CLASSPATH, _INITIALIZED_VMARGS
    lucene = _load_pylucene()
    actual_version = str(getattr(lucene, "VERSION", "<missing>"))
    if actual_version != REQUIRED_PYLUCENE_VERSION:
        raise RuntimeError(
            "PyLucene must match cuvs-lucene's Lucene version: expected "
            f"{REQUIRED_PYLUCENE_VERSION}, found {actual_version}. "
            f"{_PYLUCENE_SETUP_GUIDANCE}"
        )
    vmargs = _vmargs(config)
    with _JVM_LOCK:
        environment = lucene.getVMEnv()
        if environment is None:
            classpath, artifact_provenance, artifact_tokens = _classpath(
                config, lucene, validate_artifacts=True
            )
            environment = lucene.initVM(classpath=classpath, vmargs=vmargs)
            environment = environment or lucene.getVMEnv()
            if environment is None:
                raise RuntimeError("PyLucene did not return a JVM environment")
            _INITIALIZED_CLASSPATH = classpath
            _INITIALIZED_VMARGS = tuple(vmargs)
            _INITIALIZED_ARTIFACT_PROVENANCE = dict(artifact_provenance)
            _INITIALIZED_ARTIFACT_TOKENS = dict(artifact_tokens)
        else:
            classpath, _unused_provenance, _unused_tokens = _classpath(
                config, lucene, validate_artifacts=False
            )
            if (
                _INITIALIZED_CLASSPATH != classpath
                or _INITIALIZED_VMARGS != tuple(vmargs)
            ):
                raise RuntimeError(
                    "PyLucene's process-wide JVM is already initialized with a "
                    "different classpath or JVM arguments. Start a new process and "
                    "let cuVS Bench initialize PyLucene."
                )
            artifact_provenance = dict(_INITIALIZED_ARTIFACT_PROVENANCE or {})
            artifact_tokens = dict(_INITIALIZED_ARTIFACT_TOKENS or {})
            _verify_artifact_tokens(artifact_tokens, artifact_provenance)
        environment.attachCurrentThread()
    return lucene, artifact_provenance, artifact_tokens


@dataclass(frozen=True)
class CagraVerification:
    segment_count: int
    field_count: int
    vector_count: int
    dimensions: int

    def metadata(self) -> dict[str, int | str]:
        return {
            "persisted_index_kind": "gpu_cagra_only",
            "segment_count": self.segment_count,
            "field_count": self.field_count,
            "vector_count": self.vector_count,
            "dimensions": self.dimensions,
        }


@dataclass(frozen=True)
class _RawCagraField:
    number: int
    encoding: int
    similarity: int
    dimensions: int
    vector_count: int
    cagra_offset: int
    cagra_length: int
    brute_force_offset: int
    brute_force_length: int


class CagraVerificationError(RuntimeError):
    """Raised when persisted files do not contain an all-CAGRA index."""


@dataclass(frozen=True)
class LuceneIndexVerification:
    """Observed structure of one committed Lucene vector index."""

    codec: str
    segment_count: int
    field_count: int
    vector_count: int
    dimensions: int

    def metadata(self) -> dict[str, int | str]:
        persisted_index_kind = {
            CPU_HNSW_CODEC: "cpu_hnsw",
            ACCELERATED_HNSW_CODEC: "hnsw",
            CAGRA_CODEC: "gpu_cagra",
        }[self.codec]
        return {
            "persisted_index_kind": persisted_index_kind,
            "segment_count": self.segment_count,
            "field_count": self.field_count,
            "vector_count": self.vector_count,
            "dimensions": self.dimensions,
        }


class LuceneIndexVerifier:
    """Validate the physical codec, vector schema, counts, and live-doc state."""

    def __init__(self, runtime: "LuceneRuntime") -> None:
        self.runtime = runtime

    def verify(
        self,
        index_path: Path,
        *,
        expected_codec: str,
        expected_vector_count: int,
        expected_dimensions: int,
    ) -> LuceneIndexVerification:
        runtime = self.runtime
        runtime.attach_current_thread()
        directory = runtime.FSDirectory.open(
            runtime.Paths.get(str(index_path))
        )
        with _CleanupStack() as cleanups:
            cleanups.add("close Lucene directory", directory.close)
            segment_infos = runtime.SegmentInfos.readLatestCommit(directory)
            segments = [
                runtime.SegmentCommitInfo.cast_(raw) for raw in segment_infos
            ]
            if not segments:
                raise RuntimeError("Lucene index has no committed segments")
            for segment in segments:
                segment_name = str(segment.info.name)
                if (
                    segment.hasDeletions()
                    or int(segment.getDelCount())
                    or int(segment.getSoftDelCount())
                ):
                    raise RuntimeError(
                        "Lucene benchmark index has committed deletions in "
                        f"{segment_name!r}"
                    )
                actual_codec = str(segment.info.getCodec().getName())
                if actual_codec != expected_codec:
                    raise RuntimeError(
                        f"Segment {segment_name!r} uses {actual_codec}, not "
                        f"{expected_codec}"
                    )

            reader = runtime.DirectoryReader.open(directory)
            cleanups.add("close Lucene reader", reader.close)
            if int(reader.numDocs()) != int(reader.maxDoc()):
                raise RuntimeError("Lucene benchmark index contains deletions")
            vector_count = 0
            field_count = 0
            dimensions = set()
            for leaf in reader.leaves():
                leaf_reader = leaf.reader()
                vector_fields = []
                for raw_info in leaf_reader.getFieldInfos():
                    info = runtime.FieldInfo.cast_(raw_info)
                    if int(info.getVectorDimension()) > 0:
                        vector_fields.append(info)
                if len(vector_fields) != 1:
                    raise RuntimeError(
                        "Each Lucene segment must contain exactly one vector "
                        f"field; found {len(vector_fields)}"
                    )
                info = vector_fields[0]
                if str(info.getName()) != _VECTOR_FIELD:
                    raise RuntimeError(
                        f"Unexpected Lucene vector field {info.getName()!r}"
                    )
                if info.getVectorEncoding() != runtime.VectorEncoding.FLOAT32:
                    raise RuntimeError("Lucene vector field is not FLOAT32")
                if (
                    info.getVectorSimilarityFunction()
                    != runtime.VectorSimilarityFunction.EUCLIDEAN
                ):
                    raise RuntimeError("Lucene vector field is not Euclidean")
                values = leaf_reader.getFloatVectorValues(_VECTOR_FIELD)
                if values is None:
                    raise RuntimeError(
                        "Lucene segment is missing vector values"
                    )
                segment_vectors = int(values.size())
                if segment_vectors != int(leaf_reader.numDocs()):
                    raise RuntimeError(
                        "Lucene segment vector and document counts differ: "
                        f"{segment_vectors} != {leaf_reader.numDocs()}"
                    )
                vector_count += segment_vectors
                field_count += 1
                dimensions.add(int(info.getVectorDimension()))

        if vector_count != expected_vector_count:
            raise RuntimeError(
                f"Lucene index has {vector_count} vectors; expected "
                f"{expected_vector_count}"
            )
        if dimensions != {expected_dimensions}:
            raise RuntimeError(
                f"Lucene index dimensions are {sorted(dimensions)}; expected "
                f"{expected_dimensions}"
            )
        return LuceneIndexVerification(
            codec=expected_codec,
            segment_count=len(segments),
            field_count=field_count,
            vector_count=vector_count,
            dimensions=expected_dimensions,
        )


class CagraIndexVerifier:
    """Verify that every committed vector field contains CAGRA and no BFI."""

    def __init__(self, runtime: "LuceneRuntime") -> None:
        self.runtime = runtime

    @staticmethod
    def _suffix(segment_name: str, metadata_file: str) -> str:
        stem = metadata_file[: -len(_CAGRA_META_EXTENSION)]
        if stem == segment_name:
            return ""
        prefix = f"{segment_name}_"
        if not stem.startswith(prefix) or stem == prefix:
            raise CagraVerificationError(
                f"Unexpected CAGRA metadata filename {metadata_file!r} "
                f"for segment {segment_name!r}"
            )
        return stem[len(prefix) :]

    @staticmethod
    def _decode_fields(
        metadata_input: Any, metadata_file: str
    ) -> list[_RawCagraField]:
        fields = []
        numbers = set()
        while True:
            number = int(metadata_input.readInt())
            if number == -1:
                return fields
            if number < 0 or number in numbers:
                raise CagraVerificationError(
                    f"Invalid field number {number} in {metadata_file!r}"
                )
            numbers.add(number)
            fields.append(
                _RawCagraField(
                    number=number,
                    encoding=int(metadata_input.readInt()),
                    similarity=int(metadata_input.readInt()),
                    dimensions=int(metadata_input.readInt()),
                    vector_count=int(metadata_input.readInt()),
                    cagra_offset=int(metadata_input.readVLong()),
                    cagra_length=int(metadata_input.readVLong()),
                    brute_force_offset=int(metadata_input.readVLong()),
                    brute_force_length=int(metadata_input.readVLong()),
                )
            )

    @staticmethod
    def _validate_field(field: _RawCagraField, metadata_file: str) -> None:
        if field.encoding != _FLOAT32_ENCODING_ORDINAL:
            raise CagraVerificationError(
                f"Non-FLOAT32 vector encoding in {metadata_file!r}"
            )
        if field.similarity != _EUCLIDEAN_SIMILARITY_ORDINAL:
            raise CagraVerificationError(
                f"Non-Euclidean vector similarity in {metadata_file!r}"
            )
        if (
            not 1 <= field.dimensions <= _MAX_DIMENSIONS
            or field.vector_count < 1
        ):
            raise CagraVerificationError(
                f"Invalid vector shape in {metadata_file!r}"
            )
        if field.brute_force_length != 0:
            raise CagraVerificationError(
                f"Persisted brute-force fallback for field {field.number} in "
                f"{metadata_file!r}"
            )
        if field.cagra_length <= 0:
            raise CagraVerificationError(
                f"No persisted CAGRA index for field {field.number} in "
                f"{metadata_file!r}; cuVS may have fallen back"
            )

    def _read_fields(
        self, directory: Any, segment: Any, metadata_file: str
    ) -> list[_RawCagraField]:
        runtime = self.runtime
        suffix = self._suffix(str(segment.info.name), metadata_file)
        metadata_input = directory.openChecksumInput(metadata_file)
        with _CleanupStack() as cleanups:
            cleanups.add("close CAGRA metadata", metadata_input.close)
            runtime.CodecUtil.checkIndexHeader(
                metadata_input,
                _CAGRA_META_CODEC_NAME,
                _CAGRA_FORMAT_VERSION,
                _CAGRA_FORMAT_VERSION,
                segment.info.getId(),
                suffix,
            )
            fields = self._decode_fields(metadata_input, metadata_file)
            runtime.CodecUtil.checkFooter(metadata_input)
        for field in fields:
            self._validate_field(field, metadata_file)
        return fields

    def _verify_data(
        self,
        directory: Any,
        segment: Any,
        metadata_file: str,
        fields: Sequence[_RawCagraField],
    ) -> None:
        runtime = self.runtime
        data_file = (
            metadata_file[: -len(_CAGRA_META_EXTENSION)]
            + _CAGRA_DATA_EXTENSION
        )
        suffix = self._suffix(str(segment.info.name), metadata_file)
        data_input = directory.openInput(data_file, runtime.IOContext.READONCE)
        with _CleanupStack() as cleanups:
            cleanups.add("close CAGRA data", data_input.close)
            runtime.CodecUtil.checkIndexHeader(
                data_input,
                _CAGRA_DATA_CODEC_NAME,
                _CAGRA_FORMAT_VERSION,
                _CAGRA_FORMAT_VERSION,
                segment.info.getId(),
                suffix,
            )
            payload_start = int(data_input.getFilePointer())
            payload_end = int(data_input.length()) - int(
                runtime.CodecUtil.footerLength()
            )
            expected = payload_start
            for field in sorted(fields, key=lambda item: item.cagra_offset):
                if field.cagra_offset != expected:
                    raise CagraVerificationError(
                        f"CAGRA metadata does not exactly cover {data_file!r}"
                    )
                expected = field.cagra_offset + field.cagra_length
            if expected != payload_end:
                raise CagraVerificationError(
                    f"CAGRA metadata does not exactly cover {data_file!r}"
                )
            runtime.CodecUtil.checksumEntireFile(data_input)

    def _segment_directory(self, root: Any, segment: Any) -> tuple[Any, bool]:
        if not segment.info.getUseCompoundFile():
            return root, False
        compound_format = segment.info.getCodec().compoundFormat()
        return compound_format.getCompoundReader(root, segment.info), True

    @staticmethod
    def _metadata_files(
        directory: Any, segment: Any, *, compound: bool
    ) -> list[str]:
        # A compound reader exposes only one segment's embedded files. The root
        # directory does not, so ask SegmentInfo for that segment's files.
        names = directory.listAll() if compound else segment.info.files()
        return sorted(
            str(name)
            for name in names
            if str(name).endswith(_CAGRA_META_EXTENSION)
        )

    def _verify_segment(
        self, root: Any, segment: Any
    ) -> tuple[int, int, set[int]]:
        runtime = self.runtime
        segment_name = str(segment.info.name)
        if (
            segment.hasDeletions()
            or int(segment.getDelCount())
            or int(segment.getSoftDelCount())
        ):
            raise CagraVerificationError(
                f"CAGRA benchmark index has committed deletions in {segment_name!r}"
            )
        if str(segment.info.getCodec().getName()) != CAGRA_CODEC:
            raise CagraVerificationError(
                f"Segment {segment_name!r} uses {segment.info.getCodec().getName()}, "
                f"not {CAGRA_CODEC}"
            )
        directory, compound = self._segment_directory(root, segment)
        with _CleanupStack() as cleanups:
            if compound:
                cleanups.add("close compound directory", directory.close)
            metadata_files = self._metadata_files(
                directory, segment, compound=compound
            )
            if not metadata_files:
                raise CagraVerificationError(
                    f"No CAGRA metadata found for segment {segment_name!r}"
                )
            field_infos = (
                segment.info.getCodec()
                .fieldInfosFormat()
                .read(directory, segment.info, "", runtime.IOContext.READONCE)
            )
            verified: dict[int, _RawCagraField] = {}
            for metadata_file in metadata_files:
                fields = self._read_fields(directory, segment, metadata_file)
                self._verify_data(directory, segment, metadata_file, fields)
                for field in fields:
                    if field.number in verified:
                        raise CagraVerificationError(
                            f"Duplicate vector field {field.number} in segment "
                            f"{segment_name!r}"
                        )
                    verified[field.number] = field

            lucene_vector_fields = set()
            for raw_info in field_infos:
                info = runtime.FieldInfo.cast_(raw_info)
                if int(info.getVectorDimension()) <= 0:
                    continue
                lucene_vector_fields.add(int(info.number))
                field = verified.get(int(info.number))
                if field is None:
                    raise CagraVerificationError(
                        f"Vector field {info.getName()!r} has no CAGRA metadata"
                    )
                if str(info.getName()) != _VECTOR_FIELD:
                    raise CagraVerificationError(
                        f"Unexpected vector field {info.getName()!r}"
                    )
                if int(info.getVectorDimension()) != field.dimensions:
                    raise CagraVerificationError(
                        f"Dimension mismatch in segment {segment_name!r}"
                    )
                if info.getVectorEncoding() != runtime.VectorEncoding.FLOAT32:
                    raise CagraVerificationError("Lucene field is not FLOAT32")
                if (
                    info.getVectorSimilarityFunction()
                    != runtime.VectorSimilarityFunction.EUCLIDEAN
                ):
                    raise CagraVerificationError(
                        "Lucene field is not Euclidean"
                    )
            if set(verified) != lucene_vector_fields:
                raise CagraVerificationError(
                    f"CAGRA/Lucene vector-field mismatch in {segment_name!r}"
                )
            vector_count = sum(
                field.vector_count for field in verified.values()
            )
            if vector_count != int(segment.info.maxDoc()):
                raise CagraVerificationError(
                    f"Segment {segment_name!r} has {vector_count} vectors for "
                    f"{segment.info.maxDoc()} documents"
                )
            return (
                len(verified),
                vector_count,
                {field.dimensions for field in verified.values()},
            )

    def verify(
        self,
        index_path: Path,
        *,
        expected_vector_count: int,
        expected_dimensions: int,
    ) -> CagraVerification:
        self.runtime.attach_current_thread()
        root = self.runtime.FSDirectory.open(
            self.runtime.Paths.get(str(index_path))
        )
        segments = []
        with _CleanupStack() as cleanups:
            cleanups.add("close Lucene directory", root.close)
            segment_infos = self.runtime.SegmentInfos.readLatestCommit(root)
            for raw_segment in segment_infos:
                segment = self.runtime.SegmentCommitInfo.cast_(raw_segment)
                segments.append(self._verify_segment(root, segment))
        if not segments:
            raise CagraVerificationError(
                "Lucene index has no committed segments"
            )
        field_count = sum(item[0] for item in segments)
        vector_count = sum(item[1] for item in segments)
        dimensions = set().union(*(item[2] for item in segments))
        if vector_count != expected_vector_count:
            raise CagraVerificationError(
                f"CAGRA index has {vector_count} vectors; expected "
                f"{expected_vector_count}"
            )
        if dimensions != {expected_dimensions}:
            raise CagraVerificationError(
                f"CAGRA index dimensions are {sorted(dimensions)}; expected "
                f"{expected_dimensions}"
            )
        return CagraVerification(
            segment_count=len(segments),
            field_count=field_count,
            vector_count=vector_count,
            dimensions=expected_dimensions,
        )


@dataclass(frozen=True)
class SearchHit:
    document_id: int
    score: float


@dataclass(frozen=True)
class RuntimeBuildTiming:
    """Nanosecond timings for the Lucene build lifecycle."""

    directory_open_ns: int
    writer_setup_ns: int
    document_ingest_ns: int
    writer_commit_close_ns: int
    post_build_reader_ns: int
    directory_close_ns: int
    runtime_build_wall_ns: int
    fbin_read_ns: int | None = None


@dataclass(frozen=True)
class RuntimeBuildResult:
    segment_count: int
    timing: RuntimeBuildTiming
    topology: "RuntimeBuildTopology | None" = None


@dataclass(frozen=True)
class RuntimeFbinBuildResult:
    """Validated facts returned by the Java FBIN indexing bridge."""

    segment_count: int
    timing: RuntimeBuildTiming
    topology: "RuntimeBuildTopology"
    codec_name: str
    source_file_size: int
    source_file_vector_count: int
    dimensions: int
    header_bytes: int
    vector_count: int
    indexed_payload_bytes: int
    vector_payload_sha256: str


@dataclass(frozen=True)
class RuntimeBuildTopology:
    """Requested and observed topology for a controlled Lucene build."""

    requested_premerge_segment_count: int | None
    requested_num_indexing_threads: int | None
    actual_indexing_thread_count: int
    max_concurrent_indexing_threads: int
    indexing_execution_mode: str
    indexing_worker_document_counts: tuple[int, ...]
    observed_premerge_segment_count: int
    requested_force_merge_segment_count: int
    premerge_segment_vector_counts: tuple[int, ...]
    max_buffered_docs: int
    applied_ram_per_thread_hard_limit_mb: int
    ingest_merge_policy: str
    final_merge_policy: str | None

    def manifest(self) -> dict[str, Any]:
        """Return structured, JSON-safe evidence for index reuse."""
        return {
            "requested_premerge_segment_count": (
                self.requested_premerge_segment_count
            ),
            "requested_num_indexing_threads": (
                self.requested_num_indexing_threads
            ),
            "actual_indexing_thread_count": self.actual_indexing_thread_count,
            "max_concurrent_indexing_threads": (
                self.max_concurrent_indexing_threads
            ),
            "indexing_execution_mode": self.indexing_execution_mode,
            "indexing_worker_document_counts": list(
                self.indexing_worker_document_counts
            ),
            "observed_premerge_segment_count": (
                self.observed_premerge_segment_count
            ),
            "requested_force_merge_segment_count": (
                self.requested_force_merge_segment_count
            ),
            "premerge_segment_vector_counts": list(
                self.premerge_segment_vector_counts
            ),
            "max_buffered_docs": self.max_buffered_docs,
            "applied_ram_per_thread_hard_limit_mb": (
                self.applied_ram_per_thread_hard_limit_mb
            ),
            "ingest_merge_policy": self.ingest_merge_policy,
            "final_merge_policy": self.final_merge_policy,
        }

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "requested_num_indexing_threads": (
                self.requested_num_indexing_threads
            ),
            "actual_indexing_thread_count": (
                self.actual_indexing_thread_count
            ),
            "max_concurrent_indexing_threads": (
                self.max_concurrent_indexing_threads
            ),
            "indexing_execution_mode": self.indexing_execution_mode,
            "indexing_worker_document_counts": (
                "["
                + ",".join(
                    str(value)
                    for value in self.indexing_worker_document_counts
                )
                + "]"
            ),
            "observed_premerge_segment_count": (
                self.observed_premerge_segment_count
            ),
            "requested_force_merge_segment_count": (
                self.requested_force_merge_segment_count
            ),
            "premerge_segment_vector_counts": (
                "["
                + ",".join(
                    str(value) for value in self.premerge_segment_vector_counts
                )
                + "]"
            ),
            "max_buffered_docs": self.max_buffered_docs,
            "applied_ram_per_thread_hard_limit_mb": (
                self.applied_ram_per_thread_hard_limit_mb
            ),
            "ingest_merge_policy": self.ingest_merge_policy,
            "final_merge_policy": self.final_merge_policy,
        }
        if self.requested_premerge_segment_count is not None:
            metadata["requested_premerge_segment_count"] = (
                self.requested_premerge_segment_count
            )
        return metadata


@dataclass(frozen=True)
class _ControlledBuildTopology:
    premerge_segment_count: int
    requested_premerge_segment_count: int | None
    num_indexing_threads: int | None
    force_merge_segment_count: int
    ram_per_thread_hard_limit_mb: int
    chunk_size: int
    max_buffered_docs: int


def _controlled_build_topology(
    build_parameters: Mapping[str, Any] | None, vector_count: int
) -> _ControlledBuildTopology | None:
    """Validate the fail-closed topology contract at the JVM boundary."""
    common_keys = {
        "force_merge_segment_count",
        "ram_per_thread_hard_limit_mb",
    }
    legacy_key = "premerge_segment_count"
    threaded_key = "num_indexing_threads"
    topology_keys = common_keys | {legacy_key, threaded_key}
    parameters = build_parameters or {}
    present = topology_keys & set(parameters)
    if not present:
        return None
    if legacy_key in present and threaded_key in present:
        raise RuntimeError(
            "Controlled Lucene builds cannot combine legacy "
            "premerge_segment_count with num_indexing_threads"
        )
    mode_key = threaded_key if threaded_key in present else legacy_key
    required = common_keys | {mode_key}
    if present != required:
        missing = ", ".join(sorted(required - present))
        raise RuntimeError(
            "Controlled Lucene builds require all topology parameters; "
            f"missing: {missing}"
        )
    values = {name: parameters[name] for name in required}
    for name in (mode_key, "ram_per_thread_hard_limit_mb"):
        value = values[name]
        if type(value) is not int or value < 1:
            raise RuntimeError(
                f"Controlled Lucene build parameter {name} must be a "
                f"positive integer, got {value!r}"
            )
    premerge = int(values[mode_key])
    num_indexing_threads = premerge if mode_key == threaded_key else None
    requested_premerge_segment_count = (
        premerge if mode_key == legacy_key else None
    )
    hard_limit = int(values["ram_per_thread_hard_limit_mb"])
    force_merge_value = values["force_merge_segment_count"]
    if type(force_merge_value) is not int or force_merge_value not in (0, 1):
        raise RuntimeError(
            "Controlled Lucene build parameter force_merge_segment_count "
            "must be 0 (disabled) or 1"
        )
    force_merge = force_merge_value
    if premerge > vector_count:
        raise RuntimeError(
            f"{mode_key} cannot exceed the vector count: "
            f"{premerge} > {vector_count}"
        )
    if vector_count % premerge:
        raise RuntimeError(
            "Controlled Lucene builds require equal partitions: vector count "
            f"{vector_count} is not divisible by {mode_key} {premerge}"
        )
    chunk_size = vector_count // premerge
    max_buffered_docs = chunk_size + 1
    if max_buffered_docs > 2_147_483_647:
        raise RuntimeError(
            "Controlled Lucene segment exceeds IndexWriter's integer "
            f"maxBufferedDocs range: {max_buffered_docs}"
        )
    return _ControlledBuildTopology(
        premerge_segment_count=premerge,
        requested_premerge_segment_count=requested_premerge_segment_count,
        num_indexing_threads=num_indexing_threads,
        force_merge_segment_count=force_merge,
        ram_per_thread_hard_limit_mb=hard_limit,
        chunk_size=chunk_size,
        max_buffered_docs=max_buffered_docs,
    )


@dataclass(frozen=True)
class QueryTiming:
    """Measured boundaries for one query in this backend's serial loop."""

    query_prepare_ns: int
    pylucene_search_dispatch_ns: int
    java_index_searcher_search_ns: int | None
    result_materialization_ns: int
    client_query_ns: int


@dataclass(frozen=True)
class RuntimeSearchTiming:
    """Nanosecond timings for one search-parameter plan."""

    directory_open_ns: int
    reader_searcher_setup_ns: int
    query_corpus_wall_ns: int
    reader_close_ns: int
    directory_close_ns: int
    runtime_plan_wall_ns: int
    search_dispatch_kind: str
    queries: tuple[QueryTiming, ...]


@dataclass(frozen=True)
class RuntimeSearchResult:
    hits: list[list[SearchHit]]
    timing: RuntimeSearchTiming
    document_count: int
    dimensions: int


class LuceneRuntime:
    """Own the generated bindings and the narrow Lucene operations Bench uses."""

    def __init__(self, lucene: Any):
        from java.lang import Class, Integer, Long, System
        from java.nio.file import Paths
        from java.util import HashMap, Map
        from java.util.function import Function
        from org.apache.lucene.codecs import Codec, CodecUtil
        from org.apache.lucene.document import (
            Document,
            KnnFloatVectorField,
            NumericDocValuesField,
        )
        from org.apache.lucene.index import (
            DirectoryReader,
            FieldInfo,
            IndexWriter,
            IndexWriterConfig,
            NoMergePolicy,
            SegmentCommitInfo,
            SegmentInfos,
            SerialMergeScheduler,
            TieredMergePolicy,
            VectorEncoding,
            VectorSimilarityFunction,
        )
        from org.apache.lucene.search import (
            IndexSearcher,
            KnnFloatVectorQuery,
            TopDocs,
        )
        from org.apache.lucene.store import FSDirectory, IOContext

        self.lucene = lucene
        self.Class = Class
        self.Integer = Integer
        self.Long = Long
        self.System = System
        self.HashMap = HashMap
        self.Map = Map
        self.Function = Function
        self.Paths = Paths
        self.Codec = Codec
        self.CodecUtil = CodecUtil
        self.Document = Document
        self.KnnFloatVectorField = KnnFloatVectorField
        self.NumericDocValuesField = NumericDocValuesField
        self.DirectoryReader = DirectoryReader
        self.FieldInfo = FieldInfo
        self.IndexWriter = IndexWriter
        self.IndexWriterConfig = IndexWriterConfig
        self.NoMergePolicy = NoMergePolicy
        self.SegmentCommitInfo = SegmentCommitInfo
        self.SegmentInfos = SegmentInfos
        self.SerialMergeScheduler = SerialMergeScheduler
        self.TieredMergePolicy = TieredMergePolicy
        self.VectorEncoding = VectorEncoding
        self.VectorSimilarityFunction = VectorSimilarityFunction
        self.IndexSearcher = IndexSearcher
        self.KnnFloatVectorQuery = KnnFloatVectorQuery
        self.TopDocs = TopDocs
        self.FSDirectory = FSDirectory
        self.IOContext = IOContext
        self.index_verifier = LuceneIndexVerifier(self)
        self.cagra_verifier = CagraIndexVerifier(self)
        self._codecs: dict[str, Any] = {}
        self.artifact_provenance: dict[str, str] = {}
        self._artifact_tokens: dict[str, tuple[int, ...]] = {}
        self._java_search_timer: Any | None = None
        self._java_writer_config_hard_limit: Any | None = None
        self._java_fbin_indexer: Any | None = None

    @classmethod
    def create(cls, config: Mapping[str, Any]) -> "LuceneRuntime":
        lucene, artifact_provenance, artifact_tokens = initialize_pylucene(
            config
        )
        runtime = cls(lucene)
        runtime.artifact_provenance = artifact_provenance
        runtime._artifact_tokens = artifact_tokens
        if artifact_provenance:
            runtime._java_search_timer = runtime._load_java_search_timer()
        return runtime

    def _load_java_search_timer(self) -> Any:
        """Load the thin-JAR timer through JCC's wrapped Function interface."""
        try:
            instance = self.Class.forName(
                _INDEX_SEARCHER_TIMING_BRIDGE
            ).newInstance()
            return self.Function.cast_(instance)
        except Exception as error:
            raise RuntimeError(
                f"Could not load or adapt {_INDEX_SEARCHER_TIMING_BRIDGE} "
                "through PyLucene/JCC: "
                f"{type(error).__name__}: {error}"
            ) from error

    def _load_java_writer_config_hard_limit(self) -> Any:
        """Load the fail-closed IndexWriterConfig hard-limit bridge."""
        try:
            instance = self.Class.forName(
                RAM_PER_THREAD_HARD_LIMIT_BRIDGE
            ).newInstance()
            return self.Function.cast_(instance)
        except Exception as error:
            raise RuntimeError(
                f"Could not load or adapt {RAM_PER_THREAD_HARD_LIMIT_BRIDGE} "
                "through PyLucene/JCC: "
                f"{type(error).__name__}: {error}"
            ) from error

    def _load_java_fbin_indexer(self) -> Any:
        """Load the FBIN indexing bridge through JCC's wrapped Function."""
        try:
            instance = self.Class.forName(FBIN_INDEXING_BRIDGE).newInstance()
            return self.Function.cast_(instance)
        except Exception as error:
            raise RuntimeError(
                f"Could not load or adapt {FBIN_INDEXING_BRIDGE} through "
                "PyLucene/JCC: "
                f"{type(error).__name__}: {error}"
            ) from error

    def _set_ram_per_thread_hard_limit_mb(
        self, config: Any, requested_mb: int
    ) -> None:
        bridge = self._java_writer_config_hard_limit
        if bridge is None:
            bridge = self._load_java_writer_config_hard_limit()
            self._java_writer_config_hard_limit = bridge
        request = self.HashMap()
        request.put(_WRITER_CONFIG_REQUEST_KEY, config)
        request.put(
            _PER_THREAD_HARD_LIMIT_REQUEST_KEY,
            self.Integer.valueOf(requested_mb),
        )
        try:
            raw_response = bridge.apply(request)
            response = self.Map.cast_(raw_response)
            applied = int(
                self.Integer.cast_(
                    response.get(_PER_THREAD_HARD_LIMIT_REQUEST_KEY)
                ).intValue()
            )
        except Exception as error:
            raise RuntimeError(
                "Could not apply Lucene's per-thread RAM hard limit through "
                f"{RAM_PER_THREAD_HARD_LIMIT_BRIDGE}: "
                f"{type(error).__name__}: {error}"
            ) from error
        observed = int(config.getRAMPerThreadHardLimitMB())
        if applied != requested_mb or observed != requested_mb:
            raise RuntimeError(
                "Lucene per-thread RAM hard-limit bridge did not retain the "
                f"requested value: requested {requested_mb}, response "
                f"{applied}, config {observed}"
            )

    @property
    def pylucene_version(self) -> str:
        return str(self.lucene.VERSION)

    def attach_current_thread(self) -> None:
        _verify_artifact_stat_tokens(self._artifact_tokens)
        environment = self.lucene.getVMEnv()
        if environment is None:
            raise RuntimeError("PyLucene JVM is not initialized")
        environment.attachCurrentThread()

    def verify_artifacts(self) -> None:
        """Verify immutable artifact bytes once at an operation boundary."""
        _verify_artifact_tokens(
            self._artifact_tokens, self.artifact_provenance
        )

    def resolve_codec(self, name: str) -> Any:
        self.attach_current_thread()
        if name in self._codecs:
            return self._codecs[name]
        available = self.Codec.availableCodecs()
        if not available.contains(name):
            raise RuntimeError(
                f"Lucene codec {name!r} is unavailable. Available codecs: "
                + ", ".join(sorted(str(value) for value in available))
            )
        codec = self.Codec.forName(name)
        if str(codec.getName()) != name:
            raise RuntimeError(
                f"Requested codec {name}, resolved {codec.getName()}"
            )
        if codec.knnVectorsFormat() is None:
            raise RuntimeError(f"{name} did not initialize a vector format")
        self._codecs[name] = codec
        return codec

    def resolve_configured_hnsw_codec(
        self, max_conn: int, beam_width: int
    ) -> Any:
        """Construct an accelerated-HNSW codec from a property snapshot."""
        self.attach_current_thread()
        configured_values = {
            HNSW_MAX_CONN_PROPERTY: str(max_conn),
            HNSW_BEAM_WIDTH_PROPERTY: str(beam_width),
        }
        with _CONFIGURED_CODEC_LOCK:
            with _CleanupStack() as cleanups:
                for name, value in configured_values.items():
                    previous = self.System.getProperty(name)

                    def restore(
                        name: str = name, previous: Any = previous
                    ) -> None:
                        if previous is None:
                            self.System.clearProperty(name)
                        else:
                            self.System.setProperty(name, str(previous))

                    cleanups.add(
                        f"restore Java system property {name}", restore
                    )
                    self.System.setProperty(name, value)
                reflected = self.Class.forName(
                    CONFIGURED_ACCELERATED_HNSW_CODEC
                ).newInstance()
                codec = self.Codec.cast_(reflected)

        self._validate_codec(codec, ACCELERATED_HNSW_CODEC)
        expected = (
            "Lucene101ConfiguredHNSWCodec["
            f"maxConn={max_conn}, beamWidth={beam_width}]"
        )
        if str(codec) != expected:
            raise RuntimeError(
                "Configured accelerated-HNSW codec did not retain the "
                f"requested parameters: expected {expected!r}, got {str(codec)!r}"
            )
        return codec

    @staticmethod
    def _validate_codec(codec: Any, name: str) -> None:
        if str(codec.getName()) != name:
            raise RuntimeError(
                f"Requested codec {name}, resolved {codec.getName()}"
            )
        if codec.knnVectorsFormat() is None:
            raise RuntimeError(f"{name} did not initialize a vector format")

    def _java_vector(self, vector: np.ndarray) -> Any:
        return self.lucene.JArray("float")(vector.tolist())

    def _document(self, document_id: int, vector: np.ndarray) -> Any:
        document = self.Document()
        document.add(self.NumericDocValuesField(_ID_FIELD, document_id))
        document.add(
            self.KnnFloatVectorField(
                _VECTOR_FIELD,
                self._java_vector(vector),
                self.VectorSimilarityFunction.EUCLIDEAN,
            )
        )
        return document

    def _resolve_build_codec(
        self,
        codec_name: str,
        build_parameters: Mapping[str, Any] | None,
    ) -> Any:
        if codec_name != ACCELERATED_HNSW_CODEC:
            return self.resolve_codec(codec_name)
        if build_parameters is None or not {
            "m",
            "beam_width",
        }.issubset(build_parameters):
            raise RuntimeError(
                "Accelerated-HNSW builds require canonical m and "
                "beam_width parameters"
            )
        return self.resolve_configured_hnsw_codec(
            int(build_parameters["m"]),
            int(build_parameters["beam_width"]),
        )

    def _controlled_ingest_config(
        self,
        codec: Any,
        topology: _ControlledBuildTopology,
        *,
        create: bool,
    ) -> Any:
        config = self.IndexWriterConfig()
        config.setOpenMode(
            self.IndexWriterConfig.OpenMode.CREATE
            if create
            else self.IndexWriterConfig.OpenMode.APPEND
        )
        config.setCodec(codec)
        config.setUseCompoundFile(False)
        config.setCommitOnClose(False)
        config.setMergePolicy(self.NoMergePolicy.INSTANCE)
        config.setMergeScheduler(self.SerialMergeScheduler())
        # Lucene requires one automatic flush trigger to remain enabled. Set
        # maxBufferedDocs first, then disable the RAM trigger deliberately.
        config.setMaxBufferedDocs(topology.max_buffered_docs)
        config.setRAMBufferSizeMB(
            float(self.IndexWriterConfig.DISABLE_AUTO_FLUSH)
        )
        self._set_ram_per_thread_hard_limit_mb(
            config, topology.ram_per_thread_hard_limit_mb
        )
        return config

    def _tiered_merge_policy(self) -> Any:
        merge_policy = self.TieredMergePolicy()
        merge_policy.setNoCFSRatio(0.0)
        merge_policy.setMaxMergedSegmentMB(153600.0)
        merge_policy.setSegmentsPerTier(2.0)
        merge_policy.setMaxMergeAtOnce(500)
        return merge_policy

    def _controlled_merge_config(
        self, codec: Any, topology: _ControlledBuildTopology
    ) -> Any:
        config = self.IndexWriterConfig()
        config.setOpenMode(self.IndexWriterConfig.OpenMode.APPEND)
        config.setCodec(codec)
        config.setUseCompoundFile(False)
        config.setCommitOnClose(False)
        config.setMergeScheduler(self.SerialMergeScheduler())
        config.setMergePolicy(self._tiered_merge_policy())
        self._set_ram_per_thread_hard_limit_mb(
            config, topology.ram_per_thread_hard_limit_mb
        )
        return config

    def _reader_segment_vector_counts(self, reader: Any) -> tuple[int, ...]:
        if int(reader.numDocs()) != int(reader.maxDoc()):
            raise RuntimeError(
                "Controlled Lucene build unexpectedly contains deletions"
            )
        counts = []
        document_count = 0
        for leaf in reader.leaves():
            leaf_reader = leaf.reader()
            leaf_documents = int(leaf_reader.numDocs())
            if leaf_documents != int(leaf_reader.maxDoc()):
                raise RuntimeError(
                    "Controlled Lucene segment unexpectedly contains deletions"
                )
            values = leaf_reader.getFloatVectorValues(_VECTOR_FIELD)
            if values is None:
                raise RuntimeError(
                    "Controlled Lucene segment is missing vector values"
                )
            vector_count = int(values.size())
            if vector_count != leaf_documents:
                raise RuntimeError(
                    "Controlled Lucene segment vector and document counts "
                    f"differ: {vector_count} != {leaf_documents}"
                )
            counts.append(vector_count)
            document_count += leaf_documents
        if document_count != int(reader.numDocs()):
            raise RuntimeError(
                "Controlled Lucene leaf document counts do not match the "
                f"reader: {document_count} != {reader.numDocs()}"
            )
        return tuple(counts)

    def _committed_segment_vector_counts(
        self, directory: Any
    ) -> tuple[int, ...]:
        reader = self.DirectoryReader.open(directory)
        with _CleanupStack() as cleanups:
            cleanups.add("close Lucene topology reader", reader.close)
            return self._reader_segment_vector_counts(reader)

    def _write_controlled_chunk(
        self,
        directory: Any,
        vectors: np.ndarray,
        codec: Any,
        topology: _ControlledBuildTopology,
        *,
        start: int,
        stop: int,
        create: bool,
    ) -> tuple[int, int, int]:
        setup_started = time.perf_counter_ns()
        config = self._controlled_ingest_config(codec, topology, create=create)
        writer = self.IndexWriter(directory, config)
        setup_ns = time.perf_counter_ns() - setup_started
        try:
            ingest_started = time.perf_counter_ns()
            for document_id in range(start, stop):
                writer.addDocument(
                    self._document(document_id, vectors[document_id])
                )
            ingest_ns = time.perf_counter_ns() - ingest_started
            commit_started = time.perf_counter_ns()
            writer.flush()
            writer.commit()
            writer.close()
            commit_close_ns = time.perf_counter_ns() - commit_started
        except BaseException as error:
            _rollback_writer(writer, error)
            raise
        return setup_ns, ingest_ns, commit_close_ns

    def _force_merge_controlled_index(
        self,
        directory: Any,
        codec: Any,
        topology: _ControlledBuildTopology,
    ) -> tuple[int, int]:
        setup_started = time.perf_counter_ns()
        config = self._controlled_merge_config(codec, topology)
        writer = self.IndexWriter(directory, config)
        setup_ns = time.perf_counter_ns() - setup_started
        try:
            merge_started = time.perf_counter_ns()
            writer.forceMerge(topology.force_merge_segment_count, True)
            writer.commit()
            writer.close()
            merge_commit_close_ns = time.perf_counter_ns() - merge_started
        except BaseException as error:
            _rollback_writer(writer, error)
            raise
        return setup_ns, merge_commit_close_ns

    def build_index_from_fbin(
        self,
        index_path: Path,
        source_path: Path,
        *,
        expected_source_size: int,
        expected_file_vector_count: int,
        expected_dimensions: int,
        expected_header_bytes: int,
        vector_count: int,
        codec_name: str,
        build_parameters: Mapping[str, Any],
    ) -> RuntimeFbinBuildResult:
        """Build an eligible controlled index with one Python-to-Java call."""
        self.attach_current_thread()
        topology = _controlled_build_topology(build_parameters, vector_count)
        if (
            codec_name != ACCELERATED_HNSW_CODEC
            or topology is None
            or topology.num_indexing_threads is None
            or topology.force_merge_segment_count != 0
        ):
            raise RuntimeError(
                "The Java FBIN bridge requires accelerated HNSW, "
                "num_indexing_threads, and force_merge_segment_count=0"
            )
        codec = self._resolve_build_codec(codec_name, build_parameters)
        bridge = self._java_fbin_indexer
        if bridge is None:
            bridge = self._load_java_fbin_indexer()
            self._java_fbin_indexer = bridge

        request = self.HashMap()
        request.put("source_path", str(source_path))
        request.put("index_path", str(index_path))
        request.put("codec", codec)
        request.put("expected_codec_name", codec_name)
        request.put(
            "expected_source_size", self.Long.valueOf(expected_source_size)
        )
        request.put(
            "expected_file_vector_count",
            self.Long.valueOf(expected_file_vector_count),
        )
        request.put(
            "expected_dimensions", self.Integer.valueOf(expected_dimensions)
        )
        request.put(
            "expected_header_bytes",
            self.Integer.valueOf(expected_header_bytes),
        )
        request.put("vector_count", self.Long.valueOf(vector_count))
        request.put(
            "num_indexing_threads",
            self.Integer.valueOf(topology.num_indexing_threads),
        )
        request.put(
            "force_merge_segment_count",
            self.Integer.valueOf(topology.force_merge_segment_count),
        )
        request.put(
            "ram_per_thread_hard_limit_mb",
            self.Integer.valueOf(topology.ram_per_thread_hard_limit_mb),
        )
        try:
            response = self.Map.cast_(bridge.apply(request))
        except Exception as error:
            raise RuntimeError(
                "Java FBIN indexing bridge failed: "
                f"{type(error).__name__}: {error}"
            ) from error

        def required(name: str) -> Any:
            if not response.containsKey(name):
                raise RuntimeError(
                    f"Java FBIN indexing bridge omitted response field {name}"
                )
            return response.get(name)

        def integer(name: str) -> int:
            try:
                return int(self.Integer.cast_(required(name)).intValue())
            except RuntimeError:
                raise
            except Exception as error:
                raise RuntimeError(
                    "Java FBIN indexing bridge returned invalid integer "
                    f"field {name}: {type(error).__name__}: {error}"
                ) from error

        def long_integer(name: str) -> int:
            try:
                return int(self.Long.cast_(required(name)).longValue())
            except RuntimeError:
                raise
            except Exception as error:
                raise RuntimeError(
                    "Java FBIN indexing bridge returned invalid long field "
                    f"{name}: {type(error).__name__}: {error}"
                ) from error

        def string(name: str) -> str:
            value = required(name)
            if value is None:
                raise RuntimeError(
                    "Java FBIN indexing bridge returned null string field "
                    f"{name}"
                )
            return str(value)

        returned_codec = string("codec_name")
        source_size = long_integer("source_file_size")
        file_vector_count = long_integer("source_file_vector_count")
        dimensions = integer("dimensions")
        header_bytes = integer("header_bytes")
        returned_vector_count = long_integer("vector_count")
        indexed_payload_bytes = long_integer("indexed_payload_bytes")
        digest = string("vector_payload_sha256")
        returned_partitions = integer("num_indexing_threads")
        returned_force_merge = integer("force_merge_segment_count")
        actual_threads = integer("actual_indexing_thread_count")
        max_concurrent_threads = integer("max_concurrent_indexing_threads")
        execution_mode = string("indexing_execution_mode")
        segment_count = integer("segment_count")
        max_buffered_docs = integer("max_buffered_docs")
        applied_hard_limit = integer("applied_ram_per_thread_hard_limit_mb")
        ingest_merge_policy = string("ingest_merge_policy")

        expected_scalars = {
            "codec_name": (returned_codec, codec_name),
            "source_file_size": (source_size, expected_source_size),
            "source_file_vector_count": (
                file_vector_count,
                expected_file_vector_count,
            ),
            "dimensions": (dimensions, expected_dimensions),
            "header_bytes": (header_bytes, expected_header_bytes),
            "vector_count": (returned_vector_count, vector_count),
            "indexed_payload_bytes": (
                indexed_payload_bytes,
                vector_count
                * expected_dimensions
                * np.dtype(np.float32).itemsize,
            ),
            "num_indexing_threads": (
                returned_partitions,
                topology.num_indexing_threads,
            ),
            "force_merge_segment_count": (returned_force_merge, 0),
            "actual_indexing_thread_count": (actual_threads, 1),
            "max_concurrent_indexing_threads": (max_concurrent_threads, 1),
            "indexing_execution_mode": (
                execution_mode,
                "partitioned_sequential",
            ),
            "segment_count": (segment_count, topology.num_indexing_threads),
            "max_buffered_docs": (
                max_buffered_docs,
                topology.max_buffered_docs,
            ),
            "applied_ram_per_thread_hard_limit_mb": (
                applied_hard_limit,
                topology.ram_per_thread_hard_limit_mb,
            ),
            "ingest_merge_policy": (ingest_merge_policy, "NoMergePolicy"),
        }
        for name, (actual, expected) in expected_scalars.items():
            if actual != expected:
                raise RuntimeError(
                    "Java FBIN indexing bridge response mismatch for "
                    f"{name}: expected {expected!r}, got {actual!r}"
                )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(
                "Java FBIN indexing bridge returned an invalid SHA-256 digest"
            )

        partition_counts = tuple(
            long_integer(f"premerge_segment_vector_count_{partition}")
            for partition in range(topology.num_indexing_threads)
        )
        expected_partition_counts = (topology.chunk_size,) * int(
            topology.num_indexing_threads
        )
        if partition_counts != expected_partition_counts:
            raise RuntimeError(
                "Java FBIN indexing bridge returned unexpected partition "
                f"vector counts: expected {expected_partition_counts}, got "
                f"{partition_counts}"
            )

        timing_names = (
            "directory_open_ns",
            "writer_setup_ns",
            "document_ingest_ns",
            "fbin_read_ns",
            "writer_commit_close_ns",
            "post_build_reader_ns",
            "directory_close_ns",
            "runtime_build_wall_ns",
        )
        timings = {name: long_integer(name) for name in timing_names}
        if any(value < 0 for value in timings.values()):
            raise RuntimeError(
                "Java FBIN indexing bridge returned a negative timing"
            )
        if timings["fbin_read_ns"] > timings["document_ingest_ns"]:
            raise RuntimeError(
                "Java FBIN indexing bridge returned FBIN read time greater "
                "than document ingestion time"
            )

        runtime_timing = RuntimeBuildTiming(
            directory_open_ns=timings["directory_open_ns"],
            writer_setup_ns=timings["writer_setup_ns"],
            document_ingest_ns=timings["document_ingest_ns"],
            writer_commit_close_ns=timings["writer_commit_close_ns"],
            post_build_reader_ns=timings["post_build_reader_ns"],
            directory_close_ns=timings["directory_close_ns"],
            runtime_build_wall_ns=timings["runtime_build_wall_ns"],
            fbin_read_ns=timings["fbin_read_ns"],
        )
        runtime_topology = RuntimeBuildTopology(
            requested_premerge_segment_count=None,
            requested_num_indexing_threads=topology.num_indexing_threads,
            actual_indexing_thread_count=actual_threads,
            max_concurrent_indexing_threads=max_concurrent_threads,
            indexing_execution_mode=execution_mode,
            indexing_worker_document_counts=(vector_count,),
            observed_premerge_segment_count=segment_count,
            requested_force_merge_segment_count=returned_force_merge,
            premerge_segment_vector_counts=partition_counts,
            max_buffered_docs=max_buffered_docs,
            applied_ram_per_thread_hard_limit_mb=applied_hard_limit,
            ingest_merge_policy=ingest_merge_policy,
            final_merge_policy=None,
        )
        return RuntimeFbinBuildResult(
            segment_count=segment_count,
            timing=runtime_timing,
            topology=runtime_topology,
            codec_name=returned_codec,
            source_file_size=source_size,
            source_file_vector_count=file_vector_count,
            dimensions=dimensions,
            header_bytes=header_bytes,
            vector_count=returned_vector_count,
            indexed_payload_bytes=indexed_payload_bytes,
            vector_payload_sha256=digest,
        )

    def build_index(
        self,
        index_path: Path,
        vectors: np.ndarray,
        codec_name: str,
        build_parameters: Mapping[str, Any] | None = None,
    ) -> RuntimeBuildResult:
        self.attach_current_thread()
        runtime_started = time.perf_counter_ns()
        controlled = _controlled_build_topology(
            build_parameters, int(vectors.shape[0])
        )
        if controlled is not None and codec_name != ACCELERATED_HNSW_CODEC:
            raise RuntimeError(
                "Controlled segment topology is only supported for "
                "accelerated-HNSW builds"
            )
        directory_open_started = time.perf_counter_ns()
        directory = self.FSDirectory.open(self.Paths.get(str(index_path)))
        directory_open_ns = time.perf_counter_ns() - directory_open_started
        directory_close_ns = 0
        writer_setup_ns = 0
        document_ingest_ns = 0
        writer_commit_close_ns = 0
        post_build_reader_ns = 0
        segment_count = 0
        runtime_topology = None

        def close_directory() -> None:
            nonlocal directory_close_ns
            started = time.perf_counter_ns()
            try:
                directory.close()
            finally:
                directory_close_ns += time.perf_counter_ns() - started

        with _CleanupStack() as cleanups:
            cleanups.add("close Lucene directory", close_directory)
            writer_setup_started = time.perf_counter_ns()
            codec = self._resolve_build_codec(codec_name, build_parameters)
            writer_setup_ns += time.perf_counter_ns() - writer_setup_started
            if controlled is None:
                setup_started = time.perf_counter_ns()
                config = self.IndexWriterConfig()
                config.setOpenMode(self.IndexWriterConfig.OpenMode.CREATE)
                config.setCodec(codec)
                writer = self.IndexWriter(directory, config)
                writer_setup_ns += time.perf_counter_ns() - setup_started
                try:
                    ingest_started = time.perf_counter_ns()
                    for document_id, vector in enumerate(vectors):
                        writer.addDocument(self._document(document_id, vector))
                    document_ingest_ns = (
                        time.perf_counter_ns() - ingest_started
                    )
                    commit_started = time.perf_counter_ns()
                    writer.commit()
                    writer.close()
                    writer_commit_close_ns = (
                        time.perf_counter_ns() - commit_started
                    )
                except BaseException as error:
                    _rollback_writer(writer, error)
                    raise
                post_build_started = time.perf_counter_ns()
                reader = self.DirectoryReader.open(directory)
                with _CleanupStack() as reader_cleanups:
                    reader_cleanups.add("close Lucene reader", reader.close)
                    segment_count = int(reader.leaves().size())
                post_build_reader_ns = (
                    time.perf_counter_ns() - post_build_started
                )
            else:
                # The historical vectorsearch-benchmarks CAGRA route names
                # this control numIndexThreads, but builds K contiguous
                # partitions in sequential passes.  Preserve that behavior:
                # the removed Python shared-writer route was substantially
                # slower in validation, did not guarantee equal DWPT segments,
                # and was not the PR-2476 path this backend should reproduce.
                observed_premerge_counts: tuple[int, ...] = ()
                for chunk_number in range(controlled.premerge_segment_count):
                    start = chunk_number * controlled.chunk_size
                    stop = start + controlled.chunk_size
                    setup_ns, ingest_ns, commit_close_ns = (
                        self._write_controlled_chunk(
                            directory,
                            vectors,
                            codec,
                            controlled,
                            start=start,
                            stop=stop,
                            create=chunk_number == 0,
                        )
                    )
                    writer_setup_ns += setup_ns
                    document_ingest_ns += ingest_ns
                    writer_commit_close_ns += commit_close_ns
                    post_build_started = time.perf_counter_ns()
                    observed_counts = self._committed_segment_vector_counts(
                        directory
                    )
                    post_build_reader_ns += (
                        time.perf_counter_ns() - post_build_started
                    )
                    expected_count = chunk_number + 1
                    expected_vectors = (
                        controlled.chunk_size,
                    ) * expected_count
                    if observed_counts != expected_vectors:
                        raise RuntimeError(
                            "Controlled Lucene ingest topology mismatch after "
                            f"partition {expected_count}: expected segment "
                            f"vector counts {expected_vectors}, observed "
                            f"{observed_counts}"
                        )
                    observed_premerge_counts = observed_counts

                final_merge_policy = None
                final_counts = observed_premerge_counts
                if (
                    controlled.force_merge_segment_count == 1
                    and controlled.premerge_segment_count > 1
                ):
                    setup_ns, commit_close_ns = (
                        self._force_merge_controlled_index(
                            directory, codec, controlled
                        )
                    )
                    writer_setup_ns += setup_ns
                    writer_commit_close_ns += commit_close_ns
                    post_build_started = time.perf_counter_ns()
                    final_counts = self._committed_segment_vector_counts(
                        directory
                    )
                    post_build_reader_ns += (
                        time.perf_counter_ns() - post_build_started
                    )
                    final_merge_policy = "TieredMergePolicy"
                expected_final_counts = (
                    (int(vectors.shape[0]),)
                    if controlled.force_merge_segment_count == 1
                    else observed_premerge_counts
                )
                if final_counts != expected_final_counts:
                    raise RuntimeError(
                        "Controlled Lucene final topology mismatch: expected "
                        f"segment vector counts {expected_final_counts}, "
                        f"observed {final_counts}"
                    )
                segment_count = len(final_counts)
                requested_num_indexing_threads = (
                    controlled.num_indexing_threads
                )
                runtime_topology = RuntimeBuildTopology(
                    requested_premerge_segment_count=(
                        controlled.requested_premerge_segment_count
                    ),
                    requested_num_indexing_threads=(
                        requested_num_indexing_threads
                    ),
                    actual_indexing_thread_count=1,
                    max_concurrent_indexing_threads=1,
                    indexing_execution_mode=(
                        "partitioned_sequential"
                        if requested_num_indexing_threads is not None
                        else "legacy_premerge_sequential"
                    ),
                    indexing_worker_document_counts=((int(vectors.shape[0]),)),
                    observed_premerge_segment_count=len(
                        observed_premerge_counts
                    ),
                    requested_force_merge_segment_count=(
                        controlled.force_merge_segment_count
                    ),
                    premerge_segment_vector_counts=observed_premerge_counts,
                    max_buffered_docs=controlled.max_buffered_docs,
                    applied_ram_per_thread_hard_limit_mb=(
                        controlled.ram_per_thread_hard_limit_mb
                    ),
                    ingest_merge_policy="NoMergePolicy",
                    final_merge_policy=final_merge_policy,
                )
        return RuntimeBuildResult(
            segment_count=segment_count,
            timing=RuntimeBuildTiming(
                directory_open_ns=directory_open_ns,
                writer_setup_ns=writer_setup_ns,
                document_ingest_ns=document_ingest_ns,
                writer_commit_close_ns=writer_commit_close_ns,
                post_build_reader_ns=post_build_reader_ns,
                directory_close_ns=directory_close_ns,
                runtime_build_wall_ns=time.perf_counter_ns() - runtime_started,
            ),
            topology=runtime_topology,
        )

    @staticmethod
    def _index_dimensions(reader: Any) -> int:
        dimensions = {
            int(values.dimension())
            for leaf in reader.leaves()
            if (values := leaf.reader().getFloatVectorValues(_VECTOR_FIELD))
            is not None
            and values.size() > 0
        }
        if len(dimensions) != 1:
            raise RuntimeError(
                f"Lucene index has invalid vector dimensions: {sorted(dimensions)}"
            )
        return dimensions.pop()

    def _search(
        self, searcher: Any, query: Any, k: int
    ) -> tuple[Any, int, int | None]:
        """Search once and return optional in-JVM timing evidence."""
        dispatch_started = time.perf_counter_ns()
        if self._java_search_timer is None:
            top_docs = searcher.search(query, k)
            return (
                top_docs,
                time.perf_counter_ns() - dispatch_started,
                None,
            )
        request = self.HashMap()
        request.put(_SEARCHER_REQUEST_KEY, searcher)
        request.put(_QUERY_REQUEST_KEY, query)
        request.put(_TOP_K_REQUEST_KEY, self.Integer.valueOf(k))
        raw_response = self._java_search_timer.apply(request)
        response = self.Map.cast_(raw_response)
        top_docs = self.TopDocs.cast_(response.get(_TOP_DOCS_RESPONSE_KEY))
        elapsed = self.Long.cast_(
            response.get(_ELAPSED_NANOS_RESPONSE_KEY)
        ).longValue()
        elapsed_ns = int(elapsed)
        if elapsed_ns < 0:
            raise RuntimeError(
                "The Java IndexSearcher timing bridge returned a negative duration"
            )
        return (
            top_docs,
            time.perf_counter_ns() - dispatch_started,
            elapsed_ns,
        )

    @staticmethod
    def _materialize_hits(
        reader: Any, score_docs: Sequence[Any]
    ) -> list[SearchHit]:
        """Read each leaf's forward-only IDs, preserving score rank."""
        ranked_score_docs = sorted(
            enumerate(score_docs), key=lambda item: int(item[1].doc)
        )
        hits_by_rank: dict[int, SearchHit] = {}
        next_hit = 0
        for leaf in reader.leaves():
            leaf_reader = leaf.reader()
            doc_base = int(leaf.docBase)
            doc_limit = doc_base + int(leaf_reader.maxDoc())
            if (
                next_hit == len(ranked_score_docs)
                or int(ranked_score_docs[next_hit][1].doc) >= doc_limit
            ):
                continue
            document_ids = leaf_reader.getNumericDocValues(_ID_FIELD)
            if document_ids is None:
                raise RuntimeError(
                    "Lucene index has no numeric dataset IDs; rebuild the "
                    "index with --force"
                )
            while next_hit < len(ranked_score_docs):
                rank, score_doc = ranked_score_docs[next_hit]
                lucene_doc_id = int(score_doc.doc)
                if lucene_doc_id >= doc_limit:
                    break
                if lucene_doc_id < doc_base:
                    raise RuntimeError(
                        f"Lucene document {lucene_doc_id} is outside its leaf"
                    )
                if not document_ids.advanceExact(lucene_doc_id - doc_base):
                    raise RuntimeError(
                        f"Lucene document {lucene_doc_id} has no numeric "
                        "dataset ID; rebuild the index with --force"
                    )
                hits_by_rank[rank] = SearchHit(
                    int(document_ids.longValue()), float(score_doc.score)
                )
                next_hit += 1
        if next_hit != len(ranked_score_docs):
            lucene_doc_id = int(ranked_score_docs[next_hit][1].doc)
            raise RuntimeError(
                f"Lucene document {lucene_doc_id} is outside the index"
            )
        return [hits_by_rank[rank] for rank in range(len(ranked_score_docs))]

    def _search_one(
        self,
        searcher: Any,
        reader: Any,
        vector: np.ndarray,
        k: int,
        candidates: int,
    ) -> tuple[list[SearchHit], QueryTiming]:
        client_started = time.perf_counter_ns()
        prepare_started = time.perf_counter_ns()
        query = self.KnnFloatVectorQuery(
            _VECTOR_FIELD, self._java_vector(vector), candidates
        )
        query_prepare_ns = time.perf_counter_ns() - prepare_started

        top_docs, pylucene_search_dispatch_ns, java_search_ns = self._search(
            searcher, query, k
        )

        materialization_started = time.perf_counter_ns()
        hits = self._materialize_hits(reader, top_docs.scoreDocs)
        result_materialization_ns = (
            time.perf_counter_ns() - materialization_started
        )
        return hits, QueryTiming(
            query_prepare_ns=query_prepare_ns,
            pylucene_search_dispatch_ns=pylucene_search_dispatch_ns,
            java_index_searcher_search_ns=java_search_ns,
            result_materialization_ns=result_materialization_ns,
            client_query_ns=time.perf_counter_ns() - client_started,
        )

    def search_index(
        self,
        index_path: Path,
        queries: np.ndarray,
        *,
        k: int,
        num_candidates: int,
    ) -> RuntimeSearchResult:
        self.attach_current_thread()
        plan_started = time.perf_counter_ns()
        directory_open_started = time.perf_counter_ns()
        directory = self.FSDirectory.open(self.Paths.get(str(index_path)))
        directory_open_ns = time.perf_counter_ns() - directory_open_started
        reader_close_ns = 0
        directory_close_ns = 0

        def close_reader() -> None:
            nonlocal reader_close_ns
            started = time.perf_counter_ns()
            try:
                reader.close()
            finally:
                reader_close_ns += time.perf_counter_ns() - started

        def close_directory() -> None:
            nonlocal directory_close_ns
            started = time.perf_counter_ns()
            try:
                directory.close()
            finally:
                directory_close_ns += time.perf_counter_ns() - started

        with _CleanupStack() as cleanups:
            cleanups.add("close Lucene directory", close_directory)
            reader_setup_started = time.perf_counter_ns()
            reader = self.DirectoryReader.open(directory)
            cleanups.add("close Lucene reader", close_reader)
            dimensions = self._index_dimensions(reader)
            if queries.shape[1] != dimensions:
                raise ValueError(
                    "Query dimensions do not match the index: "
                    f"{queries.shape[1]} != {dimensions}"
                )
            document_count = int(reader.numDocs())
            if document_count < k:
                raise ValueError(
                    f"Lucene index has {document_count} documents, fewer than k={k}"
                )
            searcher = self.IndexSearcher(reader)
            reader_searcher_setup_ns = (
                time.perf_counter_ns() - reader_setup_started
            )
            all_hits: list[list[SearchHit]] = []
            query_timings: list[QueryTiming] = []
            corpus_started = time.perf_counter_ns()
            for vector in queries:
                hits, query_timing = self._search_one(
                    searcher,
                    reader,
                    vector,
                    k,
                    min(num_candidates, document_count),
                )
                all_hits.append(hits)
                query_timings.append(query_timing)
            query_corpus_wall_ns = time.perf_counter_ns() - corpus_started
        return RuntimeSearchResult(
            hits=all_hits,
            timing=RuntimeSearchTiming(
                directory_open_ns=directory_open_ns,
                reader_searcher_setup_ns=reader_searcher_setup_ns,
                query_corpus_wall_ns=query_corpus_wall_ns,
                reader_close_ns=reader_close_ns,
                directory_close_ns=directory_close_ns,
                runtime_plan_wall_ns=time.perf_counter_ns() - plan_started,
                search_dispatch_kind=(
                    TIMED_BRIDGE_PYLUCENE_DISPATCH
                    if self._java_search_timer is not None
                    else DIRECT_PYLUCENE_DISPATCH
                ),
                queries=tuple(query_timings),
            ),
            document_count=document_count,
            dimensions=dimensions,
        )
