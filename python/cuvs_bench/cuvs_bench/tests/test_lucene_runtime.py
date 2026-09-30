#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for immutable JVM setup and Java artifact provenance."""

import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cuvs_bench.backends import _lucene_runtime
from cuvs_bench.backends._lucene_runtime import (
    ACCELERATED_HNSW_CODEC,
    CONFIGURED_ACCELERATED_HNSW_CODEC_FACTORY,
    INDEX_WRITER_CONFIG_RAM_LIMIT_BRIDGE,
    _CleanupStack,
    _controlled_build_topology,
    _load_pylucene,
    _rollback_writer,
    _validate_artifacts,
    CagraIndexVerifier,
    initialize_pylucene,
    LuceneRuntime,
)
from cuvs_bench.backends._lucene_runtime_config import maven_artifact_version


_ARTIFACT_VERSION = maven_artifact_version()


@pytest.mark.parametrize(
    (
        "vector_count",
        "premerge_segments",
        "force_merge_segments",
        "chunk_size",
    ),
    (
        (10_000_000, 1, 0, 10_000_000),
        (10_000_000, 1, 1, 10_000_000),
        (100_000_000, 4, 0, 25_000_000),
        (100_000_000, 4, 1, 25_000_000),
    ),
)
def test_controlled_topology_derives_equal_partition_plan(
    vector_count: int,
    premerge_segments: int,
    force_merge_segments: int,
    chunk_size: int,
) -> None:
    topology = _controlled_build_topology(
        {
            "premerge_segment_count": premerge_segments,
            "force_merge_segment_count": force_merge_segments,
            "ram_per_thread_hard_limit_mb": 1_945,
        },
        vector_count,
    )

    assert topology is not None
    assert topology.premerge_segment_count == premerge_segments
    assert topology.force_merge_segment_count == force_merge_segments
    assert topology.ram_per_thread_hard_limit_mb == 1_945
    assert topology.allow_unsupported_lucene_ram_limit is False
    assert topology.chunk_size == chunk_size
    assert topology.max_buffered_docs == chunk_size + 1


@pytest.mark.parametrize(
    ("parameters", "vector_count", "message"),
    (
        (
            {"premerge_segment_count": 4},
            100,
            "require all topology parameters",
        ),
        (
            {
                "premerge_segment_count": True,
                "force_merge_segment_count": 1,
                "ram_per_thread_hard_limit_mb": 1_945,
            },
            100,
            "must be a positive integer",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": 2,
                "ram_per_thread_hard_limit_mb": 1_945,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": True,
                "ram_per_thread_hard_limit_mb": 1_945,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": None,
                "ram_per_thread_hard_limit_mb": 1_945,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": "0",
                "ram_per_thread_hard_limit_mb": 1_945,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": 0.0,
                "ram_per_thread_hard_limit_mb": 1_945,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": -1,
                "ram_per_thread_hard_limit_mb": 1_945,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": 1,
                "ram_per_thread_hard_limit_mb": 1_945,
            },
            101,
            "is not divisible",
        ),
    ),
)
def test_controlled_topology_rejects_ambiguous_shapes(
    parameters: dict[str, int], vector_count: int, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _controlled_build_topology(parameters, vector_count)


@pytest.mark.parametrize("hard_limit", (1, 2047))
def test_controlled_topology_accepts_supported_ram_limits(
    hard_limit: int,
) -> None:
    topology = _controlled_build_topology(
        {
            "premerge_segment_count": 1,
            "force_merge_segment_count": 0,
            "ram_per_thread_hard_limit_mb": hard_limit,
        },
        100,
    )

    assert topology is not None
    assert topology.ram_per_thread_hard_limit_mb == hard_limit


@pytest.mark.parametrize("hard_limit", (2048, 6144))
def test_controlled_topology_accepts_explicit_unsupported_ram_limits(
    hard_limit: int,
) -> None:
    topology = _controlled_build_topology(
        {
            "premerge_segment_count": 1,
            "force_merge_segment_count": 0,
            "ram_per_thread_hard_limit_mb": hard_limit,
            "allow_unsupported_lucene_ram_limit": True,
        },
        100,
    )

    assert topology is not None
    assert topology.ram_per_thread_hard_limit_mb == hard_limit
    assert topology.allow_unsupported_lucene_ram_limit is True


@pytest.mark.parametrize("hard_limit", (0, 2_147_483_648, True))
def test_controlled_topology_rejects_out_of_range_ram_limits(
    hard_limit: object,
) -> None:
    with pytest.raises(RuntimeError, match="ram_per_thread_hard_limit_mb"):
        _controlled_build_topology(
            {
                "premerge_segment_count": 1,
                "force_merge_segment_count": 0,
                "ram_per_thread_hard_limit_mb": hard_limit,
            },
            100,
        )


def test_controlled_topology_rejects_unsupported_limit_without_opt_in() -> (
    None
):
    with pytest.raises(RuntimeError, match="require.*allow_unsupported"):
        _controlled_build_topology(
            {
                "premerge_segment_count": 1,
                "force_merge_segment_count": 0,
                "ram_per_thread_hard_limit_mb": 2048,
            },
            100,
        )


def test_controlled_topology_rejects_unused_unsupported_limit_opt_in() -> None:
    with pytest.raises(RuntimeError, match="requires.*>= 2048"):
        _controlled_build_topology(
            {
                "premerge_segment_count": 1,
                "force_merge_segment_count": 0,
                "ram_per_thread_hard_limit_mb": 2047,
                "allow_unsupported_lucene_ram_limit": True,
            },
            100,
        )


def test_controlled_topology_rejects_merge_with_unsupported_limit() -> None:
    with pytest.raises(RuntimeError, match="force_merge_segment_count"):
        _controlled_build_topology(
            {
                "premerge_segment_count": 1,
                "force_merge_segment_count": 1,
                "ram_per_thread_hard_limit_mb": 6144,
                "allow_unsupported_lucene_ram_limit": True,
            },
            100,
        )


@pytest.mark.parametrize("allow_unsupported", (1, "true", None))
def test_controlled_topology_rejects_non_boolean_unsupported_limit_opt_in(
    allow_unsupported: object,
) -> None:
    with pytest.raises(RuntimeError, match="must be a boolean"):
        _controlled_build_topology(
            {
                "premerge_segment_count": 1,
                "force_merge_segment_count": 0,
                "ram_per_thread_hard_limit_mb": 6144,
                "allow_unsupported_lucene_ram_limit": allow_unsupported,
            },
            100,
        )


def _properties(group: str, artifact: str, version: str) -> bytes:
    return (
        f"groupId={group}\nartifactId={artifact}\nversion={version}\n"
    ).encode()


def _write_artifacts(
    directory: Path,
    *,
    java_version: str = _ARTIFACT_VERSION,
    lucene_version: str = _ARTIFACT_VERSION,
) -> tuple[Path, Path]:
    java_jar = directory / "cuvs-java.jar"
    with zipfile.ZipFile(java_jar, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF", "Multi-Release: true\n")
        archive.writestr("com/nvidia/cuvs/CagraIndex.class", b"")
        archive.writestr("com/nvidia/cuvs/CuVSResources.class", b"")
        archive.writestr(
            "META-INF/versions/22/com/nvidia/cuvs/spi/JDKProvider.class", b""
        )
        archive.writestr(
            "META-INF/maven/com.nvidia.cuvs/cuvs-java/pom.properties",
            _properties("com.nvidia.cuvs", "cuvs-java", java_version),
        )

    lucene_jar = directory / "cuvs-lucene.jar"
    with zipfile.ZipFile(lucene_jar, "w") as archive:
        archive.writestr(
            "com/nvidia/cuvs/lucene/CuVS2510GPUVectorsFormat.class", b""
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.class", b""
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/IndexSearcherTimingBridge.class", b""
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/IndexWriterConfigRAMLimitBridge.class",
            b"",
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/"
            "Lucene101AcceleratedHNSWCodecFactory.class",
            b"",
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.class", b""
        )
        archive.writestr(
            "META-INF/services/org.apache.lucene.codecs.Codec",
            "com.nvidia.cuvs.lucene.CuVS2510GPUSearchCodec\n"
            "com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec\n",
        )
        archive.writestr(
            "META-INF/maven/com.nvidia.cuvs.lucene/cuvs-lucene/pom.properties",
            _properties(
                "com.nvidia.cuvs.lucene", "cuvs-lucene", lucene_version
            ),
        )
    return java_jar, lucene_jar


class _FakeEnvironment:
    def attachCurrentThread(self) -> None:
        pass


class _FakeLucene:
    VERSION = "10.2.0"
    CLASSPATH = "/pylucene/lucene-core.jar"

    def __init__(self) -> None:
        self.environment = None
        self.initializations: list[tuple[str, tuple[str, ...]]] = []

    def getVMEnv(self):
        return self.environment

    def initVM(self, *, classpath, vmargs):
        self.initializations.append((classpath, tuple(vmargs)))
        self.environment = _FakeEnvironment()
        return self.environment


def _reset_jvm_state(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "_INITIALIZED_CLASSPATH",
        "_INITIALIZED_VMARGS",
        "_INITIALIZED_ARTIFACT_PROVENANCE",
        "_INITIALIZED_ARTIFACT_TOKENS",
    ):
        monkeypatch.setattr(_lucene_runtime, name, None)


def test_java_search_timer_reports_class_loading_failure() -> None:
    class MissingBridgeClass:
        @staticmethod
        def forName(_name: str):
            raise RuntimeError("unsupported bridge bytecode")

    runtime = object.__new__(LuceneRuntime)
    runtime.Class = MissingBridgeClass

    with pytest.raises(RuntimeError) as failure:
        runtime._load_java_search_timer()

    assert str(failure.value) == (
        "Could not load or adapt "
        "com.nvidia.cuvs.lucene.IndexSearcherTimingBridge through "
        "PyLucene/JCC: RuntimeError: unsupported bridge bytecode"
    )
    assert isinstance(failure.value.__cause__, RuntimeError)


def test_java_search_timer_reports_jcc_adaptation_failure() -> None:
    class LoadedBridgeClass:
        @staticmethod
        def newInstance():
            return object()

    class LoadableClass:
        @staticmethod
        def forName(_name: str):
            return LoadedBridgeClass()

    class UnsupportedFunction:
        @staticmethod
        def cast_(_instance):
            raise TypeError("Function.cast_ rejected bridge")

    runtime = object.__new__(LuceneRuntime)
    runtime.Class = LoadableClass
    runtime.Function = UnsupportedFunction

    with pytest.raises(RuntimeError) as failure:
        runtime._load_java_search_timer()

    assert "TypeError: Function.cast_ rejected bridge" in str(failure.value)
    assert isinstance(failure.value.__cause__, TypeError)


class _JavaInteger:
    def __init__(self, value: int) -> None:
        self.value = value

    @classmethod
    def valueOf(cls, value: int) -> "_JavaInteger":
        return cls(value)

    @classmethod
    def cast_(cls, value: object) -> "_JavaInteger":
        assert isinstance(value, cls)
        return value

    def intValue(self) -> int:
        return self.value


class _JavaBoolean:
    def __init__(self, value: bool) -> None:
        self.value = value

    @classmethod
    def valueOf(cls, value: bool) -> "_JavaBoolean":
        return cls(value)

    def booleanValue(self) -> bool:
        return self.value


class _JavaString:
    def __init__(self, value: str) -> None:
        self.value = value

    @classmethod
    def cast_(cls, value: object) -> "_JavaString":
        assert isinstance(value, cls)
        return value

    def __str__(self) -> str:
        return self.value


class _JavaHashMap(dict):
    def put(self, key: str, value: object) -> object | None:
        previous = self.get(key)
        self[key] = value
        return previous


class _ConfiguredCodec:
    @staticmethod
    def getName() -> str:
        return ACCELERATED_HNSW_CODEC

    @staticmethod
    def knnVectorsFormat() -> object:
        return object()


def _configured_codec_runtime(
    *, factory_error: Exception | None = None
) -> tuple[LuceneRuntime, list[dict[str, int]], list[str]]:
    requests: list[dict[str, int]] = []
    class_names: list[str] = []

    class Factory:
        def apply(self, request: _JavaHashMap) -> _JavaHashMap:
            values = {
                name: _JavaInteger.cast_(value).intValue()
                for name, value in request.items()
            }
            requests.append(values)
            if factory_error is not None:
                raise factory_error
            return _JavaHashMap(
                {
                    "codec": _ConfiguredCodec(),
                    "max_conn": _JavaInteger(values["max_conn"]),
                    "beam_width": _JavaInteger(values["beam_width"]),
                }
            )

    class ReflectedFactory:
        @staticmethod
        def newInstance() -> Factory:
            return Factory()

    class JavaClass:
        @staticmethod
        def forName(name: str):
            class_names.append(name)
            return ReflectedFactory()

    class FunctionBinding:
        @staticmethod
        def cast_(factory: object) -> object:
            return factory

    class MapBinding:
        @staticmethod
        def cast_(response: object) -> object:
            return response

    class CodecBinding:
        @staticmethod
        def cast_(codec: object):
            return codec

        @staticmethod
        def availableCodecs():
            raise AssertionError("configured codec must not use Lucene SPI")

        @staticmethod
        def forName(_name: str):
            raise AssertionError("configured codec must not use Lucene SPI")

    runtime = object.__new__(LuceneRuntime)
    runtime.Class = JavaClass
    runtime.Function = FunctionBinding
    runtime.HashMap = _JavaHashMap
    runtime.Integer = _JavaInteger
    runtime.Map = MapBinding
    runtime.Codec = CodecBinding
    runtime._java_configured_codec_factory = None
    runtime.attach_current_thread = lambda: None
    return runtime, requests, class_names


def test_configured_hnsw_codec_uses_one_atomic_factory_request() -> None:
    runtime, requests, class_names = _configured_codec_runtime()

    codec = runtime.resolve_configured_hnsw_codec(16, 80)

    assert isinstance(codec, _ConfiguredCodec)
    assert requests == [{"max_conn": 16, "beam_width": 80}]
    assert class_names == [CONFIGURED_ACCELERATED_HNSW_CODEC_FACTORY]


def test_configured_hnsw_codec_reports_factory_failure() -> None:
    runtime, requests, _class_names = _configured_codec_runtime(
        factory_error=RuntimeError("constructor failed")
    )

    with pytest.raises(RuntimeError, match="constructor failed"):
        runtime.resolve_configured_hnsw_codec(16, 80)

    assert requests == [{"max_conn": 16, "beam_width": 80}]


class _WriterConfig:
    def __init__(self, *, fail: bool = False, retain: bool = True) -> None:
        self.fail = fail
        self.retain = retain
        self.limit = 1_945

    def setRAMPerThreadHardLimitMB(self, value: int) -> None:
        if self.fail:
            raise ValueError("setter rejected value")
        if self.retain:
            self.limit = value

    def getRAMPerThreadHardLimitMB(self) -> int:
        return self.limit


def _ram_limit_runtime() -> tuple[
    LuceneRuntime, list[dict[str, object]], list[str]
]:
    requests: list[dict[str, object]] = []
    class_names: list[str] = []

    class Bridge:
        def apply(self, request: _JavaHashMap) -> _JavaHashMap:
            config = request["config"]
            hard_limit = _JavaInteger.cast_(
                request["per_thread_hard_limit_mb"]
            ).intValue()
            allow_unsupported = request[
                "allow_unsupported_lucene_ram_limit"
            ].booleanValue()
            requests.append(
                {
                    "config": config,
                    "per_thread_hard_limit_mb": hard_limit,
                    "allow_unsupported_lucene_ram_limit": allow_unsupported,
                }
            )
            config.limit = hard_limit
            return _JavaHashMap(
                {
                    "config": config,
                    "per_thread_hard_limit_mb": _JavaInteger(hard_limit),
                    "application_mode": _JavaString(
                        "unsupported_field_override"
                    ),
                }
            )

    class ReflectedBridge:
        @staticmethod
        def newInstance() -> Bridge:
            return Bridge()

    class JavaClass:
        @staticmethod
        def forName(name: str) -> ReflectedBridge:
            class_names.append(name)
            return ReflectedBridge()

    class FunctionBinding:
        @staticmethod
        def cast_(bridge: object) -> object:
            return bridge

    class MapBinding:
        @staticmethod
        def cast_(response: object) -> object:
            return response

    runtime = object.__new__(LuceneRuntime)
    runtime.Boolean = _JavaBoolean
    runtime.Class = JavaClass
    runtime.Function = FunctionBinding
    runtime.HashMap = _JavaHashMap
    runtime.Integer = _JavaInteger
    runtime.Map = MapBinding
    runtime.String = _JavaString
    runtime._java_ram_limit_bridge = None
    return runtime, requests, class_names


@pytest.mark.parametrize("hard_limit", (1, 2047))
def test_ram_per_thread_hard_limit_uses_supported_public_setter(
    hard_limit: int,
) -> None:
    config = _WriterConfig()

    application_mode = LuceneRuntime._set_ram_per_thread_hard_limit_mb(
        object.__new__(LuceneRuntime),
        config,
        hard_limit,
        allow_unsupported=False,
    )

    assert config.getRAMPerThreadHardLimitMB() == hard_limit
    assert application_mode == "public_setter"


@pytest.mark.parametrize("hard_limit", (2048, 6144))
def test_ram_per_thread_hard_limit_uses_explicit_unsupported_bridge(
    hard_limit: int,
) -> None:
    runtime, requests, class_names = _ram_limit_runtime()
    config = _WriterConfig()

    application_mode = runtime._set_ram_per_thread_hard_limit_mb(
        config, hard_limit, allow_unsupported=True
    )

    assert config.getRAMPerThreadHardLimitMB() == hard_limit
    assert application_mode == "unsupported_field_override"
    assert requests == [
        {
            "config": config,
            "per_thread_hard_limit_mb": hard_limit,
            "allow_unsupported_lucene_ram_limit": True,
        }
    ]
    assert class_names == [INDEX_WRITER_CONFIG_RAM_LIMIT_BRIDGE]


@pytest.mark.parametrize("hard_limit", (0, 2_147_483_648, True))
def test_ram_per_thread_hard_limit_rejects_out_of_range_values(
    hard_limit: object,
) -> None:
    with pytest.raises(RuntimeError, match=r"range \[1, 2147483647\]"):
        LuceneRuntime._set_ram_per_thread_hard_limit_mb(
            object.__new__(LuceneRuntime),
            _WriterConfig(),
            hard_limit,
            allow_unsupported=False,
        )


def test_ram_per_thread_hard_limit_rejects_unsupported_value_without_opt_in() -> (
    None
):
    with pytest.raises(RuntimeError, match="require.*allow_unsupported"):
        LuceneRuntime._set_ram_per_thread_hard_limit_mb(
            object.__new__(LuceneRuntime),
            _WriterConfig(),
            2048,
            allow_unsupported=False,
        )


def test_ram_per_thread_hard_limit_reports_setter_failure() -> None:
    with pytest.raises(RuntimeError, match="setter rejected value"):
        LuceneRuntime._set_ram_per_thread_hard_limit_mb(
            object.__new__(LuceneRuntime),
            _WriterConfig(fail=True),
            1_024,
            allow_unsupported=False,
        )


def test_ram_per_thread_hard_limit_rejects_failed_readback() -> None:
    with pytest.raises(RuntimeError, match="did not retain"):
        LuceneRuntime._set_ram_per_thread_hard_limit_mb(
            object.__new__(LuceneRuntime),
            _WriterConfig(retain=False),
            1_024,
            allow_unsupported=False,
        )


def test_float32_vectors_are_converted_to_a_jcc_compatible_sequence() -> None:
    received = []

    class RecordingLucene:
        @staticmethod
        def JArray(element_type: str):
            assert element_type == "float"

            def record(values):
                received.append(values)
                return values

            return record

    runtime = object.__new__(LuceneRuntime)
    runtime.lucene = RecordingLucene()
    vector = np.asarray([1.25, -2.5], dtype=np.float32)

    converted = runtime._java_vector(vector)

    assert converted == [1.25, -2.5]
    assert received == [[1.25, -2.5]]
    assert all(type(value) is float for value in received[0])


def test_numeric_document_ids_preserve_score_order_across_leaves() -> None:
    class ForwardOnlyDocumentIds:
        def __init__(self, values: dict[int, int]) -> None:
            self.values = values
            self.current = -1
            self.visited = []

        def advanceExact(self, document_id: int) -> bool:
            assert document_id >= self.current
            self.current = document_id
            self.visited.append(document_id)
            return document_id in self.values

        def longValue(self) -> int:
            return self.values[self.current]

    class LeafReader:
        def __init__(
            self, max_doc: int, document_ids: ForwardOnlyDocumentIds
        ) -> None:
            self.max_doc = max_doc
            self.document_ids = document_ids

        def maxDoc(self) -> int:
            return self.max_doc

        def getNumericDocValues(self, field: str):
            assert field == "id"
            return self.document_ids

    class Leaf:
        def __init__(self, doc_base: int, reader: LeafReader) -> None:
            self.docBase = doc_base
            self._reader = reader

        def reader(self) -> LeafReader:
            return self._reader

    class Reader:
        def __init__(self, leaves) -> None:
            self._leaves = leaves

        def leaves(self):
            return self._leaves

    first_ids = ForwardOnlyDocumentIds({2: 102, 5: 105})
    second_ids = ForwardOnlyDocumentIds({2: 108})
    reader = Reader(
        [
            Leaf(0, LeafReader(6, first_ids)),
            Leaf(6, LeafReader(4, second_ids)),
        ]
    )
    score_docs = [
        SimpleNamespace(doc=8, score=0.9),
        SimpleNamespace(doc=2, score=0.8),
        SimpleNamespace(doc=5, score=0.7),
    ]

    hits = LuceneRuntime._materialize_hits(reader, score_docs)

    assert first_ids.visited == [2, 5]
    assert second_ids.visited == [2]
    assert hits == [
        _lucene_runtime.SearchHit(108, 0.9),
        _lucene_runtime.SearchHit(102, 0.8),
        _lucene_runtime.SearchHit(105, 0.7),
    ]


def test_missing_numeric_document_id_requests_an_index_rebuild() -> None:
    class MissingDocumentIds:
        @staticmethod
        def advanceExact(_document_id: int) -> bool:
            return False

    class LeafReader:
        @staticmethod
        def maxDoc() -> int:
            return 4

        @staticmethod
        def getNumericDocValues(_field: str):
            return MissingDocumentIds()

    class Leaf:
        docBase = 0

        @staticmethod
        def reader():
            return LeafReader()

    class Reader:
        @staticmethod
        def leaves():
            return [Leaf()]

    with pytest.raises(RuntimeError, match="rebuild the index with --force"):
        LuceneRuntime._materialize_hits(
            Reader(), [SimpleNamespace(doc=3, score=1.0)]
        )


def test_missing_numeric_document_id_field_requests_an_index_rebuild() -> None:
    class LeafReader:
        @staticmethod
        def maxDoc() -> int:
            return 1

        @staticmethod
        def getNumericDocValues(_field: str):
            return None

    class Leaf:
        docBase = 0

        @staticmethod
        def reader():
            return LeafReader()

    class Reader:
        @staticmethod
        def leaves():
            return [Leaf()]

    with pytest.raises(RuntimeError, match="rebuild the index with --force"):
        LuceneRuntime._materialize_hits(
            Reader(), [SimpleNamespace(doc=0, score=1.0)]
        )


def test_cagra_verifier_selects_only_the_current_noncompound_segment() -> None:
    class RootDirectory:
        @staticmethod
        def listAll():
            return ("_c.vemc", "_n.vemc")

    class SegmentInfo:
        @staticmethod
        def files():
            return ("_c.si", "_c.vcag", "_c.vemc")

    class Segment:
        info = SegmentInfo()

    assert CagraIndexVerifier._metadata_files(
        RootDirectory(), Segment(), compound=False
    ) == ["_c.vemc"]


def test_cagra_verifier_reads_metadata_inside_a_compound_segment() -> None:
    class CompoundDirectory:
        @staticmethod
        def listAll():
            return ("_c.fnm", "_c.vcag", "_c.vemc")

    class SegmentInfo:
        @staticmethod
        def files():
            return ("_c.cfe", "_c.cfs", "_c.si")

    class Segment:
        info = SegmentInfo()

    assert CagraIndexVerifier._metadata_files(
        CompoundDirectory(), Segment(), compound=True
    ) == ["_c.vemc"]


def test_missing_pylucene_reports_the_required_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(_name: str):
        raise ImportError("no lucene module")

    monkeypatch.setattr(
        _lucene_runtime.importlib, "import_module", missing_module
    )

    with pytest.raises(ImportError) as error:
        _load_pylucene()

    message = str(error.value)
    assert "requires the custom PyLucene 10.2.0 runtime" in message
    assert "fern/pages/cuvs_bench/lucene_backend.md" in message
    assert "python/cuvs_bench/tools/pylucene/build_pylucene_10_2.sh" in message
    assert "<stable-absolute-path>/activate.sh" in message
    assert (
        "https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide/"
        "cu-vs-bench-tool/lucene-backend"
    ) in message
    assert "PyLucene import failed: no lucene module" in message


def test_incompatible_pylucene_version_fails_before_vm_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_lucene = _FakeLucene()
    fake_lucene.VERSION = "10.1.0"
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)

    with pytest.raises(RuntimeError) as error:
        initialize_pylucene({})

    message = str(error.value)
    assert "expected 10.2.0, found 10.1.0" in message
    assert "fern/pages/cuvs_bench/lucene_backend.md" in message
    assert "python/cuvs_bench/tools/pylucene/build_pylucene_10_2.sh" in message
    assert "<stable-absolute-path>/activate.sh" in message
    assert (
        "https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide/"
        "cu-vs-bench-tool/lucene-backend"
    ) in message
    assert fake_lucene.initializations == []


def test_artifact_validation_returns_exact_coordinates_paths_and_hashes(
    tmp_path: Path,
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)

    provenance, tokens = _validate_artifacts(java_jar, lucene_jar)

    assert provenance == {
        "cuvs_java_coordinates": (
            f"com.nvidia.cuvs:cuvs-java:{_ARTIFACT_VERSION}"
        ),
        "cuvs_java_jar_path": str(java_jar),
        "cuvs_java_jar_sha256": hashlib.sha256(
            java_jar.read_bytes()
        ).hexdigest(),
        "cuvs_lucene_coordinates": (
            f"com.nvidia.cuvs.lucene:cuvs-lucene:{_ARTIFACT_VERSION}"
        ),
        "cuvs_lucene_jar_path": str(lucene_jar),
        "cuvs_lucene_jar_sha256": hashlib.sha256(
            lucene_jar.read_bytes()
        ).hexdigest(),
    }
    assert set(tokens) == {str(java_jar), str(lucene_jar)}


def test_artifact_validation_rejects_mismatched_versions(
    tmp_path: Path,
) -> None:
    java_jar, lucene_jar = _write_artifacts(
        tmp_path, lucene_version=f"{_ARTIFACT_VERSION}-mismatch"
    )

    with pytest.raises(RuntimeError, match="JAR versions differ"):
        _validate_artifacts(java_jar, lucene_jar)


def test_artifact_validation_rejects_a_native_assembled_jar(
    tmp_path: Path,
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    with zipfile.ZipFile(java_jar, "a") as archive:
        archive.writestr("native/linux-x86_64/libcuvs.so", b"native")

    with pytest.raises(RuntimeError, match="embeds native libraries"):
        _validate_artifacts(java_jar, lucene_jar)


def test_artifact_validation_rejects_an_unreadable_jar(tmp_path: Path) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    java_jar.write_bytes(b"not a zip archive")

    with pytest.raises(RuntimeError, match="not a readable JAR"):
        _validate_artifacts(java_jar, lucene_jar)


def test_artifact_validation_rejects_a_lucene_assembled_jar(
    tmp_path: Path,
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    with zipfile.ZipFile(lucene_jar, "a") as archive:
        archive.writestr("org/apache/lucene/index/IndexReader.class", b"")

    with pytest.raises(RuntimeError, match="bundles Lucene classes"):
        _validate_artifacts(java_jar, lucene_jar)


@pytest.mark.parametrize("control_error", (KeyboardInterrupt, SystemExit))
def test_cleanup_process_control_takes_precedence_over_an_ordinary_failure(
    control_error: type[BaseException],
) -> None:
    def interrupt_cleanup() -> None:
        raise control_error("cleanup interrupted")

    with pytest.raises(control_error) as failure:
        with _CleanupStack() as cleanups:
            cleanups.add("close test resource", interrupt_cleanup)
            raise RuntimeError("operation failed")

    assert failure.value.__notes__ == [
        "Resource handling first failed: RuntimeError: operation failed"
    ]


@pytest.mark.parametrize("control_error", (KeyboardInterrupt, SystemExit))
def test_writer_rollback_process_control_takes_precedence(
    control_error: type[BaseException],
) -> None:
    class InterruptingWriter:
        @staticmethod
        def rollback() -> None:
            raise control_error("rollback interrupted")

    with pytest.raises(control_error) as failure:
        _rollback_writer(InterruptingWriter(), RuntimeError("write failed"))

    assert failure.value.__notes__ == [
        "Lucene writer first failed: RuntimeError: write failed"
    ]


def _controlled_runtime_with_writer(writer: Mock) -> LuceneRuntime:
    """Isolate JVM setup while exercising the real write/merge failure paths."""
    runtime = object.__new__(LuceneRuntime)
    runtime.IndexWriter = lambda _directory, _config: writer
    runtime._controlled_ingest_config = lambda *args, **kwargs: (
        object(),
        "public_setter",
    )
    runtime._controlled_merge_config = lambda *args: (
        object(),
        "public_setter",
    )
    runtime._document = lambda _document_id, _vector: object()
    return runtime


@pytest.mark.parametrize("failed_operation", ("addDocument", "flush"))
def test_controlled_ingestion_rolls_back_without_commit_on_failure(
    failed_operation: str,
) -> None:
    writer = Mock(spec=["addDocument", "flush", "commit", "close", "rollback"])
    original_error = RuntimeError(f"{failed_operation} failed")
    getattr(writer, failed_operation).side_effect = original_error
    runtime = _controlled_runtime_with_writer(writer)

    with pytest.raises(RuntimeError) as failure:
        runtime._write_controlled_chunk(
            object(),
            np.zeros((1, 2), dtype=np.float32),
            object(),
            SimpleNamespace(),
            start=0,
            stop=1,
            create=True,
        )

    assert failure.value is original_error
    writer.rollback.assert_called_once_with()
    writer.commit.assert_not_called()
    writer.close.assert_not_called()


def test_controlled_merge_rolls_back_without_commit_on_failure() -> None:
    writer = Mock(spec=["forceMerge", "commit", "close", "rollback"])
    original_error = RuntimeError("forceMerge failed")
    writer.forceMerge.side_effect = original_error
    runtime = _controlled_runtime_with_writer(writer)

    with pytest.raises(RuntimeError) as failure:
        runtime._force_merge_controlled_index(
            object(), object(), SimpleNamespace(force_merge_segment_count=1)
        )

    assert failure.value is original_error
    writer.forceMerge.assert_called_once_with(1, True)
    writer.rollback.assert_called_once_with()
    writer.commit.assert_not_called()
    writer.close.assert_not_called()


def test_reused_jvm_reports_the_initialized_artifact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    fake_lucene = _FakeLucene()
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)
    config = {
        "cuvs_java_jar": str(java_jar),
        "cuvs_lucene_jar": str(lucene_jar),
    }

    first = initialize_pylucene(config)
    second = initialize_pylucene(config)

    assert len(fake_lucene.initializations) == 1
    assert second[1:] == first[1:]


def test_reused_jvm_rejects_a_same_path_artifact_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    fake_lucene = _FakeLucene()
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)
    config = {
        "cuvs_java_jar": str(java_jar),
        "cuvs_lucene_jar": str(lucene_jar),
    }
    initialize_pylucene(config)
    with zipfile.ZipFile(java_jar, "a") as archive:
        archive.writestr("replacement", b"changed")

    with pytest.raises(RuntimeError, match="changed after"):
        initialize_pylucene(config)


def test_reused_jvm_hashes_artifacts_when_stat_tokens_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    fake_lucene = _FakeLucene()
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)
    config = {
        "cuvs_java_jar": str(java_jar),
        "cuvs_lucene_jar": str(lucene_jar),
    }
    initialize_pylucene(config)
    original_tokens = dict(_lucene_runtime._INITIALIZED_ARTIFACT_TOKENS or {})
    replacement = bytearray(java_jar.read_bytes())
    replacement[len(replacement) // 2] ^= 1
    java_jar.write_bytes(replacement)
    monkeypatch.setattr(
        _lucene_runtime,
        "_artifact_stat_token",
        lambda path: original_tokens[str(path)],
    )

    with pytest.raises(RuntimeError, match="changed after"):
        initialize_pylucene(config)


def test_artifact_hashing_occurs_once_per_operation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(LuceneRuntime)
    runtime._artifact_tokens = {}
    runtime.artifact_provenance = {}
    runtime.lucene = _FakeLucene()
    runtime.lucene.environment = _FakeEnvironment()
    hash_verifications = []
    monkeypatch.setattr(
        _lucene_runtime,
        "_verify_artifact_tokens",
        lambda _tokens, _provenance: hash_verifications.append(True),
    )

    runtime.verify_artifacts()
    runtime.attach_current_thread()
    runtime.attach_current_thread()

    assert hash_verifications == [True]


def test_reused_jvm_rejects_different_vm_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    java_jar, lucene_jar = _write_artifacts(tmp_path)
    fake_lucene = _FakeLucene()
    _reset_jvm_state(monkeypatch)
    monkeypatch.setattr(_lucene_runtime, "_load_pylucene", lambda: fake_lucene)
    config = {
        "cuvs_java_jar": str(java_jar),
        "cuvs_lucene_jar": str(lucene_jar),
    }
    initialize_pylucene(config)

    with pytest.raises(
        RuntimeError, match="different classpath or JVM arguments"
    ):
        initialize_pylucene({**config, "jvm_args": ["-Xmx1g"]})
