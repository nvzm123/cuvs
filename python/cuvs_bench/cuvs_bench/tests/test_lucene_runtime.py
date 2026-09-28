#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for immutable JVM setup and Java artifact provenance."""

import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuvs_bench.backends import _lucene_runtime
from cuvs_bench.backends._lucene_runtime import (
    ACCELERATED_HNSW_CODEC,
    CONFIGURED_ACCELERATED_HNSW_CODEC,
    HNSW_BEAM_WIDTH_PROPERTY,
    HNSW_MAX_CONN_PROPERTY,
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
def test_controlled_topology_derives_exact_equal_segments(
    vector_count: int,
    premerge_segments: int,
    force_merge_segments: int,
    chunk_size: int,
) -> None:
    topology = _controlled_build_topology(
        {
            "premerge_segment_count": premerge_segments,
            "force_merge_segment_count": force_merge_segments,
            "ram_per_thread_hard_limit_mb": 61_440,
        },
        vector_count,
    )

    assert topology is not None
    assert topology.premerge_segment_count == premerge_segments
    assert topology.force_merge_segment_count == force_merge_segments
    assert topology.ram_per_thread_hard_limit_mb == 61_440
    assert topology.chunk_size == chunk_size
    assert topology.max_buffered_docs == chunk_size + 1


@pytest.mark.parametrize("force_merge_segments", (0, 1))
def test_num_indexing_threads_uses_partition_flush_boundary(
    force_merge_segments: int,
) -> None:
    topology = _controlled_build_topology(
        {
            "num_indexing_threads": 4,
            "force_merge_segment_count": force_merge_segments,
            "ram_per_thread_hard_limit_mb": 61_440,
        },
        100_000_000,
    )

    assert topology is not None
    assert topology.requested_premerge_segment_count is None
    assert topology.num_indexing_threads == 4
    assert topology.premerge_segment_count == 4
    assert topology.force_merge_segment_count == force_merge_segments
    assert topology.chunk_size == 25_000_000
    assert topology.max_buffered_docs == 25_000_001


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
                "premerge_segment_count": 4,
                "num_indexing_threads": 4,
                "force_merge_segment_count": 0,
                "ram_per_thread_hard_limit_mb": 61_440,
            },
            100,
            "cannot combine",
        ),
        (
            {
                "premerge_segment_count": True,
                "force_merge_segment_count": 1,
                "ram_per_thread_hard_limit_mb": 61_440,
            },
            100,
            "must be a positive integer",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": 2,
                "ram_per_thread_hard_limit_mb": 61_440,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": True,
                "ram_per_thread_hard_limit_mb": 61_440,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": None,
                "ram_per_thread_hard_limit_mb": 61_440,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": "0",
                "ram_per_thread_hard_limit_mb": 61_440,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": 0.0,
                "ram_per_thread_hard_limit_mb": 61_440,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": -1,
                "ram_per_thread_hard_limit_mb": 61_440,
            },
            100,
            r"must be 0 \(disabled\) or 1",
        ),
        (
            {
                "premerge_segment_count": 4,
                "force_merge_segment_count": 1,
                "ram_per_thread_hard_limit_mb": 61_440,
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
            "com/nvidia/cuvs/lucene/CuvsBenchFbinIndexingBridge.class", b""
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/IndexSearcherTimingBridge.class", b""
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/"
            "IndexWriterConfigPerThreadHardLimitBridge.class",
            b"",
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.class", b""
        )
        archive.writestr(
            "com/nvidia/cuvs/lucene/Lucene101ConfiguredHNSWCodec.class", b""
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


def test_java_fbin_indexer_reports_class_loading_failure() -> None:
    class MissingBridgeClass:
        @staticmethod
        def forName(_name: str):
            raise RuntimeError("unsupported FBIN bridge bytecode")

    runtime = object.__new__(LuceneRuntime)
    runtime.Class = MissingBridgeClass

    with pytest.raises(RuntimeError) as failure:
        runtime._load_java_fbin_indexer()

    assert str(failure.value) == (
        "Could not load or adapt "
        "com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge through "
        "PyLucene/JCC: RuntimeError: unsupported FBIN bridge bytecode"
    )
    assert isinstance(failure.value.__cause__, RuntimeError)


def test_java_fbin_build_uses_scalar_map_contract_and_validates_result(
    tmp_path: Path,
) -> None:
    class Box:
        def __init__(self, value: int) -> None:
            self.value = value

        def intValue(self) -> int:
            return self.value

        def longValue(self) -> int:
            return self.value

    class NumberBinding:
        @staticmethod
        def valueOf(value: int) -> Box:
            return Box(value)

        @staticmethod
        def cast_(value: object) -> Box:
            assert isinstance(value, Box)
            return value

    class JavaMap(dict):
        def put(self, key: str, value: object) -> None:
            self[key] = value

        def containsKey(self, key: str) -> bool:
            return key in self

        def get(self, key: str) -> object:
            return self[key]

    class MapBinding:
        @staticmethod
        def cast_(value: object) -> JavaMap:
            assert isinstance(value, JavaMap)
            return value

    response = JavaMap()
    strings = {
        "codec_name": ACCELERATED_HNSW_CODEC,
        "vector_payload_sha256": "d" * 64,
        "indexing_execution_mode": "partitioned_sequential",
        "ingest_merge_policy": "NoMergePolicy",
    }
    integers = {
        "dimensions": 2,
        "header_bytes": 8,
        "num_indexing_threads": 4,
        "force_merge_segment_count": 0,
        "actual_indexing_thread_count": 1,
        "max_concurrent_indexing_threads": 1,
        "segment_count": 4,
        "max_buffered_docs": 3,
        "applied_ram_per_thread_hard_limit_mb": 61_440,
    }
    longs = {
        "source_file_size": 72,
        "source_file_vector_count": 8,
        "vector_count": 8,
        "indexed_payload_bytes": 64,
        "directory_open_ns": 1,
        "writer_setup_ns": 2,
        "document_ingest_ns": 30,
        "fbin_read_ns": 20,
        "writer_commit_close_ns": 4,
        "post_build_reader_ns": 5,
        "directory_close_ns": 6,
        "runtime_build_wall_ns": 48,
        **{
            f"premerge_segment_vector_count_{partition}": 2
            for partition in range(4)
        },
    }
    response.update(strings)
    response.update({name: Box(value) for name, value in integers.items()})
    response.update({name: Box(value) for name, value in longs.items()})
    requests = []

    class Bridge:
        @staticmethod
        def apply(request: JavaMap) -> JavaMap:
            requests.append(request)
            return response

    configured_codec = object()
    runtime = object.__new__(LuceneRuntime)
    runtime.attach_current_thread = lambda: None
    runtime._resolve_build_codec = lambda *_args: configured_codec
    runtime._java_fbin_indexer = Bridge()
    runtime.HashMap = JavaMap
    runtime.Map = MapBinding
    runtime.Integer = NumberBinding
    runtime.Long = NumberBinding
    source = tmp_path / "base.fbin"
    parameters = {
        "codec": ACCELERATED_HNSW_CODEC,
        "m": 16,
        "beam_width": 80,
        "num_indexing_threads": 4,
        "force_merge_segment_count": 0,
        "ram_per_thread_hard_limit_mb": 61_440,
    }

    result = runtime.build_index_from_fbin(
        tmp_path / "index",
        source,
        expected_source_size=72,
        expected_file_vector_count=8,
        expected_dimensions=2,
        expected_header_bytes=8,
        vector_count=8,
        codec_name=ACCELERATED_HNSW_CODEC,
        build_parameters=parameters,
    )

    [request] = requests
    assert set(request) == {
        "source_path",
        "index_path",
        "codec",
        "expected_codec_name",
        "expected_source_size",
        "expected_file_vector_count",
        "expected_dimensions",
        "expected_header_bytes",
        "vector_count",
        "num_indexing_threads",
        "force_merge_segment_count",
        "ram_per_thread_hard_limit_mb",
    }
    assert request["source_path"] == str(source)
    assert request["index_path"] == str(tmp_path / "index")
    assert request["codec"] is configured_codec
    assert request["expected_codec_name"] == ACCELERATED_HNSW_CODEC
    assert request["expected_source_size"].longValue() == 72
    assert request["expected_file_vector_count"].longValue() == 8
    assert request["expected_dimensions"].intValue() == 2
    assert request["expected_header_bytes"].intValue() == 8
    assert request["vector_count"].longValue() == 8
    assert request["num_indexing_threads"].intValue() == 4
    assert request["force_merge_segment_count"].intValue() == 0
    assert request["ram_per_thread_hard_limit_mb"].intValue() == 61_440
    assert result.vector_payload_sha256 == "d" * 64
    assert result.indexed_payload_bytes == 64
    assert result.timing.fbin_read_ns == 20
    assert result.topology.premerge_segment_vector_counts == (2, 2, 2, 2)
    assert result.topology.final_merge_policy is None

    response["codec_name"] = "unexpected-codec"
    with pytest.raises(RuntimeError, match="response mismatch for codec_name"):
        runtime.build_index_from_fbin(
            tmp_path / "index",
            source,
            expected_source_size=72,
            expected_file_vector_count=8,
            expected_dimensions=2,
            expected_header_bytes=8,
            vector_count=8,
            codec_name=ACCELERATED_HNSW_CODEC,
            build_parameters=parameters,
        )


class _RecordingJavaSystem:
    def __init__(self, properties: dict[str, str] | None = None) -> None:
        self.properties = dict(properties or {})

    def getProperty(self, name: str):
        return self.properties.get(name)

    def setProperty(self, name: str, value: str):
        previous = self.properties.get(name)
        self.properties[name] = value
        return previous

    def clearProperty(self, name: str):
        return self.properties.pop(name, None)


class _ConfiguredCodec:
    def __init__(self, max_conn: int, beam_width: int) -> None:
        self.max_conn = max_conn
        self.beam_width = beam_width

    @staticmethod
    def getName() -> str:
        return ACCELERATED_HNSW_CODEC

    @staticmethod
    def knnVectorsFormat() -> object:
        return object()

    def __str__(self) -> str:
        return (
            "Lucene101ConfiguredHNSWCodec["
            f"maxConn={self.max_conn}, beamWidth={self.beam_width}]"
        )


def _configured_codec_runtime(
    system: _RecordingJavaSystem,
    *,
    constructor_error: Exception | None = None,
) -> tuple[LuceneRuntime, list[dict[str, str]], list[str]]:
    snapshots: list[dict[str, str]] = []
    class_names: list[str] = []

    class ReflectedCodec:
        @staticmethod
        def newInstance():
            snapshots.append(dict(system.properties))
            if constructor_error is not None:
                raise constructor_error
            return _ConfiguredCodec(
                int(system.properties[HNSW_MAX_CONN_PROPERTY]),
                int(system.properties[HNSW_BEAM_WIDTH_PROPERTY]),
            )

    class JavaClass:
        @staticmethod
        def forName(name: str):
            class_names.append(name)
            return ReflectedCodec()

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
    runtime.System = system
    runtime.Class = JavaClass
    runtime.Codec = CodecBinding
    runtime.attach_current_thread = lambda: None
    return runtime, snapshots, class_names


def test_configured_hnsw_codec_snapshots_and_restores_java_properties() -> (
    None
):
    system = _RecordingJavaSystem({HNSW_MAX_CONN_PROPERTY: "previous"})
    runtime, snapshots, class_names = _configured_codec_runtime(system)

    codec = runtime.resolve_configured_hnsw_codec(16, 80)

    assert str(codec) == (
        "Lucene101ConfiguredHNSWCodec[maxConn=16, beamWidth=80]"
    )
    assert snapshots == [
        {
            HNSW_MAX_CONN_PROPERTY: "16",
            HNSW_BEAM_WIDTH_PROPERTY: "80",
        }
    ]
    assert class_names == [CONFIGURED_ACCELERATED_HNSW_CODEC]
    assert system.properties == {HNSW_MAX_CONN_PROPERTY: "previous"}


def test_configured_hnsw_codec_restores_properties_after_constructor_failure() -> (
    None
):
    system = _RecordingJavaSystem(
        {
            HNSW_MAX_CONN_PROPERTY: "previous-m",
            HNSW_BEAM_WIDTH_PROPERTY: "previous-beam",
        }
    )
    runtime, snapshots, _class_names = _configured_codec_runtime(
        system, constructor_error=RuntimeError("constructor failed")
    )

    with pytest.raises(RuntimeError, match="constructor failed"):
        runtime.resolve_configured_hnsw_codec(16, 80)

    assert snapshots == [
        {
            HNSW_MAX_CONN_PROPERTY: "16",
            HNSW_BEAM_WIDTH_PROPERTY: "80",
        }
    ]
    assert system.properties == {
        HNSW_MAX_CONN_PROPERTY: "previous-m",
        HNSW_BEAM_WIDTH_PROPERTY: "previous-beam",
    }


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
