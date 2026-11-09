# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-control contracts without a JVM; live tests establish actual topology."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from _lucene_test_support import RecordingRuntime, _backend_and_index, _dataset
from cuvs_bench.backends import lucene as backend_module
from cuvs_bench.backends._lucene_runtime import (
    LuceneRuntime,
    RuntimeBuildTopology,
    _controlled_build_topology,
)
from cuvs_bench.backends.lucene import (
    ACCELERATED_HNSW_ALGORITHM,
    CAGRA_ALGORITHM,
    CPU_HNSW_ALGORITHM,
    LuceneBackend,
    LuceneConfigLoader,
    _build_controls,
    _build_parameters_for,
    _controlled_index_suffix,
)


def _two_segment_parameters():
    return {
        "premerge_segment_count": 2,
        "force_merge_segment_count": 0,
        "ram_per_thread_hard_limit_mb": 512,
    }


def _fake_two_segment_evidence():
    """Fabricated boundary output, not evidence that Lucene built any segments."""
    return RuntimeBuildTopology(
        requested_premerge_segment_count=2,
        observed_premerge_segment_count=2,
        requested_force_merge_segment_count=0,
        premerge_segment_vector_counts=(2, 2),
        max_buffered_docs=3,
        applied_ram_per_thread_hard_limit_mb=512,
        ram_per_thread_hard_limit_application="public_setter",
        ingest_merge_policy="NoMergePolicy",
        final_merge_policy=None,
    )


class FakeControlledRuntime(RecordingRuntime):
    """Supply explicit fake JVM output to test backend forwarding/publication."""

    def __init__(self, *, fake_topology=None):
        super().__init__(pylucene_version="i-am-fake-data")
        self.fake_topology = fake_topology
        self.received_controls = []

    def build_index(
        self, index_path, vectors, codec_name, build_parameters=None
    ):
        self.received_controls.append(build_parameters)
        result = super().build_index(index_path, vectors, codec_name)
        self.segment_count = 2 if self.fake_topology else 1
        return replace(
            result,
            segment_count=self.segment_count,
            topology=self.fake_topology,
        )


@pytest.mark.parametrize(
    "algorithm",
    [ACCELERATED_HNSW_ALGORITHM, CAGRA_ALGORITHM, CPU_HNSW_ALGORITHM],
)
def test_default_build_does_not_invent_controls_or_change_index_names(
    algorithm,
):
    assert _build_controls(algorithm, {}) == {}
    assert _controlled_index_suffix({}) == ""


@pytest.mark.parametrize(
    "parameters, error",
    [
        pytest.param(
            {"m": 16}, "both m and beam_width", id="missing-beam-width"
        ),
        pytest.param(
            {"m": True, "beam_width": 100},
            "m must be an integer",
            id="boolean-is-not-m",
        ),
        pytest.param(
            {"m": 16, "beam_width": 0},
            "beam_width must be an integer",
            id="zero-beam-width",
        ),
        pytest.param(
            {"m": 513, "beam_width": 100},
            "m must be an integer",
            id="m-above-api-limit",
        ),
        pytest.param(
            {"premerge_segment_count": 2}, "missing", id="incomplete-topology"
        ),
        pytest.param(
            {**_two_segment_parameters(), "force_merge_segment_count": 2},
            "0 .* or 1",
            id="unsupported-merge-target",
        ),
        pytest.param(
            {
                **_two_segment_parameters(),
                "ram_per_thread_hard_limit_mb": 2048,
            },
            "require allow_unsupported",
            id="large-limit-without-opt-in",
        ),
        pytest.param(
            {
                **_two_segment_parameters(),
                "allow_unsupported_lucene_ram_limit": True,
            },
            "requires ram_per_thread",
            id="unused-unsupported-opt-in",
        ),
        pytest.param(
            {
                **_two_segment_parameters(),
                "ram_per_thread_hard_limit_mb": 6144,
                "allow_unsupported_lucene_ram_limit": True,
                "force_merge_segment_count": 1,
            },
            "must be 0",
            id="unsupported-limit-cannot-merge",
        ),
    ],
)
def test_invalid_build_controls_fail_before_creating_a_runtime(
    parameters, error
):
    with pytest.raises(ValueError, match=error):
        _build_parameters_for(ACCELERATED_HNSW_ALGORITHM, parameters)


def test_gpu_cagra_disallows_force_merge():
    with pytest.raises(
        ValueError, match="requires force_merge_segment_count=0"
    ):
        _build_parameters_for(
            CAGRA_ALGORITHM,
            {**_two_segment_parameters(), "force_merge_segment_count": 1},
        )


def test_stock_cpu_hnsw_does_not_silently_accept_accelerated_build_controls():
    with pytest.raises(
        ValueError, match="Unsupported Lucene build parameters"
    ):
        _build_parameters_for(CPU_HNSW_ALGORITHM, {"m": 16, "beam_width": 100})


@pytest.mark.parametrize(
    "vector_count, error", [(3, "equal partitions"), (1, "cannot exceed")]
)
def test_controlled_partition_sizes_are_checked_before_opening_a_writer(
    vector_count, error
):
    with pytest.raises(RuntimeError, match=error):
        _controlled_build_topology(_two_segment_parameters(), vector_count)


def test_backend_forwards_and_persists_configured_build_parameters(tmp_path):
    fake_runtime = FakeControlledRuntime(
        fake_topology=_fake_two_segment_evidence()
    )
    backend, index, _ = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    controls = {"m": 16, "beam_width": 100, **_two_segment_parameters()}
    index = replace(index, build_param={**index.build_param, **controls})

    result = backend.build(_dataset(), [index])

    assert result.success, result.error_message
    assert fake_runtime.received_controls == [controls]
    manifest = json.loads(
        (Path(index.file) / ".cuvs-bench-lucene.json").read_text()
    )
    assert manifest["build_parameters"] == controls
    assert manifest["runtime_build_topology"][
        "premerge_segment_vector_counts"
    ] == [2, 2]
    assert result.metadata["pylucene_version"] == "i-am-fake-data"


def test_different_build_parameters_cannot_reuse_the_same_index(tmp_path):
    fake_runtime = FakeControlledRuntime()
    backend, index, _ = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    original = replace(
        index, build_param={**index.build_param, "m": 16, "beam_width": 100}
    )
    assert backend.build(_dataset(), [original]).success

    changed = replace(original, build_param={**original.build_param, "m": 32})
    result = backend.build(_dataset(), [changed])

    assert not result.success
    assert "different build parameters" in result.error_message
    assert len(fake_runtime.build_calls) == 1


def test_configured_saved_index_remains_usable_when_writer_default_changes(
    tmp_path, monkeypatch
):
    """Fake codec names test routing only, not Lucene version compatibility."""
    fake_runtime = FakeControlledRuntime()
    backend, index, factory = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    controls = {"m": 16, "beam_width": 100}
    index = replace(index, build_param={**index.build_param, **controls})
    assert backend.build(_dataset(), [index]).success
    saved_codec = backend.codec
    fake_new_codec = "i-am-a-fake-new-writer"
    monkeypatch.setitem(
        backend_module._CODEC_BY_ALGORITHM,
        ACCELERATED_HNSW_ALGORITHM,
        fake_new_codec,
    )
    backend = LuceneBackend(
        {**backend.config, "codec": fake_new_codec}, runtime_factory=factory
    )
    index = replace(index, build_param={"codec": fake_new_codec, **controls})

    reused = backend.build(_dataset(), [index])
    searched = backend.search(_dataset(), [index], k=2)[0]

    assert reused.success, reused.error_message
    assert searched.success, searched.error_message
    assert (
        reused.metadata["codec"] == searched.metadata["codec"] == saved_codec
    )
    assert len(fake_runtime.build_calls) == 1


@pytest.mark.parametrize(
    "fake_counts",
    [(4,), (1, 3)],
    ids=["wrong-segment-count", "unequal-segment-sizes"],
)
def test_runtime_rejects_unexpected_committed_partition_layout(fake_counts):
    runtime = object.__new__(LuceneRuntime)
    # Only the I/O seams are fake: the production orchestration compares these
    # independent observed counts with the requested two equal partitions.
    runtime._write_controlled_chunk = Mock(
        return_value=(0, 0, 0, "public_setter")
    )
    runtime._committed_segment_vector_counts = Mock(return_value=fake_counts)
    fake_directory = object()
    fake_codec = object()
    parameters = _controlled_build_topology(_two_segment_parameters(), 4)

    with pytest.raises(RuntimeError, match="ingest topology mismatch"):
        runtime._build_controlled_index(
            fake_directory,
            np.zeros((4, 2), dtype=np.float32),
            fake_codec,
            parameters,
        )


@pytest.mark.parametrize("failed_operation", ["addDocument", "flush"])
def test_failed_partition_rolls_back_without_committing(failed_operation):
    fake_writer = Mock(
        spec=["addDocument", "flush", "commit", "close", "rollback"]
    )
    original_error = RuntimeError("deliberately injected writer failure")
    getattr(fake_writer, failed_operation).side_effect = original_error
    runtime = object.__new__(LuceneRuntime)
    runtime.IndexWriter = Mock(return_value=fake_writer)
    fake_config = object()
    runtime._controlled_ingest_config = Mock(
        return_value=(fake_config, "public_setter")
    )
    fake_document = object()
    runtime._document = Mock(return_value=fake_document)
    fake_directory = object()
    fake_codec = object()

    with pytest.raises(RuntimeError) as failure:
        runtime._write_controlled_chunk(
            fake_directory,
            np.zeros((1, 2), dtype=np.float32),
            fake_codec,
            SimpleNamespace(),
            start=0,
            stop=1,
            create=True,
        )

    assert failure.value is original_error
    fake_writer.rollback.assert_called_once_with()
    fake_writer.commit.assert_not_called()
    fake_writer.close.assert_not_called()


def test_failed_force_merge_rolls_back_without_committing():
    fake_writer = Mock(spec=["forceMerge", "commit", "close", "rollback"])
    original_error = RuntimeError("deliberately injected force-merge failure")
    fake_writer.forceMerge.side_effect = original_error
    runtime = object.__new__(LuceneRuntime)
    runtime.IndexWriter = Mock(return_value=fake_writer)
    fake_config = object()
    fake_directory = object()
    fake_codec = object()
    runtime._controlled_merge_config = Mock(
        return_value=(fake_config, "public_setter")
    )

    with pytest.raises(RuntimeError) as failure:
        runtime._force_merge_controlled_index(
            fake_directory,
            fake_codec,
            SimpleNamespace(force_merge_segment_count=1),
        )

    assert failure.value is original_error
    fake_writer.forceMerge.assert_called_once_with(1, True)
    fake_writer.rollback.assert_called_once_with()
    fake_writer.commit.assert_not_called()
    fake_writer.close.assert_not_called()


def test_missing_topology_evidence_cannot_replace_a_previous_index(tmp_path):
    fake_runtime = FakeControlledRuntime()
    backend, index, _ = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    assert backend.build(_dataset(), [index]).success
    manifest_path = Path(index.file) / ".cuvs-bench-lucene.json"
    previous_manifest = manifest_path.read_bytes()
    controlled = replace(
        index, build_param={**index.build_param, **_two_segment_parameters()}
    )

    result = backend.build(_dataset(), [controlled], force=True)

    assert not result.success
    assert "missing evidence" in result.error_message
    assert manifest_path.read_bytes() == previous_manifest


@pytest.mark.parametrize(
    "field, corrupted_value",
    [
        pytest.param(
            "observed_premerge_segment_count", True, id="boolean-count"
        ),
        pytest.param(
            "premerge_segment_vector_counts", [1, 3], id="unequal-partitions"
        ),
        pytest.param("max_buffered_docs", 2, id="flush-threshold-mismatch"),
        pytest.param(
            "applied_ram_per_thread_hard_limit_mb", 1024, id="wrong-ram-limit"
        ),
        pytest.param(
            "ram_per_thread_hard_limit_application",
            "unsupported_field_override",
            id="wrong-application-mode",
        ),
        pytest.param(
            "final_merge_policy", "TieredMergePolicy", id="unrequested-merge"
        ),
    ],
)
def test_saved_topology_must_agree_with_requested_controls_before_reuse(
    tmp_path, field, corrupted_value
):
    fake_runtime = FakeControlledRuntime(
        fake_topology=_fake_two_segment_evidence()
    )
    backend, index, _ = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    index = replace(
        index, build_param={**index.build_param, **_two_segment_parameters()}
    )
    assert backend.build(_dataset(), [index]).success
    manifest_path = Path(index.file) / ".cuvs-bench-lucene.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_build_topology"][field] = corrupted_value
    manifest_path.write_text(json.dumps(manifest))

    reused = backend.build(_dataset(), [index])

    assert not reused.success
    assert "invalid runtime build topology" in reused.error_message
    assert len(fake_runtime.build_calls) == 1


def test_force_merge_must_produce_the_requested_single_segment():
    runtime = object.__new__(LuceneRuntime)
    runtime._write_controlled_chunk = Mock(
        return_value=(0, 0, 0, "public_setter")
    )
    runtime._force_merge_controlled_index = Mock(
        return_value=(0, 0, 0, "public_setter")
    )
    # The fabricated reader reports that the requested merge left both segments intact.
    runtime._committed_segment_vector_counts = Mock(
        side_effect=[(2, 2), (2, 2)]
    )
    fake_directory = object()
    fake_codec = object()
    topology = _controlled_build_topology(
        {**_two_segment_parameters(), "force_merge_segment_count": 1}, 4
    )

    with pytest.raises(RuntimeError, match="final topology mismatch"):
        runtime._build_controlled_index(
            fake_directory,
            np.zeros((4, 2), dtype=np.float32),
            fake_codec,
            topology,
        )


def test_build_parameter_sweep_assigns_distinct_index_files(tmp_path):
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text("- name: tiny-l2\n  distance: euclidean\n  dims: 2\n")
    algorithms = tmp_path / "algorithms.yaml"
    algorithms.write_text(
        f"name: {ACCELERATED_HNSW_ALGORITHM}\n"
        "groups:\n  custom:\n    build:\n      m: [16, 32]\n"
        "      beam_width: [100]\n    search:\n      num_candidates: [10]\n"
    )

    _, configurations = LuceneConfigLoader().load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(datasets),
        algorithm_configuration=str(algorithms),
        algorithms=ACCELERATED_HNSW_ALGORITHM,
        groups="custom",
    )

    indexes = [
        index
        for configuration in configurations
        for index in configuration.indexes
    ]
    assert [index.build_param["m"] for index in indexes] == [16, 32]
    assert len({index.file for index in indexes}) == 2


def test_controlled_base_build_cannot_replace_an_ordinary_named_group(
    tmp_path,
):
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text("- name: tiny-l2\n  distance: euclidean\n  dims: 2\n")
    algorithms = tmp_path / "algorithms.yaml"
    algorithms.write_text(
        f"name: {ACCELERATED_HNSW_ALGORITHM}\n"
        "groups:\n  base:\n    build:\n      m: [16]\n      beam_width: [100]\n"
        "  m16_bw100:\n    build: {}\n"
    )
    _, configurations = LuceneConfigLoader().load(
        dataset="tiny-l2",
        dataset_path=str(tmp_path),
        dataset_configuration=str(datasets),
        algorithm_configuration=str(algorithms),
        algorithms=ACCELERATED_HNSW_ALGORITHM,
        groups="base,m16_bw100",
    )
    controlled, ordinary = [
        configuration.indexes[0] for configuration in configurations
    ]
    assert ordinary.name == f"{ACCELERATED_HNSW_ALGORITHM}_m16_bw100"
    assert controlled.file != ordinary.file

    fake_runtime = FakeControlledRuntime()
    fake_environment_backend, _, fake_factory = _backend_and_index(
        tmp_path, ACCELERATED_HNSW_ALGORITHM, fake_runtime
    )
    backends = [
        LuceneBackend(
            {
                **fake_environment_backend.config,
                **configuration.backend_config,
            },
            runtime_factory=fake_factory,
        )
        for configuration in configurations
    ]
    # Build both before reusing either: a shared path would overwrite or reject one.
    for backend, index in zip(backends, (controlled, ordinary)):
        result = backend.build(_dataset(), [index])
        assert result.success, result.error_message
    for backend, index in zip(backends, (controlled, ordinary)):
        result = backend.build(_dataset(), [index])
        assert result.success, result.error_message
    assert len(fake_runtime.build_calls) == 2
