---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvsbenchfbinindexingbridge
---

# CuvsBenchFbinIndexingBridge

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class CuvsBenchFbinIndexingBridge implements Function<Map<String, Object>, Map<String, Object>>
```

Streams a validated FBIN prefix through ordinary Lucene `IndexWriter.addDocument` calls.

This class is a narrow interoperability bridge for generated bindings that cannot directly
wrap cuVS-specific classes. It intentionally implements only standard `Function` and
`Map` types. It is not a bulk, mapped, external-vector, or out-of-core index writer: every
vector is added to a normal Lucene writer, and the resulting index is self-contained.

The request to `apply(Map)` must contain exactly these entries:

`String`: `source_path`, `index_path`, and `expected_codec_name`. The
paths must be nonempty, and the expected codec name must equal the supplied codec's name.

`Codec`: `codec`.

`Long`: `expected_source_size`, `expected_file_vector_count`, and
`vector_count`. Source size must be nonnegative; both row counts must be positive; and
`vector_count` cannot exceed the file's row count or Lucene's document limit.

`Integer`: `expected_dimensions`, `expected_header_bytes`,
`premerge_segment_count`, `force_merge_segment_count`, and
`ram_per_thread_hard_limit_mb`. Dimensions and pre-merge segments must be positive, the
header is 8 or 16 bytes, and final force merge must be zero.

`Boolean`: `allow_unsupported_lucene_ram_limit`. A RAM limit of 2048 MiB or
greater requires `true`; lower values require `false`. The larger limit deliberately
relies on a verified, non-public Lucene field override and is supported only by this controlled,
no-merge build path. `vector_count` must divide evenly into
`premerge_segment_count`; those equal contiguous partitions are built sequentially using
`CREATE` mode for the first and `APPEND` mode thereafter. Automatic and final merges
are disabled, so a successful build has exactly one segment per requested partition.

The response contains these `String` entries: `codec_name` (the configured codec
name), `vector_payload_sha256` (lowercase hexadecimal), `ingest_merge_policy`, and
`ram_per_thread_hard_limit_application`. It contains these `Integer` entries:
`dimensions`, `header_bytes`,
`premerge_segment_count`, `force_merge_segment_count`, `segment_count`,
`max_buffered_docs`, and `applied_ram_per_thread_hard_limit_mb`. It contains these
`Long` entries: `source_file_size`, `source_file_vector_count`,
`vector_count`, `indexed_payload_bytes`, and one key formed from the
`premerge_segment_vector_count_` prefix followed by each zero-based decimal segment number.

All response timing values are boxed `Long` nanoseconds: `directory_open_ns`,
`writer_setup_ns`, `document_ingest_ns`, `fbin_read_ns`,
`force_merge_ns`, `writer_commit_close_ns`, `post_build_reader_ns`,
`directory_close_ns`, and `runtime_build_wall_ns`. The runtime wall encloses the
complete call. The FBIN-read timer is nested within document ingestion; the remaining leaf phases
are disjoint. Force-merge time is always zero because this bridge rejects final force merge.

The target must be an existing, empty, non-symbolic-link directory. This bridge never removes,
replaces, or publishes that directory. Its caller owns the staging lifecycle and must discard
the whole directory after any failure. Rollback applies only to the active writer; if a later
partition fails, earlier committed partitions can remain in the unpublished staging directory.
Invalid requests and data raise `IllegalArgumentException`; I/O failures are wrapped in
`UncheckedIOException`; codec and Lucene failures propagate as runtime exceptions or
errors.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuvsBenchFbinIndexingBridge.java:102`_
