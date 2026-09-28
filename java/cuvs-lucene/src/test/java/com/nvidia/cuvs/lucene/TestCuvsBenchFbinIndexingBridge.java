/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.ACTUAL_INDEXING_THREAD_COUNT_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.APPLIED_RAM_PER_THREAD_HARD_LIMIT_MB_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.CODEC_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.CODEC_NAME_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.DIMENSIONS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.DIRECTORY_CLOSE_NS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.DIRECTORY_OPEN_NS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.DOCUMENT_INGEST_NS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.EXPECTED_CODEC_NAME_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.EXPECTED_DIMENSIONS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.EXPECTED_FILE_VECTOR_COUNT_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.EXPECTED_HEADER_BYTES_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.EXPECTED_SOURCE_SIZE_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.FBIN_READ_NS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.FORCE_MERGE_SEGMENT_COUNT_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.HEADER_BYTES_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.INDEXED_PAYLOAD_BYTES_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.INDEXING_EXECUTION_MODE_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.INDEX_PATH_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.INGEST_MERGE_POLICY_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.MAX_BUFFERED_DOCS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.MAX_CONCURRENT_INDEXING_THREADS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.NUM_INDEXING_THREADS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.POST_BUILD_READER_NS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.PREMERGE_SEGMENT_VECTOR_COUNT_PREFIX;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.RAM_PER_THREAD_HARD_LIMIT_MB_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.RUNTIME_BUILD_WALL_NS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.SEGMENT_COUNT_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.SOURCE_FILE_SIZE_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.SOURCE_FILE_VECTOR_COUNT_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.SOURCE_PATH_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.VECTOR_COUNT_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.VECTOR_PAYLOAD_SHA256_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.WRITER_COMMIT_CLOSE_NS_KEY;
import static com.nvidia.cuvs.lucene.CuvsBenchFbinIndexingBridge.WRITER_SETUP_NS_KEY;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.FSDirectory;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class TestCuvsBenchFbinIndexingBridge {
  private static final int LEGACY_HEADER_BYTES = 8;
  private static final int EXTENDED_HEADER_BYTES = 16;
  private static final int HARD_LIMIT_MB = 61_440;

  @Rule public final TemporaryFolder temporary = new TemporaryFolder();

  @Test
  public void testClassicFbinBuildsFourExactSegmentsWithReusableDocumentAndVector()
      throws Exception {
    Path root = temporary.newFolder("classic-four-partitions").toPath();
    float[][] vectors = distinctVectors(8, 3);
    Path source = writeFbin(root.resolve("base.fbin"), vectors, false);
    Path index = Files.createDirectory(root.resolve("index"));
    Codec codec = new Lucene101Codec();
    Function<Map<String, Object>, Map<String, Object>> bridge = new CuvsBenchFbinIndexingBridge();

    Map<String, Object> response =
        bridge.apply(request(source, index, codec, vectors.length, 3, 8, 4, 0));

    assertEquals(codec.getName(), response.get(CODEC_NAME_KEY));
    assertEquals(Files.size(source), response.get(SOURCE_FILE_SIZE_KEY));
    assertEquals(8L, response.get(SOURCE_FILE_VECTOR_COUNT_KEY));
    assertEquals(3, response.get(DIMENSIONS_KEY));
    assertEquals(LEGACY_HEADER_BYTES, response.get(HEADER_BYTES_KEY));
    assertEquals(8L, response.get(VECTOR_COUNT_KEY));
    assertEquals(8L * 3L * Float.BYTES, response.get(INDEXED_PAYLOAD_BYTES_KEY));
    assertEquals(payloadSha256(vectors, 8), response.get(VECTOR_PAYLOAD_SHA256_KEY));
    assertEquals(4, response.get(NUM_INDEXING_THREADS_KEY));
    assertEquals(0, response.get(FORCE_MERGE_SEGMENT_COUNT_KEY));
    assertEquals(1, response.get(ACTUAL_INDEXING_THREAD_COUNT_KEY));
    assertEquals(1, response.get(MAX_CONCURRENT_INDEXING_THREADS_KEY));
    assertEquals("partitioned_sequential", response.get(INDEXING_EXECUTION_MODE_KEY));
    assertEquals(4, response.get(SEGMENT_COUNT_KEY));
    for (int segment = 0; segment < 4; segment++) {
      assertEquals(2L, response.get(PREMERGE_SEGMENT_VECTOR_COUNT_PREFIX + segment));
    }
    assertEquals(3, response.get(MAX_BUFFERED_DOCS_KEY));
    assertEquals(HARD_LIMIT_MB, response.get(APPLIED_RAM_PER_THREAD_HARD_LIMIT_MB_KEY));
    assertEquals("NoMergePolicy", response.get(INGEST_MERGE_POLICY_KEY));
    assertNonNegativeTiming(response, DIRECTORY_OPEN_NS_KEY);
    assertNonNegativeTiming(response, WRITER_SETUP_NS_KEY);
    assertNonNegativeTiming(response, DOCUMENT_INGEST_NS_KEY);
    assertNonNegativeTiming(response, FBIN_READ_NS_KEY);
    assertNonNegativeTiming(response, WRITER_COMMIT_CLOSE_NS_KEY);
    assertNonNegativeTiming(response, POST_BUILD_READER_NS_KEY);
    assertNonNegativeTiming(response, DIRECTORY_CLOSE_NS_KEY);
    assertNonNegativeTiming(response, RUNTIME_BUILD_WALL_NS_KEY);
    assertTrue(
        (Long) response.get(FBIN_READ_NS_KEY) <= (Long) response.get(DOCUMENT_INGEST_NS_KEY));
    long disjointPhaseNanos =
        (Long) response.get(DIRECTORY_OPEN_NS_KEY)
            + (Long) response.get(WRITER_SETUP_NS_KEY)
            + (Long) response.get(DOCUMENT_INGEST_NS_KEY)
            + (Long) response.get(WRITER_COMMIT_CLOSE_NS_KEY)
            + (Long) response.get(POST_BUILD_READER_NS_KEY)
            + (Long) response.get(DIRECTORY_CLOSE_NS_KEY);
    assertTrue(disjointPhaseNanos <= (Long) response.get(RUNTIME_BUILD_WALL_NS_KEY));

    assertIndex(index, vectors, 8, 4);
  }

  @Test
  public void testExtendedFbinIndexesAndDigestsOnlySelectedPrefix() throws Exception {
    Path root = temporary.newFolder("extended-subset").toPath();
    float[][] vectors = distinctVectors(6, 4);
    Path source = writeFbin(root.resolve("base.fbin"), vectors, true);
    Path index = Files.createDirectory(root.resolve("index"));
    Codec codec = new Lucene101Codec();

    Map<String, Object> response =
        new CuvsBenchFbinIndexingBridge()
            .apply(request(source, index, codec, vectors.length, 4, 4, 1, 0));

    assertEquals(EXTENDED_HEADER_BYTES, response.get(HEADER_BYTES_KEY));
    assertEquals(4L, response.get(VECTOR_COUNT_KEY));
    assertEquals(4L * 4L * Float.BYTES, response.get(INDEXED_PAYLOAD_BYTES_KEY));
    assertEquals(payloadSha256(vectors, 4), response.get(VECTOR_PAYLOAD_SHA256_KEY));
    assertEquals(1, response.get(SEGMENT_COUNT_KEY));
    assertEquals(4L, response.get(PREMERGE_SEGMENT_VECTOR_COUNT_PREFIX + 0));
    assertIndex(index, vectors, 4, 1);
  }

  @Test
  public void testRejectsNonFiniteVectorAndRollsBackWithoutACommit() throws Exception {
    Path root = temporary.newFolder("non-finite").toPath();
    float[][] vectors = distinctVectors(4, 2);
    vectors[2][1] = Float.NaN;
    Path source = writeFbin(root.resolve("base.fbin"), vectors, false);
    Path index = Files.createDirectory(root.resolve("index"));
    Codec codec = new Lucene101Codec();

    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new CuvsBenchFbinIndexingBridge()
                    .apply(request(source, index, codec, 4, 2, 4, 1, 0)));

    assertEquals("non-finite float at vector 2, dimension 1", error.getMessage());
    try (var directory = FSDirectory.open(index)) {
      assertFalse(DirectoryReader.indexExists(directory));
    }
  }

  @Test
  public void testFailureInLaterPartitionRollsBackOnlyUncommittedWriter() throws Exception {
    Path root = temporary.newFolder("later-partition-failure").toPath();
    float[][] vectors = distinctVectors(4, 2);
    vectors[3][0] = Float.POSITIVE_INFINITY;
    Path source = writeFbin(root.resolve("base.fbin"), vectors, false);
    Path index = Files.createDirectory(root.resolve("index"));
    Codec codec = new Lucene101Codec();

    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new CuvsBenchFbinIndexingBridge()
                    .apply(request(source, index, codec, 4, 2, 4, 2, 0)));

    assertEquals("non-finite float at vector 3, dimension 0", error.getMessage());
    try (var directory = FSDirectory.open(index);
        var reader = DirectoryReader.open(directory)) {
      assertEquals(2, reader.numDocs());
      assertEquals(1, reader.leaves().size());
    }
  }

  @Test
  public void testRejectsMalformedTruncatedAndOverflowingHeadersBeforeWriting() throws Exception {
    Path root = temporary.newFolder("invalid-headers").toPath();
    Codec codec = new Lucene101Codec();

    Path padded = writeFbin(root.resolve("padded.fbin"), distinctVectors(2, 2), false);
    Files.write(padded, new byte[] {1}, java.nio.file.StandardOpenOption.APPEND);
    Path paddedIndex = Files.createDirectory(root.resolve("padded-index"));
    assertInvalidHeader(padded, paddedIndex, codec);

    Path truncated = root.resolve("truncated.fbin");
    Files.write(truncated, new byte[7]);
    Path truncatedIndex = Files.createDirectory(root.resolve("truncated-index"));
    assertInvalidHeader(truncated, truncatedIndex, codec);

    Path overflow = root.resolve("overflow.fbin");
    ByteBuffer overflowHeader =
        ByteBuffer.allocate(EXTENDED_HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
    overflowHeader.putLong(Long.MAX_VALUE).putLong(Long.MAX_VALUE);
    Files.write(overflow, overflowHeader.array());
    Path overflowIndex = Files.createDirectory(root.resolve("overflow-index"));
    assertInvalidHeader(overflow, overflowIndex, codec);
  }

  @Test
  public void testRejectsHeaderMetadataMismatchBeforeWriting() throws Exception {
    Path root = temporary.newFolder("metadata-mismatch").toPath();
    Path source = writeFbin(root.resolve("base.fbin"), distinctVectors(4, 2), false);
    Path index = Files.createDirectory(root.resolve("index"));
    Codec codec = new Lucene101Codec();
    Map<String, Object> request = request(source, index, codec, 4, 2, 4, 1, 0);
    request.put(EXPECTED_DIMENSIONS_KEY, 3);

    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class, () -> new CuvsBenchFbinIndexingBridge().apply(request));

    assertEquals(
        "expected_dimensions does not match the FBIN file: expected 3, observed 2",
        error.getMessage());
    assertDirectoryEmpty(index);
  }

  @Test
  public void testRejectsUnequalPartitionsAndForceMergeBeforeWriting() throws Exception {
    Path root = temporary.newFolder("invalid-topology").toPath();
    Path source = writeFbin(root.resolve("base.fbin"), distinctVectors(6, 2), false);
    Codec codec = new Lucene101Codec();

    Path unequalIndex = Files.createDirectory(root.resolve("unequal-index"));
    IllegalArgumentException unequal =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new CuvsBenchFbinIndexingBridge()
                    .apply(request(source, unequalIndex, codec, 6, 2, 6, 4, 0)));
    assertEquals(
        "vector_count must be divisible by num_indexing_threads for equal partitions",
        unequal.getMessage());
    assertDirectoryEmpty(unequalIndex);

    Path mergeIndex = Files.createDirectory(root.resolve("merge-index"));
    IllegalArgumentException forceMerge =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new CuvsBenchFbinIndexingBridge()
                    .apply(request(source, mergeIndex, codec, 6, 2, 6, 1, 1)));
    assertEquals("force_merge_segment_count must be 0", forceMerge.getMessage());
    assertDirectoryEmpty(mergeIndex);
  }

  @Test
  public void testRejectsNonemptyAndSymbolicLinkStagingDirectoriesWithoutDeletingThem()
      throws Exception {
    Path root = temporary.newFolder("unsafe-staging").toPath();
    Path source = writeFbin(root.resolve("base.fbin"), distinctVectors(4, 2), false);
    Codec codec = new Lucene101Codec();

    Path nonempty = Files.createDirectory(root.resolve("nonempty"));
    Path sentinel = nonempty.resolve("sentinel");
    Files.writeString(sentinel, "keep");
    IllegalArgumentException nonemptyError =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new CuvsBenchFbinIndexingBridge()
                    .apply(request(source, nonempty, codec, 4, 2, 4, 1, 0)));
    assertEquals("index_path must be empty", nonemptyError.getMessage());
    assertEquals("keep", Files.readString(sentinel));

    Path realDirectory = Files.createDirectory(root.resolve("real-index"));
    Path link = root.resolve("linked-index");
    Files.createSymbolicLink(link, realDirectory);
    IllegalArgumentException symlinkError =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new CuvsBenchFbinIndexingBridge()
                    .apply(request(source, link, codec, 4, 2, 4, 1, 0)));
    assertEquals(
        "index_path must be an existing, non-symbolic-link directory", symlinkError.getMessage());
    assertDirectoryEmpty(realDirectory);
  }

  @Test
  public void testRejectsIncompleteWronglyTypedAndUnknownRequests() throws Exception {
    CuvsBenchFbinIndexingBridge bridge = new CuvsBenchFbinIndexingBridge();
    assertEquals(
        "request must not be null",
        assertThrows(IllegalArgumentException.class, () -> bridge.apply(null)).getMessage());

    Map<String, Object> empty = new HashMap<>();
    assertEquals(
        "source_path must have type String",
        assertThrows(IllegalArgumentException.class, () -> bridge.apply(empty)).getMessage());

    Path root = temporary.newFolder("invalid-request").toPath();
    Path source = writeFbin(root.resolve("base.fbin"), distinctVectors(4, 2), false);
    Path index = Files.createDirectory(root.resolve("index"));
    Codec codec = new Lucene101Codec();
    Map<String, Object> wrongType = request(source, index, codec, 4, 2, 4, 1, 0);
    wrongType.put(VECTOR_COUNT_KEY, 4);
    assertEquals(
        "vector_count must have type Long",
        assertThrows(IllegalArgumentException.class, () -> bridge.apply(wrongType)).getMessage());

    Map<String, Object> unknown = request(source, index, codec, 4, 2, 4, 1, 0);
    unknown.put("typo", 1);
    assertEquals(
        "unsupported request key: typo",
        assertThrows(IllegalArgumentException.class, () -> bridge.apply(unknown)).getMessage());
  }

  private static Map<String, Object> request(
      Path source,
      Path index,
      Codec codec,
      long fileVectorCount,
      int dimensions,
      long selectedVectorCount,
      int partitions,
      int forceMerge)
      throws IOException {
    int headerBytes =
        Math.toIntExact(
            Files.size(source) - Math.multiplyExact(fileVectorCount * dimensions, Float.BYTES));
    Map<String, Object> request = new HashMap<>();
    request.put(SOURCE_PATH_KEY, source.toString());
    request.put(INDEX_PATH_KEY, index.toString());
    request.put(CODEC_KEY, codec);
    request.put(EXPECTED_CODEC_NAME_KEY, codec.getName());
    request.put(EXPECTED_SOURCE_SIZE_KEY, Files.size(source));
    request.put(EXPECTED_FILE_VECTOR_COUNT_KEY, fileVectorCount);
    request.put(EXPECTED_DIMENSIONS_KEY, dimensions);
    request.put(EXPECTED_HEADER_BYTES_KEY, headerBytes);
    request.put(VECTOR_COUNT_KEY, selectedVectorCount);
    request.put(NUM_INDEXING_THREADS_KEY, partitions);
    request.put(FORCE_MERGE_SEGMENT_COUNT_KEY, forceMerge);
    request.put(RAM_PER_THREAD_HARD_LIMIT_MB_KEY, HARD_LIMIT_MB);
    return request;
  }

  private static Path writeFbin(Path path, float[][] vectors, boolean extended) throws IOException {
    int dimensions = vectors[0].length;
    int headerBytes = extended ? EXTENDED_HEADER_BYTES : LEGACY_HEADER_BYTES;
    ByteBuffer bytes =
        ByteBuffer.allocate(
                Math.addExact(
                    headerBytes,
                    Math.multiplyExact(
                        Math.multiplyExact(vectors.length, dimensions), Float.BYTES)))
            .order(ByteOrder.LITTLE_ENDIAN);
    if (extended) {
      bytes.putLong(vectors.length).putLong(dimensions);
    } else {
      bytes.putInt(vectors.length).putInt(dimensions);
    }
    for (float[] vector : vectors) {
      for (float value : vector) {
        bytes.putFloat(value);
      }
    }
    Files.write(path, bytes.array());
    return path;
  }

  private static float[][] distinctVectors(int rows, int dimensions) {
    float[][] vectors = new float[rows][dimensions];
    for (int row = 0; row < rows; row++) {
      for (int dimension = 0; dimension < dimensions; dimension++) {
        vectors[row][dimension] = row * 100.0f + dimension + 0.25f;
      }
    }
    return vectors;
  }

  private static String payloadSha256(float[][] vectors, int selectedRows)
      throws NoSuchAlgorithmException {
    ByteBuffer payload =
        ByteBuffer.allocate(selectedRows * vectors[0].length * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    for (int row = 0; row < selectedRows; row++) {
      for (float value : vectors[row]) {
        payload.putFloat(value);
      }
    }
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    return HexFormat.of().formatHex(digest.digest(payload.array()));
  }

  private static void assertIndex(
      Path index, float[][] expectedVectors, int selectedRows, int expectedSegments)
      throws IOException {
    try (var directory = FSDirectory.open(index);
        var reader = DirectoryReader.open(directory)) {
      assertEquals(selectedRows, reader.numDocs());
      assertEquals(expectedSegments, reader.leaves().size());
      int observed = 0;
      for (var context : reader.leaves()) {
        LeafReader leaf = context.reader();
        NumericDocValues ids = leaf.getNumericDocValues("id");
        FloatVectorValues vectors = leaf.getFloatVectorValues("vector");
        KnnVectorValues.DocIndexIterator iterator = vectors.iterator();
        for (int document = iterator.nextDoc();
            document != DocIdSetIterator.NO_MORE_DOCS;
            document = iterator.nextDoc()) {
          assertTrue(ids.advanceExact(document));
          int id = Math.toIntExact(ids.longValue());
          assertArrayEquals(expectedVectors[id], vectors.vectorValue(iterator.index()), 0.0f);
          observed++;
        }
      }
      assertEquals(selectedRows, observed);
    }
  }

  private static void assertInvalidHeader(Path source, Path index, Codec codec) throws Exception {
    Map<String, Object> request = new HashMap<>();
    request.put(SOURCE_PATH_KEY, source.toString());
    request.put(INDEX_PATH_KEY, index.toString());
    request.put(CODEC_KEY, codec);
    request.put(EXPECTED_CODEC_NAME_KEY, codec.getName());
    request.put(EXPECTED_SOURCE_SIZE_KEY, Files.size(source));
    request.put(EXPECTED_FILE_VECTOR_COUNT_KEY, 1L);
    request.put(EXPECTED_DIMENSIONS_KEY, 1);
    request.put(EXPECTED_HEADER_BYTES_KEY, LEGACY_HEADER_BYTES);
    request.put(VECTOR_COUNT_KEY, 1L);
    request.put(NUM_INDEXING_THREADS_KEY, 1);
    request.put(FORCE_MERGE_SEGMENT_COUNT_KEY, 0);
    request.put(RAM_PER_THREAD_HARD_LIMIT_MB_KEY, HARD_LIMIT_MB);

    assertThrows(
        IllegalArgumentException.class, () -> new CuvsBenchFbinIndexingBridge().apply(request));
    assertDirectoryEmpty(index);
  }

  private static void assertDirectoryEmpty(Path directory) throws IOException {
    try (var children = Files.list(directory)) {
      assertFalse(children.findAny().isPresent());
    }
  }

  private static void assertNonNegativeTiming(Map<String, Object> response, String key) {
    assertTrue(key, (Long) response.get(key) >= 0L);
  }
}
