/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Stream;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Streams a validated FBIN prefix through ordinary Lucene {@link IndexWriter#addDocument} calls.
 *
 * <p>This class is a narrow interoperability bridge for generated bindings that cannot directly
 * wrap cuVS-specific classes. It intentionally implements only standard {@link Function} and
 * {@link Map} types. It is not a bulk, mapped, external-vector, or out-of-core index writer: every
 * vector is added to a normal Lucene writer, and the resulting index is self-contained.
 *
 * <p>The historical {@code num_indexing_threads} request key denotes a number of equal contiguous
 * partitions. Those partitions are built sequentially with one actual indexing thread, using
 * {@code CREATE} mode for the first and {@code APPEND} mode thereafter. Automatic and final
 * merges are disabled.
 *
 * <p>The target must be an existing, empty, non-symbolic-link directory. This bridge never removes,
 * replaces, or publishes that directory. Its caller remains responsible for validating and
 * atomically installing the staged index.
 */
public final class CuvsBenchFbinIndexingBridge
    implements Function<Map<String, Object>, Map<String, Object>> {
  public static final String SOURCE_PATH_KEY = "source_path";
  public static final String INDEX_PATH_KEY = "index_path";
  public static final String CODEC_KEY = "codec";
  public static final String EXPECTED_CODEC_NAME_KEY = "expected_codec_name";
  public static final String EXPECTED_SOURCE_SIZE_KEY = "expected_source_size";
  public static final String EXPECTED_FILE_VECTOR_COUNT_KEY = "expected_file_vector_count";
  public static final String EXPECTED_DIMENSIONS_KEY = "expected_dimensions";
  public static final String EXPECTED_HEADER_BYTES_KEY = "expected_header_bytes";
  public static final String VECTOR_COUNT_KEY = "vector_count";
  public static final String NUM_INDEXING_THREADS_KEY = "num_indexing_threads";
  public static final String FORCE_MERGE_SEGMENT_COUNT_KEY = "force_merge_segment_count";
  public static final String RAM_PER_THREAD_HARD_LIMIT_MB_KEY = "ram_per_thread_hard_limit_mb";

  public static final String CODEC_NAME_KEY = "codec_name";
  public static final String SOURCE_FILE_SIZE_KEY = "source_file_size";
  public static final String SOURCE_FILE_VECTOR_COUNT_KEY = "source_file_vector_count";
  public static final String DIMENSIONS_KEY = "dimensions";
  public static final String HEADER_BYTES_KEY = "header_bytes";
  public static final String INDEXED_PAYLOAD_BYTES_KEY = "indexed_payload_bytes";
  public static final String VECTOR_PAYLOAD_SHA256_KEY = "vector_payload_sha256";
  public static final String ACTUAL_INDEXING_THREAD_COUNT_KEY = "actual_indexing_thread_count";
  public static final String MAX_CONCURRENT_INDEXING_THREADS_KEY =
      "max_concurrent_indexing_threads";
  public static final String INDEXING_EXECUTION_MODE_KEY = "indexing_execution_mode";
  public static final String SEGMENT_COUNT_KEY = "segment_count";
  public static final String PREMERGE_SEGMENT_VECTOR_COUNT_PREFIX =
      "premerge_segment_vector_count_";
  public static final String MAX_BUFFERED_DOCS_KEY = "max_buffered_docs";
  public static final String APPLIED_RAM_PER_THREAD_HARD_LIMIT_MB_KEY =
      "applied_ram_per_thread_hard_limit_mb";
  public static final String INGEST_MERGE_POLICY_KEY = "ingest_merge_policy";
  public static final String DIRECTORY_OPEN_NS_KEY = "directory_open_ns";
  public static final String WRITER_SETUP_NS_KEY = "writer_setup_ns";
  public static final String DOCUMENT_INGEST_NS_KEY = "document_ingest_ns";
  public static final String FBIN_READ_NS_KEY = "fbin_read_ns";
  public static final String WRITER_COMMIT_CLOSE_NS_KEY = "writer_commit_close_ns";
  public static final String POST_BUILD_READER_NS_KEY = "post_build_reader_ns";
  public static final String DIRECTORY_CLOSE_NS_KEY = "directory_close_ns";
  public static final String RUNTIME_BUILD_WALL_NS_KEY = "runtime_build_wall_ns";

  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector";
  private static final int LEGACY_HEADER_BYTES = 8;
  private static final int EXTENDED_HEADER_BYTES = 16;
  private static final int FLOAT_BYTES = Float.BYTES;
  private static final int TARGET_BUFFER_BYTES = 8 * 1024 * 1024;
  private static final String PARTITIONED_SEQUENTIAL = "partitioned_sequential";

  private static final Set<String> REQUEST_KEYS =
      Set.of(
          SOURCE_PATH_KEY,
          INDEX_PATH_KEY,
          CODEC_KEY,
          EXPECTED_CODEC_NAME_KEY,
          EXPECTED_SOURCE_SIZE_KEY,
          EXPECTED_FILE_VECTOR_COUNT_KEY,
          EXPECTED_DIMENSIONS_KEY,
          EXPECTED_HEADER_BYTES_KEY,
          VECTOR_COUNT_KEY,
          NUM_INDEXING_THREADS_KEY,
          FORCE_MERGE_SEGMENT_COUNT_KEY,
          RAM_PER_THREAD_HARD_LIMIT_MB_KEY);

  @Override
  public Map<String, Object> apply(Map<String, Object> request) {
    long runtimeStarted = System.nanoTime();
    try {
      Request parsed = parseRequest(request);
      Map<String, Object> response = build(parsed);
      response.put(RUNTIME_BUILD_WALL_NS_KEY, System.nanoTime() - runtimeStarted);
      return response;
    } catch (IOException error) {
      throw new UncheckedIOException("FBIN Lucene indexing bridge failed", error);
    }
  }

  private static Map<String, Object> build(Request request) throws IOException {
    Path sourcePath = resolveSourcePath(request.sourcePath());
    SourceSnapshot sourceBefore = SourceSnapshot.capture(sourcePath);
    if (sourceBefore.size() != request.expectedSourceSize()) {
      throw new IllegalArgumentException(
          EXPECTED_SOURCE_SIZE_KEY
              + " does not match the source file: expected "
              + request.expectedSourceSize()
              + ", observed "
              + sourceBefore.size());
    }

    try (FileChannel source = FileChannel.open(sourcePath, StandardOpenOption.READ)) {
      FbinHeader header = readHeader(source, sourcePath);
      validateExpectedHeader(request, header);
      BuildPlan plan = BuildPlan.create(request, header);
      Path indexPath = resolveEmptyStagingDirectory(request.indexPath());
      if (sourcePath.equals(indexPath)) {
        throw new IllegalArgumentException("source_path and index_path must be different");
      }

      MessageDigest digest = sha256();
      BuildOutcome outcome = buildIndex(source, indexPath, request, header, plan, digest);

      if (source.size() != sourceBefore.size()) {
        throw new IOException("source file size changed while the index was being built");
      }
      SourceSnapshot sourceAfter = SourceSnapshot.capture(sourcePath);
      if (!sourceBefore.sameFileAndState(sourceAfter)) {
        throw new IOException("source file changed while the index was being built");
      }

      Map<String, Object> response = new HashMap<>();
      response.put(CODEC_NAME_KEY, request.codec().getName());
      response.put(SOURCE_FILE_SIZE_KEY, header.fileSize());
      response.put(SOURCE_FILE_VECTOR_COUNT_KEY, header.vectorCount());
      response.put(DIMENSIONS_KEY, header.dimensions());
      response.put(HEADER_BYTES_KEY, header.headerBytes());
      response.put(VECTOR_COUNT_KEY, plan.vectorCount());
      response.put(INDEXED_PAYLOAD_BYTES_KEY, plan.indexedPayloadBytes());
      response.put(VECTOR_PAYLOAD_SHA256_KEY, HexFormat.of().formatHex(digest.digest()));
      response.put(NUM_INDEXING_THREADS_KEY, request.numIndexingThreads());
      response.put(FORCE_MERGE_SEGMENT_COUNT_KEY, request.forceMergeSegmentCount());
      response.put(ACTUAL_INDEXING_THREAD_COUNT_KEY, 1);
      response.put(MAX_CONCURRENT_INDEXING_THREADS_KEY, 1);
      response.put(INDEXING_EXECUTION_MODE_KEY, PARTITIONED_SEQUENTIAL);
      response.put(SEGMENT_COUNT_KEY, outcome.segmentVectorCounts().length);
      for (int i = 0; i < outcome.segmentVectorCounts().length; i++) {
        response.put(PREMERGE_SEGMENT_VECTOR_COUNT_PREFIX + i, outcome.segmentVectorCounts()[i]);
      }
      response.put(MAX_BUFFERED_DOCS_KEY, plan.maxBufferedDocs());
      response.put(APPLIED_RAM_PER_THREAD_HARD_LIMIT_MB_KEY, request.ramPerThreadHardLimitMb());
      response.put(INGEST_MERGE_POLICY_KEY, NoMergePolicy.class.getSimpleName());
      response.put(DIRECTORY_OPEN_NS_KEY, outcome.timings().directoryOpenNs);
      response.put(WRITER_SETUP_NS_KEY, outcome.timings().writerSetupNs);
      response.put(DOCUMENT_INGEST_NS_KEY, outcome.timings().documentIngestNs);
      response.put(FBIN_READ_NS_KEY, outcome.timings().fbinReadNs);
      response.put(WRITER_COMMIT_CLOSE_NS_KEY, outcome.timings().writerCommitCloseNs);
      response.put(POST_BUILD_READER_NS_KEY, outcome.timings().postBuildReaderNs);
      response.put(DIRECTORY_CLOSE_NS_KEY, outcome.timings().directoryCloseNs);
      return response;
    }
  }

  private static BuildOutcome buildIndex(
      FileChannel source,
      Path indexPath,
      Request request,
      FbinHeader header,
      BuildPlan plan,
      MessageDigest digest)
      throws IOException {
    BuildTimings timings = new BuildTimings();
    long directoryOpenStarted = System.nanoTime();
    Directory directory = FSDirectory.open(indexPath);
    timings.directoryOpenNs = System.nanoTime() - directoryOpenStarted;

    try {
      float[] vector = new float[header.dimensions()];
      NumericDocValuesField idField = new NumericDocValuesField(ID_FIELD, 0L);
      KnnFloatVectorField vectorField =
          new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN);
      Document document = new Document();
      document.add(idField);
      document.add(vectorField);

      ByteBuffer buffer =
          ByteBuffer.allocateDirect(plan.bufferBytes()).order(ByteOrder.LITTLE_ENDIAN);
      long sourcePosition = header.headerBytes();
      long[] observedCounts = new long[0];

      for (int partition = 0; partition < request.numIndexingThreads(); partition++) {
        long setupStarted = System.nanoTime();
        IndexWriterConfig config = writerConfig(request, plan, partition == 0);
        IndexWriter writer = new IndexWriter(directory, config);
        timings.writerSetupNs += System.nanoTime() - setupStarted;

        try {
          long ingestStarted = System.nanoTime();
          long partitionStart = Math.multiplyExact((long) partition, plan.partitionVectorCount());
          long partitionStop = Math.addExact(partitionStart, plan.partitionVectorCount());
          long nextDocumentId = partitionStart;
          while (nextDocumentId < partitionStop) {
            int bufferedRows =
                Math.toIntExact(
                    Math.min((long) plan.rowsPerBuffer(), partitionStop - nextDocumentId));
            int bytesToRead = Math.multiplyExact(bufferedRows, plan.rowBytes());
            buffer.clear();
            buffer.limit(bytesToRead);
            long readStarted = System.nanoTime();
            readFully(source, buffer, sourcePosition);
            timings.fbinReadNs += System.nanoTime() - readStarted;
            sourcePosition = Math.addExact(sourcePosition, bytesToRead);
            buffer.flip();
            digest.update(buffer.asReadOnlyBuffer());

            FloatBuffer floats = buffer.asFloatBuffer();
            for (int row = 0; row < bufferedRows; row++) {
              floats.get(vector);
              validateFinite(vector, nextDocumentId);
              idField.setLongValue(nextDocumentId);
              vectorField.setVectorValue(vector);
              writer.addDocument(document);
              nextDocumentId++;
            }
          }
          timings.documentIngestNs += System.nanoTime() - ingestStarted;

          long commitStarted = System.nanoTime();
          writer.flush();
          writer.commit();
          writer.close();
          writer = null;
          timings.writerCommitCloseNs += System.nanoTime() - commitStarted;
        } catch (IOException | RuntimeException | Error failure) {
          rollback(writer, failure);
          throw failure;
        }

        long readerStarted = System.nanoTime();
        observedCounts =
            validateCommittedIndex(
                directory, partition + 1, plan.partitionVectorCount(), header.dimensions());
        timings.postBuildReaderNs += System.nanoTime() - readerStarted;
      }

      long expectedPosition = Math.addExact(header.headerBytes(), plan.indexedPayloadBytes());
      if (sourcePosition != expectedPosition) {
        throw new IllegalStateException(
            "FBIN stream position mismatch: expected "
                + expectedPosition
                + ", observed "
                + sourcePosition);
      }
      return new BuildOutcome(observedCounts, timings);
    } finally {
      long closeStarted = System.nanoTime();
      try {
        directory.close();
      } finally {
        timings.directoryCloseNs += System.nanoTime() - closeStarted;
      }
    }
  }

  private static IndexWriterConfig writerConfig(Request request, BuildPlan plan, boolean create) {
    IndexWriterConfig config = new IndexWriterConfig();
    config.setOpenMode(
        create ? IndexWriterConfig.OpenMode.CREATE : IndexWriterConfig.OpenMode.APPEND);
    config.setCodec(request.codec());
    config.setUseCompoundFile(false);
    config.setCommitOnClose(false);
    config.setMergePolicy(NoMergePolicy.INSTANCE);
    config.setMergeScheduler(new SerialMergeScheduler());
    config.setMaxBufferedDocs(plan.maxBufferedDocs());
    config.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    IndexWriterConfigPerThreadHardLimitBridge.setAndVerify(
        config, request.ramPerThreadHardLimitMb());
    return config;
  }

  private static long[] validateCommittedIndex(
      Directory directory, int expectedSegments, long expectedVectorsPerSegment, int dimensions)
      throws IOException {
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      if (reader.numDocs() != reader.maxDoc()) {
        throw new IllegalStateException("committed index unexpectedly contains deletions");
      }
      if (reader.leaves().size() != expectedSegments) {
        throw new IllegalStateException(
            "committed segment count mismatch: expected "
                + expectedSegments
                + ", observed "
                + reader.leaves().size());
      }

      long[] counts = new long[expectedSegments];
      long observedDocuments = 0L;
      for (int segment = 0; segment < expectedSegments; segment++) {
        LeafReaderContext context = reader.leaves().get(segment);
        LeafReader leaf = context.reader();
        if (leaf.numDocs() != leaf.maxDoc()) {
          throw new IllegalStateException("committed segment unexpectedly contains deletions");
        }
        if (leaf.numDocs() != expectedVectorsPerSegment) {
          throw new IllegalStateException(
              "committed segment vector count mismatch at segment "
                  + segment
                  + ": expected "
                  + expectedVectorsPerSegment
                  + ", observed "
                  + leaf.numDocs());
        }

        FloatVectorValues vectors = leaf.getFloatVectorValues(VECTOR_FIELD);
        if (vectors == null) {
          throw new IllegalStateException("committed segment is missing vector values");
        }
        if (vectors.dimension() != dimensions || vectors.size() != leaf.numDocs()) {
          throw new IllegalStateException("committed vector shape does not match the FBIN input");
        }

        NumericDocValues ids = leaf.getNumericDocValues(ID_FIELD);
        if (ids == null) {
          throw new IllegalStateException("committed segment is missing NumericDocValues IDs");
        }
        for (int localDocument = 0; localDocument < leaf.maxDoc(); localDocument++) {
          if (!ids.advanceExact(localDocument)) {
            throw new IllegalStateException(
                "committed document " + localDocument + " is missing its NumericDocValues ID");
          }
          long expectedId = Math.addExact((long) context.docBase, localDocument);
          if (ids.longValue() != expectedId) {
            throw new IllegalStateException(
                "committed document ID mismatch: expected "
                    + expectedId
                    + ", observed "
                    + ids.longValue());
          }
        }

        counts[segment] = leaf.numDocs();
        observedDocuments = Math.addExact(observedDocuments, leaf.numDocs());
      }
      long expectedDocuments = Math.multiplyExact(expectedSegments, expectedVectorsPerSegment);
      if (observedDocuments != expectedDocuments || observedDocuments != reader.numDocs()) {
        throw new IllegalStateException(
            "committed document count mismatch: expected "
                + expectedDocuments
                + ", observed "
                + observedDocuments);
      }
      return counts;
    }
  }

  private static void rollback(IndexWriter writer, Throwable failure) {
    if (writer == null) {
      return;
    }
    try {
      writer.rollback();
    } catch (Throwable rollbackFailure) {
      failure.addSuppressed(rollbackFailure);
    }
  }

  private static void validateFinite(float[] vector, long documentId) {
    for (int dimension = 0; dimension < vector.length; dimension++) {
      if (!Float.isFinite(vector[dimension])) {
        throw new IllegalArgumentException(
            "non-finite float at vector " + documentId + ", dimension " + dimension);
      }
    }
  }

  private static Path resolveSourcePath(String configuredPath) throws IOException {
    Path source = Path.of(configuredPath).toRealPath();
    if (!Files.isRegularFile(source, LinkOption.NOFOLLOW_LINKS)) {
      throw new IllegalArgumentException("source_path must identify a regular file");
    }
    Path fileName = source.getFileName();
    if (fileName == null || !fileName.toString().endsWith(".fbin")) {
      throw new IllegalArgumentException("source_path must have the .fbin extension");
    }
    return source;
  }

  private static Path resolveEmptyStagingDirectory(String configuredPath) throws IOException {
    Path requested = Path.of(configuredPath).toAbsolutePath().normalize();
    if (Files.isSymbolicLink(requested)
        || !Files.isDirectory(requested, LinkOption.NOFOLLOW_LINKS)) {
      throw new IllegalArgumentException(
          "index_path must be an existing, non-symbolic-link directory");
    }
    Path real = requested.toRealPath(LinkOption.NOFOLLOW_LINKS);
    if (!requested.equals(real)) {
      throw new IllegalArgumentException("index_path must be a real, normalized directory path");
    }
    try (Stream<Path> children = Files.list(real)) {
      if (children.findAny().isPresent()) {
        throw new IllegalArgumentException("index_path must be empty");
      }
    }
    return real;
  }

  private static FbinHeader readHeader(FileChannel source, Path sourcePath) throws IOException {
    long fileSize = source.size();
    int availableHeaderBytes = Math.toIntExact(Math.min(fileSize, EXTENDED_HEADER_BYTES));
    if (availableHeaderBytes < LEGACY_HEADER_BYTES) {
      throw new IllegalArgumentException(
          "FBIN file is too small to contain a header: " + sourcePath);
    }

    ByteBuffer bytes = ByteBuffer.allocate(availableHeaderBytes).order(ByteOrder.LITTLE_ENDIAN);
    readFully(source, bytes, 0L);
    bytes.flip();

    long legacyRows = Integer.toUnsignedLong(bytes.getInt(0));
    long legacyDimensions = Integer.toUnsignedLong(bytes.getInt(Integer.BYTES));
    if (matchesFileSize(fileSize, LEGACY_HEADER_BYTES, legacyRows, legacyDimensions)) {
      return checkedHeader(legacyRows, legacyDimensions, LEGACY_HEADER_BYTES, fileSize, sourcePath);
    }

    if (availableHeaderBytes == EXTENDED_HEADER_BYTES) {
      long extendedRows = bytes.getLong(0);
      long extendedDimensions = bytes.getLong(Long.BYTES);
      if (extendedRows >= 0L
          && extendedDimensions >= 0L
          && matchesFileSize(fileSize, EXTENDED_HEADER_BYTES, extendedRows, extendedDimensions)) {
        return checkedHeader(
            extendedRows, extendedDimensions, EXTENDED_HEADER_BYTES, fileSize, sourcePath);
      }
    }

    throw new IllegalArgumentException(
        "FBIN file size does not match either the classic or extended header: " + sourcePath);
  }

  private static FbinHeader checkedHeader(
      long rows, long dimensions, int headerBytes, long fileSize, Path sourcePath) {
    if (rows < 1L) {
      throw new IllegalArgumentException("FBIN vector count must be positive: " + sourcePath);
    }
    if (dimensions < 1L || dimensions > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "FBIN dimensions must be in [1, " + Integer.MAX_VALUE + "]: " + sourcePath);
    }
    return new FbinHeader(rows, Math.toIntExact(dimensions), headerBytes, fileSize);
  }

  private static boolean matchesFileSize(
      long fileSize, int headerBytes, long rows, long dimensions) {
    try {
      long values = Math.multiplyExact(rows, dimensions);
      long payloadBytes = Math.multiplyExact(values, FLOAT_BYTES);
      return fileSize == Math.addExact(headerBytes, payloadBytes);
    } catch (ArithmeticException overflow) {
      return false;
    }
  }

  private static void validateExpectedHeader(Request request, FbinHeader header) {
    if (header.fileSize() != request.expectedSourceSize()) {
      throw mismatch(EXPECTED_SOURCE_SIZE_KEY, request.expectedSourceSize(), header.fileSize());
    }
    if (header.vectorCount() != request.expectedFileVectorCount()) {
      throw mismatch(
          EXPECTED_FILE_VECTOR_COUNT_KEY, request.expectedFileVectorCount(), header.vectorCount());
    }
    if (header.dimensions() != request.expectedDimensions()) {
      throw mismatch(EXPECTED_DIMENSIONS_KEY, request.expectedDimensions(), header.dimensions());
    }
    if (header.headerBytes() != request.expectedHeaderBytes()) {
      throw mismatch(
          EXPECTED_HEADER_BYTES_KEY, request.expectedHeaderBytes(), header.headerBytes());
    }
  }

  private static IllegalArgumentException mismatch(String name, long expected, long observed) {
    return new IllegalArgumentException(
        name + " does not match the FBIN file: expected " + expected + ", observed " + observed);
  }

  private static void readFully(FileChannel source, ByteBuffer target, long position)
      throws IOException {
    long next = position;
    while (target.hasRemaining()) {
      int read = source.read(target, next);
      if (read < 0) {
        throw new IOException("unexpected EOF while reading FBIN at byte " + next);
      }
      if (read == 0) {
        Thread.onSpinWait();
        continue;
      }
      next = Math.addExact(next, read);
    }
  }

  private static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is unavailable", impossible);
    }
  }

  private static Request parseRequest(Map<String, Object> request) {
    if (request == null) {
      throw new IllegalArgumentException("request must not be null");
    }
    for (Object key : request.keySet()) {
      if (!(key instanceof String stringKey) || !REQUEST_KEYS.contains(stringKey)) {
        throw new IllegalArgumentException("unsupported request key: " + key);
      }
    }

    String sourcePath = requiredValue(request, SOURCE_PATH_KEY, String.class);
    String indexPath = requiredValue(request, INDEX_PATH_KEY, String.class);
    Codec codec = requiredValue(request, CODEC_KEY, Codec.class);
    String expectedCodecName = requiredValue(request, EXPECTED_CODEC_NAME_KEY, String.class);
    Long expectedSourceSize = requiredValue(request, EXPECTED_SOURCE_SIZE_KEY, Long.class);
    Long expectedFileVectorCount =
        requiredValue(request, EXPECTED_FILE_VECTOR_COUNT_KEY, Long.class);
    Integer expectedDimensions = requiredValue(request, EXPECTED_DIMENSIONS_KEY, Integer.class);
    Integer expectedHeaderBytes = requiredValue(request, EXPECTED_HEADER_BYTES_KEY, Integer.class);
    Long vectorCount = requiredValue(request, VECTOR_COUNT_KEY, Long.class);
    Integer numIndexingThreads = requiredValue(request, NUM_INDEXING_THREADS_KEY, Integer.class);
    Integer forceMergeSegmentCount =
        requiredValue(request, FORCE_MERGE_SEGMENT_COUNT_KEY, Integer.class);
    Integer hardLimit = requiredValue(request, RAM_PER_THREAD_HARD_LIMIT_MB_KEY, Integer.class);

    if (sourcePath.isEmpty()) {
      throw new IllegalArgumentException(SOURCE_PATH_KEY + " must not be empty");
    }
    if (indexPath.isEmpty()) {
      throw new IllegalArgumentException(INDEX_PATH_KEY + " must not be empty");
    }
    if (!codec.getName().equals(expectedCodecName)) {
      throw new IllegalArgumentException(
          EXPECTED_CODEC_NAME_KEY
              + " does not match the supplied codec: expected "
              + expectedCodecName
              + ", observed "
              + codec.getName());
    }
    if (expectedSourceSize < 0L) {
      throw new IllegalArgumentException(EXPECTED_SOURCE_SIZE_KEY + " must not be negative");
    }
    if (expectedFileVectorCount < 1L) {
      throw new IllegalArgumentException(EXPECTED_FILE_VECTOR_COUNT_KEY + " must be positive");
    }
    if (expectedDimensions < 1) {
      throw new IllegalArgumentException(EXPECTED_DIMENSIONS_KEY + " must be positive");
    }
    if (expectedHeaderBytes != LEGACY_HEADER_BYTES
        && expectedHeaderBytes != EXTENDED_HEADER_BYTES) {
      throw new IllegalArgumentException(EXPECTED_HEADER_BYTES_KEY + " must be 8 or 16");
    }
    if (vectorCount < 1L) {
      throw new IllegalArgumentException(VECTOR_COUNT_KEY + " must be positive");
    }
    if (numIndexingThreads < 1) {
      throw new IllegalArgumentException(NUM_INDEXING_THREADS_KEY + " must be positive");
    }
    if (forceMergeSegmentCount != 0) {
      throw new IllegalArgumentException(FORCE_MERGE_SEGMENT_COUNT_KEY + " must be 0");
    }
    if (hardLimit < 1) {
      throw new IllegalArgumentException(RAM_PER_THREAD_HARD_LIMIT_MB_KEY + " must be positive");
    }

    return new Request(
        sourcePath,
        indexPath,
        codec,
        expectedCodecName,
        expectedSourceSize,
        expectedFileVectorCount,
        expectedDimensions,
        expectedHeaderBytes,
        vectorCount,
        numIndexingThreads,
        forceMergeSegmentCount,
        hardLimit);
  }

  private static <T> T requiredValue(
      Map<String, Object> request, String key, Class<T> expectedType) {
    Object value = request.get(key);
    if (!expectedType.isInstance(value)) {
      throw new IllegalArgumentException(key + " must have type " + expectedType.getSimpleName());
    }
    return expectedType.cast(value);
  }

  private record Request(
      String sourcePath,
      String indexPath,
      Codec codec,
      String expectedCodecName,
      long expectedSourceSize,
      long expectedFileVectorCount,
      int expectedDimensions,
      int expectedHeaderBytes,
      long vectorCount,
      int numIndexingThreads,
      int forceMergeSegmentCount,
      int ramPerThreadHardLimitMb) {}

  private record FbinHeader(long vectorCount, int dimensions, int headerBytes, long fileSize) {}

  private record BuildPlan(
      long vectorCount,
      long partitionVectorCount,
      int rowBytes,
      long indexedPayloadBytes,
      int maxBufferedDocs,
      int rowsPerBuffer,
      int bufferBytes) {
    static BuildPlan create(Request request, FbinHeader header) {
      if (request.vectorCount() > header.vectorCount()) {
        throw new IllegalArgumentException(
            VECTOR_COUNT_KEY
                + " cannot exceed the FBIN vector count: "
                + request.vectorCount()
                + " > "
                + header.vectorCount());
      }
      if (request.vectorCount() > IndexWriter.MAX_DOCS) {
        throw new IllegalArgumentException(
            VECTOR_COUNT_KEY + " cannot exceed IndexWriter.MAX_DOCS=" + IndexWriter.MAX_DOCS);
      }
      if (request.numIndexingThreads() > request.vectorCount()) {
        throw new IllegalArgumentException(
            NUM_INDEXING_THREADS_KEY + " cannot exceed vector_count");
      }
      if (request.vectorCount() % request.numIndexingThreads() != 0L) {
        throw new IllegalArgumentException(
            "vector_count must be divisible by num_indexing_threads for equal partitions");
      }

      try {
        long partitionVectorCount = request.vectorCount() / request.numIndexingThreads();
        int rowBytes = Math.multiplyExact(header.dimensions(), FLOAT_BYTES);
        long indexedPayloadBytes = Math.multiplyExact(request.vectorCount(), rowBytes);
        int maxBufferedDocs = Math.toIntExact(Math.addExact(partitionVectorCount, 1L));
        int rowsPerBuffer = Math.max(1, TARGET_BUFFER_BYTES / rowBytes);
        int bufferBytes = Math.multiplyExact(rowsPerBuffer, rowBytes);
        return new BuildPlan(
            request.vectorCount(),
            partitionVectorCount,
            rowBytes,
            indexedPayloadBytes,
            maxBufferedDocs,
            rowsPerBuffer,
            bufferBytes);
      } catch (ArithmeticException overflow) {
        throw new IllegalArgumentException(
            "FBIN build dimensions overflow supported ranges", overflow);
      }
    }
  }

  private record SourceSnapshot(Object fileKey, long size, long modifiedMillis) {
    static SourceSnapshot capture(Path source) throws IOException {
      BasicFileAttributes attributes =
          Files.readAttributes(source, BasicFileAttributes.class, LinkOption.NOFOLLOW_LINKS);
      if (!attributes.isRegularFile()) {
        throw new IllegalArgumentException("source_path must remain a regular file");
      }
      return new SourceSnapshot(
          attributes.fileKey(), attributes.size(), attributes.lastModifiedTime().toMillis());
    }

    boolean sameFileAndState(SourceSnapshot other) {
      boolean sameKey = fileKey == null ? other.fileKey == null : fileKey.equals(other.fileKey);
      return sameKey && size == other.size && modifiedMillis == other.modifiedMillis;
    }
  }

  private static final class BuildTimings {
    private long directoryOpenNs;
    private long writerSetupNs;
    private long documentIngestNs;
    private long fbinReadNs;
    private long writerCommitCloseNs;
    private long postBuildReaderNs;
    private long directoryCloseNs;
  }

  private record BuildOutcome(long[] segmentVectorCounts, BuildTimings timings) {}
}
