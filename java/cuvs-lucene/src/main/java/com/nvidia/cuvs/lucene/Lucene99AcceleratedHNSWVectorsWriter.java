/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.createMultiLayerHnswGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.createSingleVectorHnswGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.printInfoStream;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.writeEmpty;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.writeGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.writeMeta;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_INDEX_EXT;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_META_CODEC_NAME;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.QuantizationType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;

/**
 * This class extends upon the KnnVectorsWriter to
 * enable the creation of GPU-based accelerated HNSW based vector search.
 *
 * @since 25.10
 */
public class Lucene99AcceleratedHNSWVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(Lucene99AcceleratedHNSWVectorsWriter.class);
  private static final String COMPONENT = "Lucene99AcceleratedHNSWVectorsWriter";
  private static final String POST_INGEST_OVERLAP_PROPERTY =
      "cuvs.lucene.experimentalPostIngestOverlap";
  private static final LuceneProvider LUCENE_PROVIDER;
  private static final Integer VERSION_CURRENT;

  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private final FlatVectorsWriter flatVectorsWriter;
  private final List<FieldWriter> fields = new ArrayList<>();
  private final InfoStream infoStream;
  private final boolean postIngestOverlapEnabled;
  private final String segmentName;
  private IndexOutput hnswMeta = null;
  private IndexOutput hnswVectorIndex = null;
  private String vemFileName;
  private String vexFileName;
  private boolean finished;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      VERSION_CURRENT = LUCENE_PROVIDER.getStaticIntParam("VERSION_CURRENT");
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsWriter}
   *
   * @param state instance of the {@link org.apache.lucene.index.SegmentWriteState}
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   * @param flatVectorsWriter instance of the {@link org.apache.lucene.codecs.hnsw.FlatVectorsWriter}
   * @throws IOException IOException
   */
  public Lucene99AcceleratedHNSWVectorsWriter(
      SegmentWriteState state,
      AcceleratedHNSWParams acceleratedHNSWParams,
      FlatVectorsWriter flatVectorsWriter)
      throws IOException {
    super();
    this.flatVectorsWriter = flatVectorsWriter;
    this.infoStream = state.infoStream;
    this.postIngestOverlapEnabled = Boolean.getBoolean(POST_INGEST_OVERLAP_PROPERTY);
    this.segmentName = state.segmentInfo.name;
    this.acceleratedHNSWParams = acceleratedHNSWParams;
    vemFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, HNSW_META_CODEC_EXT);
    vexFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, HNSW_INDEX_EXT);
    boolean success = false;
    try {
      hnswMeta = state.directory.createOutput(vemFileName, state.context);
      hnswVectorIndex = state.directory.createOutput(vexFileName, state.context);
      CodecUtil.writeIndexHeader(
          hnswMeta,
          HNSW_META_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          hnswVectorIndex,
          HNSW_INDEX_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
      printInfoStream(infoStream, COMPONENT, "Lucene99AcceleratedHNSWVectorsWriter is initialized");
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /**
   * Add new field for indexing.
   */
  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    var encoding = fieldInfo.getVectorEncoding();
    if (encoding != FLOAT32) {
      throw new IllegalArgumentException("Expected float32, got:" + encoding);
    }
    var writer = Objects.requireNonNull(flatVectorsWriter.addField(fieldInfo));
    var cuvsFieldWriter = new FieldWriter(QuantizationType.NONE, fieldInfo, writer);
    fields.add(cuvsFieldWriter);
    return writer;
  }

  /**
   * Flush/sorting path: builds a host matrix from the heap vectors, then delegates to {@link
   * #writeNonTrivialField(FieldInfo, CuVSMatrix)}.
   *
   * @param fieldInfo instance of FieldInfo that has the field description
   * @param vectors vectors to index
   * @throws IOException
   */
  private void writeFieldInternal(FieldInfo fieldInfo, List<float[]> vectors) throws IOException {
    if (writeTrivialField(fieldInfo, vectors.size())) {
      return;
    }
    long startedAt = System.nanoTime();
    CuVSMatrix dataset = Utils.createHostFloatMatrix(vectors, fieldInfo.getVectorDimension());
    reportPhase(
        "host_matrix_materialization",
        startedAt,
        (long) vectors.size() * fieldInfo.getVectorDimension() * Float.BYTES);
    writeNonTrivialField(fieldInfo, dataset);
  }

  /**
   * Builds the intermediate CAGRA index and writes the HNSW index. This non-trivial path is shared
   * by flushes and merges after zero- and one-vector cases have been handled.
   *
   * @param fieldInfo instance of FieldInfo that has the field description
   * @param dataset matrix of all vectors to index
   * @throws IOException
   */
  private void writeNonTrivialField(FieldInfo fieldInfo, CuVSMatrix dataset) throws IOException {
    try {
      int size = (int) dataset.size();
      CagraIndexParams params =
          CagraIndexParamsFactory.create(acceleratedHNSWParams, dataset.size(), dataset.columns());
      long startedAt = System.nanoTime();
      CagraIndex cagraIndex =
          CagraIndex.newBuilder(getCuVSResourcesInstance())
              .withDataset(dataset)
              .withIndexParams(params)
              .build();
      reportPhase("cagra_build", startedAt);
      startedAt = System.nanoTime();
      CuVSMatrix adjacencyListMatrix = cagraIndex.getGraph();
      reportPhase("cagra_graph_access", startedAt);
      int dimensions = fieldInfo.getVectorDimension();
      startedAt = System.nanoTime();
      GPUBuiltHnswGraph hnswGraph =
          createMultiLayerHnswGraph(
              fieldInfo,
              dimensions,
              adjacencyListMatrix,
              dataset,
              acceleratedHNSWParams.getHnswLayers(),
              params,
              QuantizationType.NONE,
              acceleratedHNSWParams.getWriterThreads());
      reportPhase("hnsw_graph_materialization", startedAt);
      long vectorIndexOffset = hnswVectorIndex.getFilePointer();
      startedAt = System.nanoTime();
      int[][] graphLevelNodeOffsets =
          writeGraph(hnswGraph, hnswVectorIndex, acceleratedHNSWParams.getWriterThreads());
      long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;
      reportPhase("hnsw_graph_serialization", startedAt, vectorIndexLength);
      long metadataStart = System.nanoTime();
      long metadataBytesBefore = hnswVectorIndex.getFilePointer() + hnswMeta.getFilePointer();
      writeMeta(
          hnswVectorIndex,
          hnswMeta,
          fieldInfo,
          vectorIndexOffset,
          vectorIndexLength,
          size,
          hnswGraph,
          graphLevelNodeOffsets);
      reportPhase(
          "hnsw_metadata_serialization",
          metadataStart,
          hnswVectorIndex.getFilePointer() + hnswMeta.getFilePointer() - metadataBytesBefore);
      startedAt = System.nanoTime();
      cagraIndex.close();
      reportPhase("cagra_close", startedAt);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /** Writes the empty or one-vector representation, if {@code size} is trivial. */
  private boolean writeTrivialField(FieldInfo fieldInfo, int size) throws IOException {
    if (size == 0) {
      writeEmpty(fieldInfo, hnswMeta);
      return true;
    }
    if (size == 1) {
      writeSingleVectorGraph(fieldInfo);
      return true;
    }
    return false;
  }

  /**
   * Build the indexes and writes it to the disk.
   */
  @Override
  public void flush(int maxDoc, DocMap sortMap) throws IOException {
    long flushStartedAt = System.nanoTime();
    try {
      if (postIngestOverlapEnabled) {
        long preparationStartedAt = System.nanoTime();
        List<PendingGraphField> graphFields = prepareGraphFields(fields, sortMap);
        reportPhase("graph_input_preparation", preparationStartedAt);
        PostIngestFlushCoordinator.runOverlapped(
            () -> timedFlatFlush(maxDoc, sortMap), () -> timedGraphWrite(graphFields));
      } else {
        timedFlatFlush(maxDoc, sortMap);
        long graphStartedAt = System.nanoTime();
        try {
          for (var field : fields) {
            if (sortMap == null) {
              writeField(field);
            } else {
              writeSortingField(field, sortMap);
            }
          }
        } finally {
          reportPhase("graph_branch", graphStartedAt);
        }
      }
    } finally {
      reportPhase("flush_total", flushStartedAt);
    }
  }

  private void timedFlatFlush(int maxDoc, DocMap sortMap) throws IOException {
    long startedAt = System.nanoTime();
    try {
      flatVectorsWriter.flush(maxDoc, sortMap);
    } finally {
      reportPhase("flat_vector_flush", startedAt, floatPayloadBytes(fields));
    }
  }

  private void timedGraphWrite(List<PendingGraphField> graphFields) throws IOException {
    long startedAt = System.nanoTime();
    try {
      for (PendingGraphField graphField : graphFields) {
        writeFieldInternal(graphField.fieldInfo(), graphField.vectors());
      }
    } finally {
      reportPhase("graph_branch", startedAt);
    }
  }

  static List<PendingGraphField> prepareGraphFields(List<FieldWriter> fields, Sorter.DocMap sortMap)
      throws IOException {
    List<PendingGraphField> graphFields = new ArrayList<>(fields.size());
    for (FieldWriter field : fields) {
      List<float[]> vectors = field.getFloatVectors();
      if (sortMap != null) {
        DocsWithFieldSet docsWithField = field.getDocsWithFieldSet();
        int[] newToOldOrdinal = new int[docsWithField.cardinality()];
        mapOldOrdToNewOrd(docsWithField, sortMap, null, newToOldOrdinal, null);
        List<float[]> sortedVectors = new ArrayList<>(vectors.size());
        for (int oldOrdinal : newToOldOrdinal) {
          sortedVectors.add(vectors.get(oldOrdinal));
        }
        vectors = sortedVectors;
      }
      graphFields.add(new PendingGraphField(field.fieldInfo(), vectors));
    }
    return List.copyOf(graphFields);
  }

  private static long floatPayloadBytes(List<FieldWriter> fields) {
    long bytes = 0L;
    for (FieldWriter field : fields) {
      bytes +=
          (long) field.getFloatVectors().size()
              * field.fieldInfo().getVectorDimension()
              * Float.BYTES;
    }
    return bytes;
  }

  record PendingGraphField(FieldInfo fieldInfo, List<float[]> vectors) {}

  private void reportPhase(String phase, long startedAt) {
    reportPhase(phase, startedAt, -1L);
  }

  private void reportPhase(String phase, long startedAt, long bytes) {
    reportPhase(infoStream, segmentName, phase, startedAt, bytes);
  }

  static void reportPhase(
      InfoStream infoStream, String segmentName, String phase, long startedAt, long bytes) {
    String byteMetric = bytes < 0L ? "" : " bytes=" + bytes;
    try {
      printInfoStream(
          infoStream,
          COMPONENT,
          "benchmark_phase="
              + phase
              + " segment="
              + segmentName
              + " duration_ns="
              + (System.nanoTime() - startedAt)
              + byteMetric);
    } catch (RuntimeException ignored) {
      // Optional benchmark telemetry must not change indexing success or failure semantics.
    }
  }

  /**
   * Builds the index and writes it to the disk.
   *
   * @param fieldData
   * @throws IOException
   */
  private void writeField(FieldWriter fieldData) throws IOException {
    writeFieldInternal(fieldData.fieldInfo(), fieldData.getFloatVectors());
  }

  /**
   * Builds the index and writes it to the disk.
   *
   * @param fieldData instance of GPUFieldWriter
   * @param sortMap instance of the DocMap
   * @throws IOException
   */
  private void writeSortingField(FieldWriter fieldData, Sorter.DocMap sortMap) throws IOException {
    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()];
    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);
    List<float[]> sortedVectors = new ArrayList<float[]>();
    List<float[]> floatVectors = fieldData.getFloatVectors();
    for (int i = 0; i < floatVectors.size(); i++) {
      sortedVectors.add(floatVectors.get(new2OldOrd[i]));
    }
    writeFieldInternal(fieldData.fieldInfo(), sortedVectors);
  }

  /**
   * Builds and writes a single vector graph.
   *
   * @param fieldInfo instance of FieldInfo
   * @throws IOException I/O Exceptions
   */
  private void writeSingleVectorGraph(FieldInfo fieldInfo) throws IOException {
    try {
      int size = 1;
      int dimensions = fieldInfo.getVectorDimension();
      GPUBuiltHnswGraph hnswGraph = createSingleVectorHnswGraph(size, dimensions);
      long vectorIndexOffset = hnswVectorIndex.getFilePointer();
      int[][] graphLevelNodeOffsets = writeGraph(hnswGraph, hnswVectorIndex);
      long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;
      writeMeta(
          hnswVectorIndex,
          hnswMeta,
          fieldInfo,
          vectorIndexOffset,
          vectorIndexLength,
          size,
          hnswGraph,
          graphLevelNodeOffsets);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Streams merged vectors directly into a native host-memory matrix without materializing a
   * {@code List<float[]>} on the Java heap. This avoids retaining both the heap list and native
   * matrix during a force merge.
   */
  private void vectorBasedMerge(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {
      long startedAt = System.nanoTime();
      int size = countMergedVectors(fieldInfo, mergeState);
      reportPhase("merged_vector_count", startedAt);
      if (writeTrivialField(fieldInfo, size)) {
        return;
      }
      FloatVectorValues mergedVectors =
          KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
      int dims = fieldInfo.getVectorDimension();
      startedAt = System.nanoTime();
      CuVSHostMatrix dataset =
          buildMergedDataset(
              mergedVectors, size, CuVSMatrix.hostBuilder(size, dims, CuVSMatrix.DataType.FLOAT));
      reportPhase("merged_matrix_materialization", startedAt, (long) size * dims * Float.BYTES);
      writeNonTrivialField(fieldInfo, dataset);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /* Replays merged vectors into a builder that remains responsible for storage until build. */
  static CuVSHostMatrix buildMergedDataset(
      FloatVectorValues mergedVectors, int expectedSize, CuVSMatrix.Builder<CuVSHostMatrix> builder)
      throws IOException {
    try (builder) {
      KnnVectorValues.DocIndexIterator it = mergedVectors.iterator();
      int replayed = 0;
      for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
        if (replayed == expectedSize) {
          throw mergeReplayMismatch(expectedSize, (long) replayed + 1L, true);
        }
        builder.addVector(mergedVectors.vectorValue(it.index()));
        replayed = Math.incrementExact(replayed);
      }
      if (replayed != expectedSize) {
        throw mergeReplayMismatch(expectedSize, replayed, false);
      }
      return builder.build();
    }
  }

  private static IOException mergeReplayMismatch(int expected, long observed, boolean lowerBound) {
    return new IOException(
        "Merged vector count changed between passes: expected "
            + expected
            + (lowerBound ? ", observed at least " : ", observed ")
            + observed);
  }

  /** Counts the live vectors that the merge iterator will actually yield. */
  private static int countMergedVectors(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException {
    FloatVectorValues mergedVectors =
        KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
    int count = 0;
    KnnVectorValues.DocIndexIterator it = mergedVectors.iterator();
    for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
      try {
        count = Math.incrementExact(count);
      } catch (ArithmeticException tooManyVectors) {
        throw new IOException(
            "Merged vector count exceeds the supported integer range", tooManyVectors);
      }
    }
    return count;
  }

  /**
   * Write field for merging.
   */
  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    long startedAt = System.nanoTime();
    try {
      flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
    } finally {
      reportPhase("flat_vector_merge", startedAt);
    }
    startedAt = System.nanoTime();
    try {
      vectorBasedMerge(fieldInfo, mergeState);
    } finally {
      reportPhase("graph_merge", startedAt);
    }
  }

  /**
   * Called once at the end before close.
   */
  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    long startedAt = System.nanoTime();
    try {
      flatVectorsWriter.finish();
    } finally {
      reportPhase("flat_vector_finish", startedAt);
    }
    startedAt = System.nanoTime();
    try {
      if (hnswMeta != null) {
        // write end of fields marker
        hnswMeta.writeInt(-1);
        CodecUtil.writeFooter(hnswMeta);
      }
      if (hnswVectorIndex != null) {
        CodecUtil.writeFooter(hnswVectorIndex);
      }
    } finally {
      reportPhase("hnsw_finish", startedAt);
    }
  }

  /**
   * Closes the resources.
   */
  @Override
  public void close() throws IOException {
    printInfoStream(infoStream, COMPONENT, "Closing resources");
    long startedAt = System.nanoTime();
    try {
      IOUtils.close(hnswMeta, hnswVectorIndex, flatVectorsWriter);
      closeCuVSResourcesInstance();
    } finally {
      reportPhase("writer_close", startedAt);
    }
  }

  /**
   * Returns the memory usage of this object in bytes.
   */
  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (var field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }
}
