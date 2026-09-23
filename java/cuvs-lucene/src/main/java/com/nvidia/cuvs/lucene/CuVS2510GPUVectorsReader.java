/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_INDEX_EXT;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_META_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.VERSION_CURRENT;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.VERSION_GRAPH_ONLY_PERSISTENCE;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.VERSION_START;
import static com.nvidia.cuvs.lucene.GPUSearchParams.CagraPersistenceMode.GRAPH_AND_DATASET;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceQuery;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IOContext.Context;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.IntToIntFunction;

/**
 * KnnVectorsReader instance associated with cuVS format for reading vectors from an index.
 *
 * @since 25.10
 */
public class CuVS2510GPUVectorsReader extends KnnVectorsReader {

  private static final LuceneProvider LUCENE_PROVIDER;
  private static final List<VectorSimilarityFunction> VECTOR_SIMILARITY_FUNCTIONS;

  private final FlatVectorsReader flatVectorsReader;
  private final FieldInfos fieldInfos;
  private final IntObjectHashMap<FieldEntry> fields;
  private final IntObjectHashMap<GPUIndex> cuvsIndices;
  private final IndexInput cuvsIndexInput;
  private final CuVSResources indexResources;
  private final FilterBitsetCache filterBitsetCache;
  private final boolean mergeContext;
  private final boolean flatVectorsAlreadySequential;
  private boolean closed;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      VECTOR_SIMILARITY_FUNCTIONS = LUCENE_PROVIDER.getSimilarityFunctions();
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes the {@link CuVS2510GPUVectorsReader}, checks and loads the index.
   *
   * @param state instance of the SegmentReadState
   * @param flatReader instance of the FlatVectorsReader
   *
   * @throws IOException I/O exception
   */
  public CuVS2510GPUVectorsReader(SegmentReadState state, FlatVectorsReader flatReader)
      throws IOException {
    this(
        state,
        flatReader,
        new FilterBitsetCache(FilterBitsetCacheConfig.DEFAULT),
        ThreadLocalCuVSResourcesProvider::createRequiredIndependentCuVSResourcesInstance);
  }

  /** Initializes the reader with the cache owned by its vectors format. */
  CuVS2510GPUVectorsReader(
      SegmentReadState state, FlatVectorsReader flatReader, FilterBitsetCache filterBitsetCache)
      throws IOException {
    this(
        state,
        flatReader,
        filterBitsetCache,
        ThreadLocalCuVSResourcesProvider::createRequiredIndependentCuVSResourcesInstance);
  }

  /**
   * Initializes the reader with the cache owned by its vectors format.
   *
   * @param state instance of the SegmentReadState
   * @param flatReader instance of the FlatVectorsReader
   * @param filterBitsetCache filter cache shared by readers from the same vectors format
   * @throws IOException I/O exception
   */
  CuVS2510GPUVectorsReader(
      SegmentReadState state,
      FlatVectorsReader flatReader,
      FilterBitsetCache filterBitsetCache,
      CuVSReaderResourcesFactory readerResourcesFactory)
      throws IOException {
    this.flatVectorsReader = flatReader;
    this.filterBitsetCache = filterBitsetCache;
    this.fieldInfos = state.fieldInfos;
    this.mergeContext = state.context.context().equals(Context.MERGE);
    this.flatVectorsAlreadySequential =
        state.context.context().equals(Context.MERGE)
            || state.context.context().equals(Context.FLUSH);
    this.fields = new IntObjectHashMap<>();
    boolean success = false;
    int versionMeta = -1;
    try {
      this.indexResources = mergeContext ? null : createIndexResources(readerResourcesFactory);
      String metaFileName =
          IndexFileNames.segmentFileName(
              state.segmentInfo.name, state.segmentSuffix, CUVS_META_CODEC_EXT);
      try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
        Throwable priorException = null;
        try {
          versionMeta =
              CodecUtil.checkIndexHeader(
                  meta,
                  CUVS_META_CODEC_NAME,
                  VERSION_START,
                  VERSION_CURRENT,
                  state.segmentInfo.getId(),
                  state.segmentSuffix);
          readFields(meta, versionMeta);
        } catch (Throwable exception) {
          priorException = exception;
        } finally {
          CodecUtil.checkFooter(meta, priorException);
        }
        var ioContext = state.context.withReadAdvice(ReadAdvice.SEQUENTIAL);
        cuvsIndexInput = openCuVSInput(state, versionMeta, ioContext);
        /*
         * Only load indexes on the GPU when this reader is opening for searches.
         * Do not load indexes on the GPU when this reader is opening during merge calls.
         * With this approach we reduce device memory usage by approximately 50% during merges.
         */
        if (mergeContext) {
          cuvsIndices = null;
        } else {
          cuvsIndices = loadCuVSIndices();
        }
        success = true;
      }
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  private static CuVSResources createIndexResources(
      CuVSReaderResourcesFactory readerResourcesFactory) throws IOException {
    if (readerResourcesFactory == null) {
      throw new NullPointerException("readerResourcesFactory");
    }
    try {
      CuVSResources resources = readerResourcesFactory.create();
      if (resources == null) {
        throw new IllegalStateException("readerResourcesFactory returned null");
      }
      return resources;
    } catch (Throwable t) {
      throw Utils.handleThrowable(t);
    }
  }

  /**
   * Opens and returns the IndexInput for the segment file.
   *
   * @param state instance of the SegmentReadState
   * @param versionMeta the version number
   * @param context instance of the IOContext
   * @return an instance of the IndexInput
   * @throws IOException
   */
  private static IndexInput openCuVSInput(
      SegmentReadState state, int versionMeta, IOContext context) throws IOException {
    String fileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, CUVS_INDEX_EXT);
    IndexInput in = state.directory.openInput(fileName, context);
    boolean success = false;
    try {
      int versionVectorData =
          CodecUtil.checkIndexHeader(
              in,
              CUVS_INDEX_CODEC_NAME,
              VERSION_START,
              VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      checkVersion(versionMeta, versionVectorData, in);
      CodecUtil.retrieveChecksum(in);
      success = true;
      return in;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(in);
      }
    }
  }

  /**
   * Confirms that the vector dimensions are as expected.
   *
   * @param info instance of the FieldInfo that describes document fields
   * @param fieldEntry instance of the FieldEntry that holds the meta information for the field
   */
  private void validateFieldEntry(FieldInfo info, FieldEntry fieldEntry) {
    int dimension = info.getVectorDimension();
    if (dimension != fieldEntry.dims()) {
      throw new IllegalStateException(
          "Inconsistent vector dimension for field=\""
              + info.name
              + "\"; "
              + dimension
              + " != "
              + fieldEntry.dims());
    }
  }

  /**
   * Reads the fieldInfo for each index field and loads FieldEntry in a map.
   *
   * @param meta instance of the ChecksumIndexInput
   * @throws IOException
   */
  private void readFields(ChecksumIndexInput meta, int version) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      FieldEntry fieldEntry = readField(meta, info, version);
      validateFieldEntry(info, fieldEntry);
      fields.put(info.number, fieldEntry);
    }
  }

  /**
   * Checks the distance function validity and returns it.
   *
   * @param input instance of DataInput
   * @return an instance of VectorSimilarityFunction
   * @throws IOException
   */
  static VectorSimilarityFunction readSimilarityFunction(DataInput input) throws IOException {
    int i = input.readInt();
    if (i < 0 || i >= VECTOR_SIMILARITY_FUNCTIONS.size()) {
      throw new IllegalArgumentException("invalid distance function: " + i);
    }
    return VECTOR_SIMILARITY_FUNCTIONS.get(i);
  }

  /**
   * Reads the vector encoding (The numeric data type of the vector values) from the DataInput.
   *
   * @param input instance of DataInput
   * @return the vector encoding
   * @throws IOException
   */
  static VectorEncoding readVectorEncoding(DataInput input) throws IOException {
    int encodingId = input.readInt();
    if (encodingId < 0 || encodingId >= VectorEncoding.values().length) {
      throw new CorruptIndexException("Invalid vector encoding id: " + encodingId, input);
    }
    return VectorEncoding.values()[encodingId];
  }

  /**
   * Reads the field from IndexInput using FieldInfo.
   *
   * @param input instance of IndexInput
   * @param info instance of FieldInfo
   * @return the field entry
   * @throws IOException
   */
  private FieldEntry readField(IndexInput input, FieldInfo info, int version) throws IOException {
    VectorEncoding vectorEncoding = readVectorEncoding(input);
    VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
    if (similarityFunction != info.getVectorSimilarityFunction()) {
      throw new IllegalStateException(
          "Inconsistent vector similarity function for field=\""
              + info.name
              + "\"; "
              + similarityFunction
              + " != "
              + info.getVectorSimilarityFunction());
    }
    return FieldEntry.readEntry(input, vectorEncoding, info.getVectorSimilarityFunction(), version);
  }

  /**
   * Gets the FieldEntry from the map using the field name. Check the encoding as well.
   *
   * @param field name of the field
   * @param expectedEncoding expected encoding
   * @return an instance of FieldEntry that Holds the meta information for the field
   */
  private FieldEntry getFieldEntry(String field, VectorEncoding expectedEncoding) {
    final FieldInfo info = fieldInfos.fieldInfo(field);
    final FieldEntry fieldEntry;
    if (info == null || (fieldEntry = fields.get(info.number)) == null) {
      throw new IllegalArgumentException("field=\"" + field + "\" not found");
    }
    if (fieldEntry.vectorEncoding != expectedEncoding) {
      throw new IllegalArgumentException(
          "field=\""
              + field
              + "\" is encoded as: "
              + fieldEntry.vectorEncoding
              + " expected: "
              + expectedEncoding);
    }
    return fieldEntry;
  }

  /**
   * Gets the FieldEntry for the given field name, or {@code null} when this segment holds no
   * cuVS index for it.
   *
   * @param field name of the field
   * @return the meta information for the field, or {@code null}
   */
  FieldEntry getFieldEntry(String field) {
    final FieldInfo info = fieldInfos.fieldInfo(field);
    return info == null ? null : fields.get(info.number);
  }

  /**
   * Deserializes the CAGRA index of a single field onto the GPU, bypassing the {@link GPUIndex}
   * cache. This is how {@link CuVS2510GPUVectorsWriter} obtains the inputs for the cuVS merge API:
   * a reader opened with {@link Context#MERGE} has no cached indexes at all, and a cached index
   * belongs to the {@link com.nvidia.cuvs.CuVSResources} of whichever thread opened the reader,
   * while the merge API requires every input to share one resources instance - the merging
   * thread's.
   *
   * <p>The returned index is owned by the caller and must be closed by it.
   *
   * @param field name of the field
   * @return a freshly loaded CAGRA index, or {@code null} if this segment has none for the field
   * @throws IOException I/O exception
   */
  CagraIndex openCagraIndexForMerge(String field) throws IOException {
    FieldEntry fieldEntry = getFieldEntry(field);
    if (fieldEntry == null || fieldEntry.cagraIndexLength() == 0) {
      return null;
    }
    try (var slice =
            cuvsIndexInput.slice(
                "cagra index", fieldEntry.cagraIndexOffset(), fieldEntry.cagraIndexLength());
        var in = new IndexInputInputStream(slice)) {
      CuVSResources mergeResources = getCuVSResourcesInstance();
      if (mergeResources == null) {
        throw new UnsupportedOperationException("cuVS is not supported");
      }
      return loadCagraIndex(field, fieldEntry, in, mergeResources);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
      throw new AssertionError("unreachable");
    }
  }

  /**
   * Invokes loadCuVSIndex for each field and returns the map of {@link GPUIndex}.
   *
   * @return the map containing {@link GPUIndex} objects
   * @throws IOException
   */
  private IntObjectHashMap<GPUIndex> loadCuVSIndices() throws IOException {
    var indices = new IntObjectHashMap<GPUIndex>();
    try {
      for (var field : fields) {
        var fieldEntry = field.value;
        int fieldNumber = field.key;
        FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldNumber);
        if (fieldInfo == null) {
          throw new CorruptIndexException("Invalid field number: " + fieldNumber, cuvsIndexInput);
        }
        var cuvsIndex = loadCuVSIndex(fieldInfo.name, fieldEntry);
        indices.put(fieldNumber, cuvsIndex);
      }
      return indices;
    } catch (Throwable t) {
      for (var index : indices) {
        try {
          index.value.close();
        } catch (Throwable closeFailure) {
          t.addSuppressed(closeFailure);
        }
      }
      throw Utils.handleThrowable(t);
    }
  }

  /**
   * Loads the CAGRA and brute force index (if exists) onto the GPU.
   *
   * @param fieldEntry instance of {@link FieldEntry}
   * @return return the instance of {@link GPUIndex}
   * @throws IOException
   */
  private GPUIndex loadCuVSIndex(String field, FieldEntry fieldEntry) throws IOException {
    CagraIndex cagraIndex = null;
    BruteForceIndex bruteForceIndex = null;
    try {
      long len = fieldEntry.cagraIndexLength();
      if (len > 0) {
        long off = fieldEntry.cagraIndexOffset();
        try (var slice = cuvsIndexInput.slice("cagra index", off, len);
            var in = new IndexInputInputStream(slice)) {
          cagraIndex = loadCagraIndex(field, fieldEntry, in, indexResources);
        }
      }
      len = fieldEntry.bruteForceIndexLength();
      if (len > 0) {
        long off = fieldEntry.bruteForceIndexOffset();
        try (var slice = cuvsIndexInput.slice("bf index", off, len);
            var in = new IndexInputInputStream(slice)) {
          bruteForceIndex = BruteForceIndex.newBuilder(indexResources).from(in).build();
        }
      }
    } catch (Throwable t) {
      closeAndSuppress(t, bruteForceIndex);
      closeAndSuppress(t, cagraIndex);
      throw Utils.handleThrowable(t);
    }
    return new GPUIndex(cagraIndex, bruteForceIndex);
  }

  static void closeAndSuppress(Throwable failure, AutoCloseable closeable) {
    if (closeable == null) {
      return;
    }
    try {
      closeable.close();
    } catch (Throwable closeFailure) {
      failure.addSuppressed(closeFailure);
    }
  }

  /** Loads either the original graph-and-dataset format or a graph hydrated from flat vectors. */
  private CagraIndex loadCagraIndex(
      String field,
      FieldEntry fieldEntry,
      IndexInputInputStream inputStream,
      CuVSResources resources)
      throws Throwable {
    if (fieldEntry.cagraPersistenceMode() == GRAPH_AND_DATASET) {
      return CagraIndex.newBuilder(resources).from(inputStream).build();
    }

    CuVSMatrix dataset =
        withSequentialFlatVectors(
            flatVectorsReader,
            flatVectorsAlreadySequential,
            hydrationReader -> replayFlatVectors(field, fieldEntry, hydrationReader, resources),
            CuVSMatrix::close);
    try {
      return CagraIndex.newBuilder(resources).fromGraph(inputStream).withDataset(dataset).build();
    } catch (Throwable t) {
      closeAndSuppress(t, dataset);
      throw t;
    }
  }

  /** Replays the flat-vector ordinals exactly, because CAGRA row ids are vector ordinals. */
  private CuVSMatrix replayFlatVectors(
      String field,
      FieldEntry fieldEntry,
      FlatVectorsReader hydrationReader,
      CuVSResources resources)
      throws IOException {
    FloatVectorValues values = hydrationReader.getFloatVectorValues(field);
    if (values == null) {
      throw new CorruptIndexException(
          "Missing flat vectors for CAGRA field \"" + field + "\"", cuvsIndexInput);
    }
    if (values.dimension() != fieldEntry.dims()) {
      throw new CorruptIndexException(
          "Flat-vector dimension mismatch for field \""
              + field
              + "\": expected "
              + fieldEntry.dims()
              + " but got "
              + values.dimension(),
          cuvsIndexInput);
    }
    if (values.size() != fieldEntry.count()) {
      throw new CorruptIndexException(
          "Flat-vector count mismatch for field \""
              + field
              + "\": expected "
              + fieldEntry.count()
              + " but got "
              + values.size(),
          cuvsIndexInput);
    }

    try (CuVSMatrix.Builder<?> builder =
        CuVSMatrix.cagraPaddedDeviceBuilder(
            resources, fieldEntry.count(), fieldEntry.dims(), CuVSMatrix.DataType.FLOAT)) {
      KnnVectorValues.DocIndexIterator iterator = values.iterator();
      int expectedOrdinal = 0;
      for (int doc = iterator.nextDoc(); doc != NO_MORE_DOCS; doc = iterator.nextDoc()) {
        int ordinal = iterator.index();
        if (ordinal != expectedOrdinal) {
          throw new CorruptIndexException(
              "Non-contiguous flat-vector ordinal for field \""
                  + field
                  + "\": expected "
                  + expectedOrdinal
                  + " but got "
                  + ordinal,
              cuvsIndexInput);
        }
        float[] vector = values.vectorValue(ordinal);
        if (vector.length != fieldEntry.dims()) {
          throw new CorruptIndexException(
              "Flat-vector row dimension mismatch for field \""
                  + field
                  + "\" at ordinal "
                  + ordinal
                  + ": expected "
                  + fieldEntry.dims()
                  + " but got "
                  + vector.length,
              cuvsIndexInput);
        }
        builder.addVector(vector);
        expectedOrdinal++;
      }
      if (expectedOrdinal != fieldEntry.count()) {
        throw new CorruptIndexException(
            "Flat-vector replay count mismatch for field \""
                + field
                + "\": expected "
                + fieldEntry.count()
                + " but replayed "
                + expectedOrdinal,
            cuvsIndexInput);
      }
      return builder.build();
    }
  }

  @FunctionalInterface
  interface FlatReaderOperation<T> {
    T apply(FlatVectorsReader reader) throws Throwable;
  }

  @FunctionalInterface
  interface ResultCleanup<T> {
    void close(T result) throws Throwable;
  }

  /** Runs synchronous flat-vector replay under Lucene's sequential-reader lifecycle. */
  static <T> T withSequentialFlatVectors(
      FlatVectorsReader reader,
      boolean readerAlreadySequential,
      FlatReaderOperation<T> operation,
      ResultCleanup<? super T> resultCleanup)
      throws IOException {
    if (readerAlreadySequential) {
      // MERGE/FLUSH IOContexts are required to use SEQUENTIAL advice. Lucene's flat reader keeps
      // that advice when its constructor asks for RANDOM, so finishMerge() here would incorrectly
      // reset a reader that remains owned by the surrounding operation.
      try {
        return operation.apply(reader);
      } catch (Throwable t) {
        throw Utils.handleThrowable(t);
      }
    }

    FlatVectorsReader sequentialReader = reader.getMergeInstance();
    T result;
    try {
      result = operation.apply(sequentialReader);
    } catch (Throwable operationFailure) {
      try {
        sequentialReader.finishMerge();
      } catch (Throwable finishFailure) {
        operationFailure.addSuppressed(finishFailure);
      }
      throw Utils.handleThrowable(operationFailure);
    }

    try {
      sequentialReader.finishMerge();
    } catch (Throwable finishFailure) {
      try {
        resultCleanup.close(result);
      } catch (Throwable cleanupFailure) {
        finishFailure.addSuppressed(cleanupFailure);
      }
      throw Utils.handleThrowable(finishFailure);
    }
    return result;
  }

  /**
   * Closes the resources.
   */
  @Override
  public synchronized void close() throws IOException {
    if (closed) {
      return;
    }
    closed = true;
    List<Closeable> closeables = new ArrayList<>();
    closeables.add(flatVectorsReader);
    closeables.add(cuvsIndexInput);
    if (cuvsIndices != null) {
      stream(cuvsIndices.values().iterator()).map(cursor -> cursor.value).forEach(closeables::add);
    }
    if (indexResources != null) {
      closeables.add(indexResources::close);
    }
    IOUtils.close(closeables);
  }

  static <T> Stream<T> stream(Iterator<T> iterator) {
    return StreamSupport.stream(((Iterable<T>) () -> iterator).spliterator(), false);
  }

  /**
   * Checks consistency of this reader.
   */
  @Override
  public void checkIntegrity() throws IOException {
    flatVectorsReader.checkIntegrity();
    CodecUtil.checksumEntireFile(cuvsIndexInput);
  }

  /**
   * Returns the FloatVectorValues for the given field.
   */
  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return flatVectorsReader.getFloatVectorValues(field);
  }

  /**
   * Returns the ByteVectorValues for the given field.
   *
   * This is not supported.
   */
  @Override
  public ByteVectorValues getByteVectorValues(String field) {
    throw new UnsupportedOperationException("byte vectors are not currently supported");
  }

  /** Native float to float function */
  private interface FloatToFloatFunction {
    float apply(float v);
  }

  /**
   * Get the score normalization function.
   *
   * @param sim instance of VectorSimilarityFunction
   * @return an instance of the FloatToFloatFunction
   */
  private static FloatToFloatFunction getScoreNormalizationFunc(VectorSimilarityFunction sim) {
    // TODO: check for different similarities
    return score -> (1f / (1f + score));
  }

  /**
   * Returns the k nearest neighbor documents using cuVS's CAGRA or brute force algorithm for this field, to the given vector.
   */
  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    var fieldEntry = getFieldEntry(field, VectorEncoding.FLOAT32);
    if (fieldEntry.count() == 0 || knnCollector.k() == 0) {
      return;
    }

    var fieldNumber = fieldInfos.fieldInfo(field).number;
    GPUIndex cuvsIndex = cuvsIndices != null ? cuvsIndices.get(fieldNumber) : null;
    if (cuvsIndex == null) {
      throw new IllegalStateException("Index not found for field:" + field);
    }

    final FloatVectorValues rawValues = flatVectorsReader.getFloatVectorValues(field);
    final Bits acceptedOrds = rawValues.getAcceptOrds(acceptDocs);
    BitSet[] mask = null;
    int maskLength = 0;
    int topK = knnCollector.k();

    if (acceptDocs != null) {
      mask = new BitSet[1]; // As there is only one query "target"
      mask[0] = new BitSet(acceptedOrds.length());
      /*
       * Need to find if there is a better alternative for below loop
       * in subsequent code improvement iterations. This is needed as
       * there is a difference between Lucene's Bits "acceptDocs" and
       * what our API accepts.
       */
      for (int i = 0; i < acceptedOrds.length(); i++) {
        if (acceptedOrds.get(i)) {
          mask[0].set(i);
        }
      }
      int cardinality = mask[0].cardinality();
      if (cardinality == 0) {
        // Every vector in this segment is deleted or filtered out. cuVS cannot be asked for zero
        // neighbours, and Lucene's contract for such a leaf is to collect nothing, so return
        // before touching the GPU.
        return;
      }
      topK = Math.min(knnCollector.k() + 10, cardinality);
      // numDocs must be the total vector count so cuVS sizes the prefilter to cover every ordinal.
      // BitSet.length() is (highest set bit + 1), which under a selective filter is smaller than
      // the
      // vector count, leaving the trailing ordinals outside the filter and thus default-accepted.
      maskLength = acceptedOrds.length();
    }

    try {
      List<Map<Integer, Float>> searchResult = null;
      if (knnCollector.k() <= 1024 && cuvsIndex.getCagraIndex() != null) {
        CagraSearchParams searchParams;
        if (knnCollector instanceof GPUPerLeafCuVSKnnCollector) {
          GPUPerLeafCuVSKnnCollector collector = (GPUPerLeafCuVSKnnCollector) knnCollector;
          searchParams =
              new CagraSearchParams.Builder()
                  .withItopkSize(Math.max(collector.getiTopK(), topK))
                  .withSearchWidth(collector.getSearchWidth())
                  .withThreadBlockSize(collector.getThreadBlockSize())
                  .withMaxIterations(collector.getMaxIterations())
                  .withAlgo(collector.getSearchAlgo())
                  .build();
        } else {
          // Setting itopK as topK because in any case iTopK should be ATLEAST equal to topK
          searchParams = new CagraSearchParams.Builder().withItopkSize(topK).build();
        }
        CagraIndex cagraIndex = cuvsIndex.getCagraIndex();
        assert cagraIndex != null;
        CuVSResources queryResources = getCuVSResourcesInstance();
        try (CuVSMatrix.Builder<?> builder =
            CuVSMatrix.deviceBuilder(queryResources, 1, target.length, CuVSMatrix.DataType.FLOAT)) {
          builder.addVector(target);
          try (CuVSMatrix queryVector = builder.build()) {
            CagraQuery.Builder queryBuilder =
                new CagraQuery.Builder(queryResources)
                    .withTopK(topK)
                    .withSearchParams(searchParams)
                    .withQueryVectors(queryVector);
            if (acceptDocs != null) {
              queryBuilder.withPrefilter(mask[0], maskLength);
            }
            searchResult = cagraIndex.search(queryBuilder.build()).getResults();
          }
        }
      } else {
        BruteForceIndex bruteforceIndex = cuvsIndex.getBruteforceIndex();
        assert bruteforceIndex != null;
        BruteForceQuery query = null;
        float[][] queryVector = new float[][] {target};
        if (acceptDocs != null) {
          query =
              new BruteForceQuery.Builder(getCuVSResourcesInstance())
                  .withQueryVectors(queryVector)
                  .withPrefilters(mask, maskLength)
                  .withTopK(topK)
                  .build();
        } else {
          query =
              new BruteForceQuery.Builder(getCuVSResourcesInstance())
                  .withQueryVectors(queryVector)
                  .withTopK(topK)
                  .build();
        }
        searchResult = bruteforceIndex.search(query).getResults();
      }

      // List expected to have only one entry because of single query "target". A zero-row
      // response can only mean no neighbours were produced, so there is nothing to collect.
      if (searchResult.isEmpty()) {
        return;
      }
      assert searchResult.size() == 1;
      final IntToIntFunction ordToDocFunction = (IntToIntFunction) rawValues::ordToDoc;
      final FloatToFloatFunction scoreCorrectionFunction =
          getScoreNormalizationFunc(fieldEntry.similarityFunction);

      for (Entry<Integer, Float> entry : searchResult.getFirst().entrySet()) {
        int ord = entry.getKey();
        float score = entry.getValue();
        if (knnCollector.earlyTerminated()) {
          break;
        }
        // Empty top-k slots (fewer than k passing candidates) carry a sentinel distance of FLT_MAX.
        // Prefer this over the neighbor-index sentinel: the index sentinel is not uniform across
        // CAGRA search algorithms (single-CTA emits 0x7FFFFFFF, multi-CTA 0xFFFFFFFF), so the
        // distance is the reliable, algorithm-independent signal for an empty slot.
        if (score == Float.MAX_VALUE || ord < 0) {
          continue;
        }
        float correctedScore = scoreCorrectionFunction.apply(score);
        int doc = ordToDocFunction.apply(ord);
        knnCollector.incVisitedCount(1);
        knnCollector.collect(doc, correctedScore);
      }
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Return the k nearest neighbor documents as determined by comparison of their vector values for this field, to the given vector.
   *
   * This is not supported.
   */
  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    throw new UnsupportedOperationException("Byte vectors are not currently supported");
  }

  /**
   * Holds the meta information for the field.
   */
  record FieldEntry(
      VectorEncoding vectorEncoding,
      VectorSimilarityFunction similarityFunction,
      int dims,
      int count,
      long cagraIndexOffset,
      long cagraIndexLength,
      long bruteForceIndexOffset,
      long bruteForceIndexLength,
      GPUSearchParams.CagraPersistenceMode cagraPersistenceMode) {

    /**
     * Returns an instance of FieldEntry.
     *
     * @param input instance of IndexInput
     * @param vectorEncoding The numeric data type of the vector values
     * @param similarityFunction Vector similarity function; used in search to return top K most similar vectors to a target vector
     * @return an instance of FieldEntry
     * @throws IOException I/O Exceptions
     */
    static FieldEntry readEntry(
        IndexInput input,
        VectorEncoding vectorEncoding,
        VectorSimilarityFunction similarityFunction,
        int version)
        throws IOException {
      var dims = input.readInt();
      var count = input.readInt();
      GPUSearchParams.CagraPersistenceMode persistenceMode = GRAPH_AND_DATASET;
      if (version >= VERSION_GRAPH_ONLY_PERSISTENCE) {
        int persistenceModeId = input.readInt();
        try {
          persistenceMode = GPUSearchParams.CagraPersistenceMode.fromId(persistenceModeId);
        } catch (IllegalArgumentException e) {
          throw new CorruptIndexException(
              "Invalid CAGRA persistence mode: " + persistenceModeId, input);
        }
      }
      var cagraIndexOffset = input.readVLong();
      var cagraIndexLength = input.readVLong();
      var bruteForceIndexOffset = input.readVLong();
      var bruteForceIndexLength = input.readVLong();
      return new FieldEntry(
          vectorEncoding,
          similarityFunction,
          dims,
          count,
          cagraIndexOffset,
          cagraIndexLength,
          bruteForceIndexOffset,
          bruteForceIndexLength,
          persistenceMode);
    }
  }

  /**
   * Checks the version and throws CorruptIndexException on mismatch.
   *
   * @param versionMeta
   * @param versionVectorData
   * @param in
   * @throws CorruptIndexException
   */
  private static void checkVersion(int versionMeta, int versionVectorData, IndexInput in)
      throws CorruptIndexException {
    if (versionMeta != versionVectorData) {
      throw new CorruptIndexException(
          "Format versions mismatch: meta="
              + versionMeta
              + ", "
              + CUVS_META_CODEC_NAME
              + "="
              + versionVectorData,
          in);
    }
  }

  /**
   * Returns the {@link CagraIndex} for the given field, or {@code null} if unavailable
   * (e.g., during a merge or when the field is missing).
   *
   * @param field the vector field name
   * @return the CAGRA index, or {@code null}
   */
  public CagraIndex getCagraIndexForField(String field) {
    if (cuvsIndices == null) return null;
    FieldInfo info = fieldInfos.fieldInfo(field);
    if (info == null) return null;
    GPUIndex gpuIndex = cuvsIndices.get(info.number);
    if (gpuIndex == null) return null;
    return gpuIndex.getCagraIndex();
  }

  /** Returns the filter cache owned by the vectors format that created this reader. */
  FilterBitsetCache getFilterBitsetCache() {
    return filterBitsetCache;
  }

  /**
   * Gets the instance of FieldInfos.
   *
   * @return the instance of FieldInfos
   */
  public FieldInfos getFieldInfos() {
    return fieldInfos;
  }

  /**
   * Gets the map of {@link GPUIndex} objects.
   *
   * @return the map of GPU index objects
   */
  public IntObjectHashMap<GPUIndex> getCuvsIndexes() {
    return cuvsIndices;
  }

  /**
   * Gets the map of FieldEntry objects that hold the meta information for the field.
   *
   * @return the map of FieldEntry objects
   */
  public IntObjectHashMap<FieldEntry> getFieldEntries() {
    return fields;
  }
}
