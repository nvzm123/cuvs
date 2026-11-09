/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.quantization.QuantizedVectorsReader;

/**
 * A CPU float-vector query that selects a starting point for a single-layer HNSW graph by scoring
 * random vector ordinals. The best sampled ordinal seeds one Lucene HNSW traversal; this does not
 * run a separate search for every sample.
 *
 * <p>Sampling is with replacement and deterministic for the target, field, segment document base,
 * and vector count. It includes physically stored vectors that may be deleted or rejected by a
 * filter, just as ordinary HNSW traversal may use those nodes. Lucene still enforces the original
 * filter and live-document mask on returned results. Each sample consumes one visited-vector unit;
 * the winning node is scored and counted again by Lucene when traversal starts.
 *
 * <p>Only unquantized float-vector readers exposing a one-level {@link HnswGraph} are seeded.
 * Quantized readers, other graph topologies, non-codec readers, and explicitly seeded strategies
 * retain Lucene's search behavior. The query does not change an index or require a GPU. Lucene
 * continues to own exact-search fallback, result collection, and multi-segment search.
 *
 * @since 26.12
 */
public class BaseLayerKnnFloatVectorQuery extends KnnFloatVectorQuery {
  public static final int DEFAULT_NUM_ENTRY_POINTS = 42;

  private final int numEntryPoints;

  /** Creates an unfiltered query using {@value #DEFAULT_NUM_ENTRY_POINTS} samples. */
  public BaseLayerKnnFloatVectorQuery(String field, float[] target, int k) {
    this(field, target, k, null);
  }

  /** Creates a filtered query using {@value #DEFAULT_NUM_ENTRY_POINTS} samples. */
  public BaseLayerKnnFloatVectorQuery(String field, float[] target, int k, Query filter) {
    this(field, target, k, filter, DEFAULT_NUM_ENTRY_POINTS);
  }

  /**
   * Creates a query with the requested number of random samples per eligible segment search.
   *
   * @param field vector field
   * @param target query vector
   * @param k number of results, at least one
   * @param filter optional document filter
   * @param numEntryPoints number of samples, nonnegative; zero disables random entry selection
   */
  public BaseLayerKnnFloatVectorQuery(
      String field, float[] target, int k, Query filter, int numEntryPoints) {
    super(field, Objects.requireNonNull(target, "target").clone(), k, filter);
    LuceneVersionGuard.ensureCompatible();
    if (numEntryPoints < 0) {
      throw new IllegalArgumentException("numEntryPoints must be nonnegative");
    }
    this.numEntryPoints = numEntryPoints;
  }

  /** Returns the number of candidate entry-point samples, not the number of graph traversals. */
  public int getNumEntryPoints() {
    return numEntryPoints;
  }

  @Override
  protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
    KnnCollectorManager delegate = super.getKnnCollectorManager(k, searcher);
    if (numEntryPoints == 0) {
      return delegate;
    }
    return new CompatSeededKnnCollectorManager(delegate) {
      @Override
      KnnCollector seed(KnnCollector collector, LeafReaderContext context) throws IOException {
        return seedCollector(collector, context, searcher.getTimeout());
      }
    };
  }

  private KnnCollector seedCollector(
      KnnCollector collector, LeafReaderContext context, QueryTimeout timeout) throws IOException {
    // In particular, preserve the seeds supplied by Lucene's optimistic second search of a leaf.
    if (collector.getSearchStrategy() instanceof KnnSearchStrategy.Seeded
        || !(context.reader() instanceof CodecReader codecReader)) {
      return collector;
    }
    FieldInfo fieldInfo = codecReader.getFieldInfos().fieldInfo(field);
    if (fieldInfo == null
        || !fieldInfo.hasVectorValues()
        || fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32) {
      return collector;
    }
    KnnVectorsReader vectorsReader = codecReader.getVectorReader();
    if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      vectorsReader = fieldsReader.getFieldReader(field);
    }
    if (!(vectorsReader instanceof HnswGraphProvider graphProvider)
        || (vectorsReader instanceof QuantizedVectorsReader quantized
            && quantized.getQuantizedVectorValues(field) != null)) {
      return collector;
    }
    HnswGraph graph = graphProvider.getGraph(field);
    if (graph.size() == 0 || graph.numLevels() != 1) {
      return collector;
    }
    FloatVectorValues vectors = codecReader.getFloatVectorValues(field);
    if (vectors == null || vectors.size() != graph.size()) {
      return collector;
    }
    long seed = 31L * Arrays.hashCode(target) + field.hashCode();
    seed = 31L * seed + context.docBase;
    seed = 31L * seed + vectors.size();
    DocIdSetIterator entryPoint =
        new SampledEntryPoint(
            vectors, fieldInfo.getVectorSimilarityFunction(), collector, timeout, seed);
    KnnSearchStrategy original = collector.getSearchStrategy();
    KnnSearchStrategy.Seeded strategy =
        new KnnSearchStrategy.Seeded(
            entryPoint, 1, original == null ? KnnSearchStrategy.Hnsw.DEFAULT : original);
    return new KnnCollector.Decorator(collector) {
      @Override
      public KnnSearchStrategy getSearchStrategy() {
        return strategy;
      }
    };
  }

  /**
   * Sampling is deferred until HNSW consumes its seed, so an exhaustive reader search or an unused
   * collector does not spend its visit budget selecting an entry point.
   */
  private final class SampledEntryPoint extends DocIdSetIterator {
    private final FloatVectorValues vectors;
    private final VectorSimilarityFunction similarity;
    private final KnnCollector collector;
    private final QueryTimeout timeout;
    private final long seed;
    private int ordinal = -1;

    private SampledEntryPoint(
        FloatVectorValues vectors,
        VectorSimilarityFunction similarity,
        KnnCollector collector,
        QueryTimeout timeout,
        long seed) {
      this.vectors = vectors;
      this.similarity = similarity;
      this.collector = collector;
      this.timeout = timeout;
      this.seed = seed;
    }

    @Override
    public int docID() {
      return ordinal;
    }

    @Override
    public int nextDoc() throws IOException {
      if (ordinal != -1) {
        return ordinal = NO_MORE_DOCS;
      }
      SplittableRandom random = new SplittableRandom(seed);
      int bestOrdinal = 0;
      float bestScore = Float.NEGATIVE_INFINITY;
      for (int i = 0; i < numEntryPoints; i++) {
        // The timeout wrapper is installed outside this manager, after this collector is created.
        if (collector.earlyTerminated() || (timeout != null && timeout.shouldExit())) {
          break;
        }
        int candidate = random.nextInt(vectors.size());
        float score = similarity.compare(target, vectors.vectorValue(candidate));
        collector.incVisitedCount(1);
        if (score > bestScore) {
          bestScore = score;
          bestOrdinal = candidate;
        }
      }
      return ordinal = bestOrdinal;
    }

    @Override
    public int advance(int target) throws IOException {
      return slowAdvance(target);
    }

    @Override
    public long cost() {
      return 1;
    }
  }

  @Override
  public boolean equals(Object other) {
    return super.equals(other)
        && numEntryPoints == ((BaseLayerKnnFloatVectorQuery) other).numEntryPoints;
  }

  @Override
  public int hashCode() {
    return 31 * super.hashCode() + numEntryPoints;
  }

  @Override
  public String toString(String field) {
    return super.toString(field) + "[numEntryPoints=" + numEntryPoints + "]";
  }
}
