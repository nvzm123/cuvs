/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.function.Function;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.quantization.QuantizedVectorsReader;

/** Exercises query-side seeding with persisted CPU-built graphs, without a GPU or native library. */
public class TestBaseLayerKnnFloatVectorQuery extends LuceneTestCase {

  private static final String FIELD = "vector";
  private static final int VECTOR_COUNT = 64;
  private static final int SINGLE_LAYER_MAX_CONN = 512;
  private static final int MULTI_LAYER_MAX_CONN = 2;
  private static final int BEAM_WIDTH = 100;
  private static final int SEGMENT_ID_STRIDE = 1000;
  private static final int DELETED_VECTOR_ORDINAL = VECTOR_COUNT - 2;
  // The fixed query/leaf sample includes this self-match; its first and last draws are 23 and 30.
  private static final int SAMPLE_SELF_MATCH_ORDINAL = 47;

  public void testFactoryDefaultsExplicitZeroAndQueryIdentity() {
    Function<Map<String, Object>, Query> factory = new BaseLayerKnnFloatVectorQueryFactory();
    Map<String, Object> request = new HashMap<>();
    request.put("field", FIELD);
    request.put("target", new float[] {7f});
    request.put("k", 5);

    var defaults = (BaseLayerKnnFloatVectorQuery) factory.apply(request);
    assertEquals(42, defaults.getNumEntryPoints());
    assertEquals(new BaseLayerKnnFloatVectorQuery(FIELD, new float[] {7f}, 5), defaults);
    assertEquals(
        new BaseLayerKnnFloatVectorQuery(FIELD, new float[] {7f}, 5).hashCode(),
        defaults.hashCode());
    request.put("num_entry_points", 0);
    var disabled = (BaseLayerKnnFloatVectorQuery) factory.apply(request);
    assertEquals(0, disabled.getNumEntryPoints());
    assertNotEquals(defaults, disabled);
    assertEquals(4, request.size());

    for (Object invalid : new Object[] {-1, 42L, "42", null}) {
      request.put("num_entry_points", invalid);
      expectThrows(IllegalArgumentException.class, () -> factory.apply(request));
    }
    request.put("num_entry_points", 42);
    request.put("unrecognized", 1);
    expectThrows(IllegalArgumentException.class, () -> factory.apply(request));
    expectThrows(
        IllegalArgumentException.class,
        () -> new BaseLayerKnnFloatVectorQuery(FIELD, new float[] {7f}, 5, null, -1));
  }

  public void testSelectsOneSeedLazilyAndRepeatablyOnSavedBaseLayer() throws Exception {
    try (Directory directory = newDirectory()) {
      writeIndex(directory, SINGLE_LAYER_MAX_CONN, 1, false, false);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReaderContext leaf = reader.leaves().get(0);
        assertEquals(1, graphOf(leaf.reader()).numLevels());
        IndexSearcher searcher = new IndexSearcher(reader);
        var query =
            new BaseLayerKnnFloatVectorQuery(FIELD, new float[] {SAMPLE_SELF_MATCH_ORDINAL}, 5);
        if (Version.LATEST.minor >= 3) {
          KnnCollectorManager manager = query.getKnnCollectorManager(5, searcher);
          // These methods first appeared in 10.3; invoking the public interface keeps this test
          // source compatible with 10.2 while proving that the version adapter preserves them.
          assertEquals(true, KnnCollectorManager.class.getMethod("isOptimistic").invoke(manager));
          KnnCollector optimistic =
              (KnnCollector)
                  KnnCollectorManager.class
                      .getMethod(
                          "newOptimisticCollector",
                          int.class,
                          KnnSearchStrategy.class,
                          LeafReaderContext.class,
                          int.class)
                      .invoke(manager, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT, leaf, 2);
          assertEquals("preserve Lucene's per-leaf k, not the query k", 2, optimistic.k());
          assertEquals(1, seeded(optimistic).numberOfEntryPoints());
        }
        int selected = -1;
        for (int repeat = 0; repeat < 3; repeat++) {
          KnnCollector collector = collector(query, searcher, leaf, Integer.MAX_VALUE);
          KnnSearchStrategy.Seeded strategy = seeded(collector);
          assertEquals(
              "samples choose one search entry, not 42 searches",
              1,
              strategy.numberOfEntryPoints());
          assertEquals(
              "selection must wait until HNSW consumes the seed", 0, collector.visitedCount());
          int ordinal = strategy.entryPoints().nextDoc();
          assertTrue(ordinal >= 0 && ordinal < VECTOR_COUNT);
          // This fixed query/leaf sample contains the self-match. Its independently known score
          // of 1 must beat every other sampled vector, regardless of their visitation order.
          assertEquals(
              "select the best sample, not the first or last draw",
              SAMPLE_SELF_MATCH_ORDINAL,
              ordinal);
          assertEquals(42, collector.visitedCount());
          assertEquals(NO_MORE_DOCS, strategy.entryPoints().nextDoc());
          if (repeat == 0) {
            selected = ordinal;
          } else {
            assertEquals("repeated queries must choose the same entry", selected, ordinal);
          }
        }

        KnnCollector collector = collector(query, searcher, leaf, Integer.MAX_VALUE);
        TestLuceneCompat.searchNearestVectors(
            leaf.reader(), FIELD, query.getTargetCopy(), collector, leaf.reader().getLiveDocs());
        assertTrue("the real HNSW reader must consume the samples", collector.visitedCount() > 42);
        TopDocs hits = collector.topDocs();
        assertEquals(5, hits.scoreDocs.length);
        assertEquals(
            Integer.toString(SAMPLE_SELF_MATCH_ORDINAL),
            leaf.reader().storedFields().document(hits.scoreDocs[0].doc).get("id"));
        assertEquals(1f, hits.scoreDocs[0].score, 0f);
      }
    }
  }

  public void testSamplingWithReplacementOnTinyGraphAndPreservesExplicitSeeds() throws Exception {
    try (Directory directory = newDirectory()) {
      writeIndex(directory, SINGLE_LAYER_MAX_CONN, 1, false, false, 2);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReaderContext leaf = reader.leaves().get(0);
        assertEquals(1, graphOf(leaf.reader()).numLevels());
        IndexSearcher searcher = new IndexSearcher(reader);
        var query = new BaseLayerKnnFloatVectorQuery(FIELD, new float[] {1f}, 1);
        KnnCollector collector = collector(query, searcher, leaf, Integer.MAX_VALUE);
        assertEquals(1, seeded(collector).entryPoints().nextDoc());
        assertEquals(
            "42 draws are allowed even when the graph has only two vectors",
            42,
            collector.visitedCount());

        KnnSearchStrategy.Seeded original =
            new KnnSearchStrategy.Seeded(
                org.apache.lucene.search.DocIdSetIterator.all(2),
                1,
                KnnSearchStrategy.Hnsw.DEFAULT);
        KnnCollector explicitlySeeded =
            query
                .getKnnCollectorManager(1, searcher)
                .newCollector(Integer.MAX_VALUE, original, leaf);
        assertSame(
            "Lucene's reentrant seed must take precedence",
            original,
            explicitlySeeded.getSearchStrategy());
        assertEquals(0, original.entryPoints().nextDoc());
        assertEquals(0, explicitlySeeded.visitedCount());
      }
    }
  }

  public void testSelectionHonorsVisitBudgetAndTimeout() throws Exception {
    try (Directory directory = newDirectory()) {
      writeIndex(directory, SINGLE_LAYER_MAX_CONN, 1, false, false);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReaderContext leaf = reader.leaves().get(0);
        assertEquals(1, graphOf(leaf.reader()).numLevels());
        IndexSearcher searcher = new IndexSearcher(reader);
        var query =
            new BaseLayerKnnFloatVectorQuery(FIELD, new float[] {SAMPLE_SELF_MATCH_ORDINAL}, 5);
        KnnCollector collector = collector(query, searcher, leaf, 3);
        seeded(collector).entryPoints().nextDoc();
        assertEquals("sampling shares Lucene's visit budget", 3, collector.visitedCount());
        assertTrue(collector.earlyTerminated());

        searcher.setTimeout(() -> true);
        collector = collector(query, searcher, leaf, Integer.MAX_VALUE);
        seeded(collector).entryPoints().nextDoc();
        assertEquals("an expired query must not score samples", 0, collector.visitedCount());
        assertEquals(0, searcher.search(query, 5).scoreDocs.length);
      }
    }
  }

  public void testReaderExhaustiveSearchDoesNotSpendBudgetOnSamples() throws Exception {
    try (Directory directory = newDirectory()) {
      writeIndex(directory, SINGLE_LAYER_MAX_CONN, 1, false, false);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReaderContext leaf = reader.leaves().get(0);
        assertEquals(1, graphOf(leaf.reader()).numLevels());
        var query =
            new BaseLayerKnnFloatVectorQuery(
                FIELD, new float[] {SAMPLE_SELF_MATCH_ORDINAL}, VECTOR_COUNT);
        KnnCollector collector =
            collector(query, new IndexSearcher(reader), leaf, Integer.MAX_VALUE);
        TestLuceneCompat.searchNearestVectors(
            leaf.reader(), FIELD, query.getTargetCopy(), collector, null);
        assertEquals(
            "the exhaustive reader path must not also sample",
            VECTOR_COUNT,
            collector.visitedCount());
        assertEquals(VECTOR_COUNT, collector.topDocs().scoreDocs.length);
      }
    }
  }

  public void testZeroAndMultilayerQueriesKeepLuceneSearch() throws Exception {
    for (int maxConn : new int[] {SINGLE_LAYER_MAX_CONN, MULTI_LAYER_MAX_CONN}) {
      try (Directory directory = newDirectory()) {
        writeIndex(directory, maxConn, 1, false, false);
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
          LeafReaderContext leaf = reader.leaves().get(0);
          HnswGraph graph = graphOf(leaf.reader());
          if (maxConn == SINGLE_LAYER_MAX_CONN) {
            assertEquals(1, graph.numLevels());
          } else {
            assertTrue("the control must really have upper layers", graph.numLevels() > 1);
          }
          IndexSearcher searcher = new IndexSearcher(reader);
          int samples = maxConn == SINGLE_LAYER_MAX_CONN ? 0 : 42;
          var query =
              new BaseLayerKnnFloatVectorQuery(
                  FIELD, new float[] {SAMPLE_SELF_MATCH_ORDINAL}, 5, null, samples);
          assertFalse(
              collector(query, searcher, leaf, Integer.MAX_VALUE).getSearchStrategy()
                  instanceof KnnSearchStrategy.Seeded);
          assertSameHits(
              searcher.search(
                  new KnnFloatVectorQuery(FIELD, new float[] {SAMPLE_SELF_MATCH_ORDINAL}, 5), 5),
              searcher.search(query, 5));
        }
      }
    }
  }

  public void testSparseDeletedAndFilteredDocumentsAcrossSegments() throws Exception {
    try (Directory directory = newDirectory()) {
      writeIndex(directory, SINGLE_LAYER_MAX_CONN, 2, true, true);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(2, reader.leaves().size());
        IndexSearcher searcher = new IndexSearcher(reader);
        float target = SEGMENT_ID_STRIDE + DELETED_VECTOR_ORDINAL;
        for (LeafReaderContext leaf : reader.leaves()) {
          assertEquals(1, graphOf(leaf.reader()).numLevels());
          FloatVectorValues values = leaf.reader().getFloatVectorValues(FIELD);
          assertEquals(VECTOR_COUNT, values.size());
          assertTrue(
              "the fixture must have missing-vector documents",
              values.size() < leaf.reader().maxDoc());
          assertNotEquals("vector ordinals must differ from doc IDs", 0, values.ordToDoc(0));
          assertTrue(
              "the physical graph must retain a deleted vector", leaf.reader().hasDeletions());
          var query = new BaseLayerKnnFloatVectorQuery(FIELD, new float[] {target}, 5);
          KnnCollector collector = collector(query, searcher, leaf, 3);
          int selected = seeded(collector).entryPoints().nextDoc();
          assertTrue(selected >= 0 && selected < values.size());
          assertEquals("each leaf owns its own visit budget", 3, collector.visitedCount());
        }

        Query filter = new TermQuery(new Term("parity", "even"));
        var query = new BaseLayerKnnFloatVectorQuery(FIELD, new float[] {target}, 5, filter);
        TopDocs hits = searcher.search(query, 5);
        assertEquals(5, hits.scoreDocs.length);
        Set<Integer> seen = new HashSet<>();
        for (int rank = 0; rank < hits.scoreDocs.length; rank++) {
          int id =
              Integer.parseInt(
                  searcher.storedFields().document(hits.scoreDocs[rank].doc).get("id"));
          assertTrue(seen.add(id));
          assertEquals(
              "nearest live even vector, independently ordered by distance",
              SEGMENT_ID_STRIDE + DELETED_VECTOR_ORDINAL - 2 * (rank + 1),
              id);
          assertEquals(1f / (1f + (target - id) * (target - id)), hits.scoreDocs[rank].score, 0f);
        }
        assertSameHits(hits, searcher.search(query, 5));
        assertEquals(
            0,
            searcher.search(new BaseLayerKnnFloatVectorQuery("missing", new float[] {1f}, 5), 5)
                .scoreDocs
                .length);
      }
    }
  }

  public void testEmptyIndexHasNoResults() throws Exception {
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig())) {
        writer.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(
            0,
            new IndexSearcher(reader)
                .search(new BaseLayerKnnFloatVectorQuery(FIELD, new float[] {1f}, 5), 5)
                .scoreDocs
                .length);
      }
    }
  }

  public void testScalarQuantizedBaseLayerKeepsLuceneSearch() throws Exception {
    try (Directory directory = newDirectory()) {
      writeIndex(directory, scalarQuantizedFormat(), 1, false, false, VECTOR_COUNT);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReaderContext leaf = reader.leaves().get(0);
        assertEquals(
            "quantization, not topology, must bypass sampling",
            1,
            graphOf(leaf.reader()).numLevels());
        KnnVectorsReader vectorsReader = vectorsReaderOf(leaf.reader());
        assertTrue(vectorsReader instanceof QuantizedVectorsReader);
        QuantizedVectorsReader quantized = (QuantizedVectorsReader) vectorsReader;
        assertNotNull(
            "the fixture must actually store quantized vectors",
            quantized.getQuantizedVectorValues(FIELD));
        if (Version.LATEST.minor >= 4) {
          assertNull(
              "Lucene104 quantization is not detected through the legacy scalar state",
              quantized.getQuantizationState(FIELD));
        }
        IndexSearcher searcher = new IndexSearcher(reader);
        var query =
            new BaseLayerKnnFloatVectorQuery(FIELD, new float[] {SAMPLE_SELF_MATCH_ORDINAL}, 5);
        assertFalse(
            collector(query, searcher, leaf, Integer.MAX_VALUE).getSearchStrategy()
                instanceof KnnSearchStrategy.Seeded);
        assertSameHits(
            searcher.search(
                new KnnFloatVectorQuery(FIELD, new float[] {SAMPLE_SELF_MATCH_ORDINAL}, 5), 5),
            searcher.search(query, 5));
      }
    }
  }

  private static KnnCollector collector(
      BaseLayerKnnFloatVectorQuery query,
      IndexSearcher searcher,
      LeafReaderContext leaf,
      int visitedLimit)
      throws Exception {
    return query
        .getKnnCollectorManager(query.getK(), searcher)
        .newCollector(visitedLimit, KnnSearchStrategy.Hnsw.DEFAULT, leaf);
  }

  private static KnnSearchStrategy.Seeded seeded(KnnCollector collector) {
    assertTrue(
        "single-layer search must use its selected entry",
        collector.getSearchStrategy() instanceof KnnSearchStrategy.Seeded);
    return (KnnSearchStrategy.Seeded) collector.getSearchStrategy();
  }

  private static HnswGraph graphOf(LeafReader leaf) throws Exception {
    return ((HnswGraphProvider) vectorsReaderOf(leaf)).getGraph(FIELD);
  }

  private static KnnVectorsReader vectorsReaderOf(LeafReader leaf) {
    KnnVectorsReader reader = ((CodecReader) leaf).getVectorReader();
    if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      reader = fieldsReader.getFieldReader(FIELD);
    }
    return reader;
  }

  private static void assertSameHits(TopDocs expected, TopDocs actual) {
    assertEquals(expected.scoreDocs.length, actual.scoreDocs.length);
    for (int i = 0; i < expected.scoreDocs.length; i++) {
      assertEquals(expected.scoreDocs[i].doc, actual.scoreDocs[i].doc);
      assertEquals(expected.scoreDocs[i].score, actual.scoreDocs[i].score, 0f);
    }
  }

  private static void writeIndex(
      Directory directory, int maxConn, int segments, boolean sparse, boolean delete)
      throws Exception {
    writeIndex(directory, maxConn, segments, sparse, delete, VECTOR_COUNT);
  }

  private static void writeIndex(
      Directory directory,
      int maxConn,
      int segments,
      boolean sparse,
      boolean delete,
      int vectorCount)
      throws Exception {
    KnnVectorsFormat format =
        Version.LATEST.minor >= 4
            // Lucene 10.4 otherwise omits the graph for these small fixtures.
            ? Lucene99HnswVectorsFormat.class
                .getConstructor(int.class, int.class, int.class)
                .newInstance(maxConn, BEAM_WIDTH, 0)
            : new Lucene99HnswVectorsFormat(maxConn, BEAM_WIDTH);
    writeIndex(directory, format, segments, sparse, delete, vectorCount);
  }

  private static void writeIndex(
      Directory directory,
      KnnVectorsFormat format,
      int segments,
      boolean sparse,
      boolean delete,
      int vectorCount)
      throws Exception {
    long previousSeed = HnswGraphBuilder.randSeed;
    try {
      // The first 64 level draws for seed 0 and M=512 are all zero. The assertion after reopening
      // protects the test from a change in Lucene's topology generation; M=2 supplies its control.
      HnswGraphBuilder.randSeed = 0;
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setMergePolicy(NoMergePolicy.INSTANCE)
              .setCodec(TestUtil.alwaysKnnVectorsFormat(format));
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int segment = 0; segment < segments; segment++) {
          for (int ordinal = 0; ordinal < vectorCount; ordinal++) {
            if (sparse) {
              Document missingVector = new Document();
              missingVector.add(new StringField("gap", "yes", Field.Store.NO));
              writer.addDocument(missingVector);
            }
            int id = SEGMENT_ID_STRIDE * segment + ordinal;
            Document document = new Document();
            document.add(new StringField("id", Integer.toString(id), Field.Store.YES));
            document.add(
                new StringField("parity", ordinal % 2 == 0 ? "even" : "odd", Field.Store.NO));
            document.add(new KnnFloatVectorField(FIELD, new float[] {id}, EUCLIDEAN));
            writer.addDocument(document);
          }
          writer.commit();
        }
        if (delete) {
          for (int segment = 0; segment < segments; segment++) {
            writer.deleteDocuments(
                new Term(
                    "id", Integer.toString(SEGMENT_ID_STRIDE * segment + DELETED_VECTOR_ORDINAL)));
          }
        }
      }
    } finally {
      HnswGraphBuilder.randSeed = previousSeed;
    }
  }

  private static KnnVectorsFormat scalarQuantizedFormat() throws Exception {
    if (Version.LATEST.minor < 4) {
      return (KnnVectorsFormat)
          Class.forName(
                  "org.apache.lucene.codecs.lucene99.Lucene99HnswScalarQuantizedVectorsFormat")
              .getConstructor(int.class, int.class)
              .newInstance(SINGLE_LAYER_MAX_CONN, BEAM_WIDTH);
    }
    Class<?> encoding =
        Class.forName(
            "org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat$ScalarEncoding");
    return (KnnVectorsFormat)
        Class.forName(
                "org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat")
            .getConstructor(
                encoding, int.class, int.class, int.class, ExecutorService.class, int.class)
            .newInstance(
                encoding.getField("UNSIGNED_BYTE").get(null),
                SINGLE_LAYER_MAX_CONN,
                BEAM_WIDTH,
                1,
                null,
                0);
  }
}
