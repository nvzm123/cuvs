/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
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
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.junit.Test;

/**
 * Repro for the CI-observed {@code EOFException} in {@code OffHeapFloatVectorValues} during
 * concurrent KNN search over an accelerated-HNSW index after deletions and a real merge.
 *
 * <p>Rather than relying on a random concurrent search happening to traverse a bad graph node
 * (which only reproduced intermittently, on one CI node), this walks the <em>entire</em> merged
 * HNSW graph directly and asserts every neighbor ordinal is within the merged segment's actual
 * flat-vector count. This targets the root cause: the merged view's {@code size()} is the sum of
 * the source segments' on-disk vector counts, while its iterator yields only live vectors. Using
 * {@code size()} to allocate the CAGRA input produces phantom graph nodes whenever the merge drops
 * deleted documents, even though the authoritative flat {@code .vec} file contains only the live
 * vectors.
 *
 * <p>These tests are deterministic: they check the graph's full ordinal domain and the zero- and
 * one-live-vector merge boundaries rather than depending on a search happening to reach a phantom
 * node.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestMergedGraphOrdinalBounds extends LuceneTestCase {

  private static final String ID_FIELD = "id";
  private static final String FIELD = "vector";

  @Test
  public void testMergedGraphOrdinalsStayWithinFlatVectorBounds() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    Random random = new Random(1234);
    int segmentSize = 300;
    int dimension = 32;
    // Interspersed deletions on both segments, so the merge must drop a scattered subset of
    // ordinals from each -- not just a contiguous prefix/suffix -- when it re-derives the merged
    // vector set.
    int deleteEveryNth = 4;

    Codec codec =
        TestUtil.alwaysKnnVectorsFormat(
            new Lucene99AcceleratedHNSWVectorsFormat(
                new AcceleratedHNSWParams.Builder().withWriterThreads(4).build()));
    IndexWriterConfig config =
        new IndexWriterConfig().setCodec(codec).setMergePolicy(NoMergePolicy.INSTANCE);

    Map<Integer, float[]> expectedLiveVectorsById = new HashMap<>();
    int expectedLiveVectors;
    try (Directory dir = newDirectory();
        IndexWriter writer = new IndexWriter(dir, config)) {
      int deletedFromSegment1 =
          addSegmentWithInterspersedDeletions(
              writer, 0, segmentSize, dimension, deleteEveryNth, random, expectedLiveVectorsById);
      writer.commit(); // segment 1, alone
      int deletedFromSegment2 =
          addSegmentWithInterspersedDeletions(
              writer,
              segmentSize,
              segmentSize,
              dimension,
              deleteEveryNth,
              random,
              expectedLiveVectorsById);
      writer.commit(); // segment 2, alone

      expectedLiveVectors = 2 * segmentSize - deletedFromSegment1 - deletedFromSegment2;
      assertEquals(expectedLiveVectors, expectedLiveVectorsById.size());

      try (DirectoryReader sourceReader = DirectoryReader.open(writer)) {
        assertEquals("the repro requires two source segments", 2, sourceReader.leaves().size());
        for (var context : sourceReader.leaves()) {
          LeafReader sourceLeaf = context.reader();
          assertTrue("each source segment must carry deletions", sourceLeaf.hasDeletions());
          assertEquals(segmentSize, sourceLeaf.maxDoc());
          assertEquals(segmentSize, sourceLeaf.getFloatVectorValues(FIELD).size());
        }
      }

      // NoMergePolicy blocks forced merges too, so swap it out now that the two segments (each
      // with their own interspersed deletions already committed) are set up.
      writer.getConfig().setMergePolicy(new TieredMergePolicy());
      writer.forceMerge(1);
      writer.commit();

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals(
            "expected the forced merge to produce a single segment", 1, reader.leaves().size());
        LeafReader leaf = reader.leaves().get(0).reader();

        FloatVectorValues flatValues = leaf.getFloatVectorValues(FIELD);
        assertEquals(
            "merged flat vector count should equal (added - deleted)",
            expectedLiveVectors,
            flatValues.size());

        KnnVectorValues.DocIndexIterator vectors = flatValues.iterator();
        int previousId = -1;
        int seen = 0;
        for (int doc = vectors.nextDoc(); doc != NO_MORE_DOCS; doc = vectors.nextDoc()) {
          int id = Integer.parseInt(leaf.storedFields().document(doc).get(ID_FIELD));
          assertTrue("merged vectors must retain source document order", id > previousId);
          float[] expected = expectedLiveVectorsById.get(id);
          assertNotNull("unexpected live vector id " + id, expected);
          assertArrayEquals(expected, flatValues.vectorValue(vectors.index()), 0.0f);
          previousId = id;
          seen++;
        }
        assertEquals(expectedLiveVectors, seen);

        HnswGraph graph = graphOf(leaf);
        int level0NodeCount = graph.getNodesOnLevel(0).size();
        assertEquals(
            "HNSW graph's level-0 node count disagrees with the merged flat vector file's actual"
                + " live-vector count",
            flatValues.size(),
            level0NodeCount);

        assertAllNeighborOrdinalsInBounds(graph, flatValues.size());
      }
    }
  }

  @Test
  public void testForcedMergeWithExactlyOneLiveVector() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    float[] survivorVector = new float[] {1.0f, 2.0f, 3.0f, 4.0f};
    Codec codec = TestUtil.alwaysKnnVectorsFormat(new Lucene99AcceleratedHNSWVectorsFormat());
    IndexWriterConfig config =
        new IndexWriterConfig().setCodec(codec).setMergePolicy(NoMergePolicy.INSTANCE);

    try (Directory dir = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(dir, config)) {
        addVectorDocument(writer, "survivor", survivorVector);
        addNonVectorDocument(writer, "sentinel-1");
        writer.commit();

        addVectorDocument(writer, "deleted", new float[] {4.0f, 3.0f, 2.0f, 1.0f});
        addNonVectorDocument(writer, "sentinel-2");
        writer.commit();

        writer.deleteDocuments(new Term(ID_FIELD, "deleted"));
        writer.commit();
        writer.getConfig().setMergePolicy(new TieredMergePolicy());
        writer.forceMerge(1);
        writer.commit();
      }

      // Open only after the writer is closed so this exercises the persisted merged output.
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals(
            "expected the forced merge to produce a single segment", 1, reader.leaves().size());
        assertEquals("one vector doc and two sentinels should remain", 3, reader.numDocs());
        LeafReader leaf = reader.leaves().get(0).reader();

        FloatVectorValues flatValues = leaf.getFloatVectorValues(FIELD);
        assertNotNull(flatValues);
        assertEquals(
            "the merged flat vector file must contain only the survivor", 1, flatValues.size());
        assertArrayEquals(survivorVector, flatValues.vectorValue(0), 0.0f);
        assertEquals(
            "survivor", leaf.storedFields().document(flatValues.ordToDoc(0)).get(ID_FIELD));

        HnswGraph graph = graphOf(leaf);
        assertEquals("the single-vector graph must have one level-0 node", 1, graph.size());
        assertEquals(1, graph.numLevels());
        assertEquals(1, graph.getNodesOnLevel(0).size());
        assertAllNeighborOrdinalsInBounds(graph, flatValues.size());
        graph.seek(0, 0);
        assertEquals(
            "a single-node graph must not have an edge", NO_MORE_DOCS, graph.nextNeighbor());

        ((CodecReader) leaf).getVectorReader().checkIntegrity();

        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs results = searcher.search(new KnnFloatVectorQuery(FIELD, survivorVector, 1), 1);
        assertEquals("the only live vector must be searchable", 1, results.totalHits.value());
        assertEquals(1, results.scoreDocs.length);
        assertEquals(
            "survivor", searcher.storedFields().document(results.scoreDocs[0].doc).get(ID_FIELD));
      }
    }
  }

  @Test
  public void testForcedMergeWithZeroLiveVectors() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    float[] queryVector = new float[] {1.0f, 2.0f, 3.0f, 4.0f};
    Codec codec = TestUtil.alwaysKnnVectorsFormat(new Lucene99AcceleratedHNSWVectorsFormat());
    IndexWriterConfig config =
        new IndexWriterConfig().setCodec(codec).setMergePolicy(NoMergePolicy.INSTANCE);

    try (Directory dir = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(dir, config)) {
        addVectorDocument(writer, "deleted-1", queryVector);
        addNonVectorDocument(writer, "sentinel-1");
        writer.commit();

        addVectorDocument(writer, "deleted-2", new float[] {4.0f, 3.0f, 2.0f, 1.0f});
        addNonVectorDocument(writer, "sentinel-2");
        writer.commit();

        writer.deleteDocuments(new Term(ID_FIELD, "deleted-1"));
        writer.deleteDocuments(new Term(ID_FIELD, "deleted-2"));
        writer.commit();
        writer.getConfig().setMergePolicy(new TieredMergePolicy());
        writer.forceMerge(1);
        writer.commit();
      }

      // The sentinels keep a real merged segment on disk after every vector document is deleted.
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals(
            "expected the forced merge to produce a single segment", 1, reader.leaves().size());
        assertEquals("only the non-vector sentinels should remain", 2, reader.numDocs());
        LeafReader leaf = reader.leaves().get(0).reader();

        FloatVectorValues flatValues = leaf.getFloatVectorValues(FIELD);
        assertNotNull(flatValues);
        assertEquals("the merged flat vector file must be empty", 0, flatValues.size());
        assertEquals(NO_MORE_DOCS, flatValues.iterator().nextDoc());

        HnswGraph graph = graphOf(leaf);
        assertEquals("an empty vector field must expose an empty graph", 0, graph.size());
        assertEquals(0, graph.numLevels());
        assertEquals(0, graph.getNodesOnLevel(0).size());

        ((CodecReader) leaf).getVectorReader().checkIntegrity();

        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs results = searcher.search(new KnnFloatVectorQuery(FIELD, queryVector, 1), 1);
        assertEquals(
            "an empty vector field must return no nearest neighbors", 0, results.totalHits.value());
        assertEquals(0, results.scoreDocs.length);
      }
    }
  }

  /**
   * Every neighbor referenced anywhere in the graph, at every level, must be a valid ordinal into
   * the merged segment's actual flat vector data -- otherwise a reader resolving that neighbor's
   * vector (e.g. mid-search, to score it) reads past the end of the flat file.
   */
  private static void assertAllNeighborOrdinalsInBounds(HnswGraph graph, int liveVectorCount)
      throws Exception {
    for (int level = 0; level < graph.numLevels(); level++) {
      HnswGraph.NodesIterator nodes = graph.getNodesOnLevel(level);
      while (nodes.hasNext()) {
        int node = nodes.nextInt();
        assertTrue(
            "node "
                + node
                + " at level "
                + level
                + " is itself out of bounds (live vectors: "
                + liveVectorCount
                + ")",
            node >= 0 && node < liveVectorCount);
        graph.seek(level, node);
        for (int neighbor = graph.nextNeighbor();
            neighbor != NO_MORE_DOCS;
            neighbor = graph.nextNeighbor()) {
          assertTrue(
              "node "
                  + node
                  + " at level "
                  + level
                  + " has a neighbor ordinal "
                  + neighbor
                  + " out of bounds for the merged segment's "
                  + liveVectorCount
                  + " live vectors",
              neighbor >= 0 && neighbor < liveVectorCount);
        }
      }
    }
  }

  private static HnswGraph graphOf(LeafReader leaf) throws Exception {
    KnnVectorsReader knnReader = ((CodecReader) leaf).getVectorReader();
    if (knnReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      knnReader = fieldsReader.getFieldReader(FIELD);
    }
    return ((HnswGraphProvider) knnReader).getGraph(FIELD);
  }

  private static void addVectorDocument(IndexWriter writer, String id, float[] vector)
      throws Exception {
    Document document = new Document();
    document.add(new StringField(ID_FIELD, id, Field.Store.YES));
    document.add(new KnnFloatVectorField(FIELD, vector, EUCLIDEAN));
    writer.addDocument(document);
  }

  private static void addNonVectorDocument(IndexWriter writer, String id) throws Exception {
    Document document = new Document();
    document.add(new StringField(ID_FIELD, id, Field.Store.YES));
    writer.addDocument(document);
  }

  /**
   * Adds {@code count} documents (global ids {@code [startId, startId + count)}), then deletes
   * every {@code deleteEveryNth}-th one by id, scattering the deletions across the segment rather
   * than leaving a contiguous surviving range.
   *
   * @return the number of documents deleted from this segment
   */
  private static int addSegmentWithInterspersedDeletions(
      IndexWriter writer,
      int startId,
      int count,
      int dimension,
      int deleteEveryNth,
      Random random,
      Map<Integer, float[]> expectedLiveVectorsById)
      throws Exception {
    float[][] dataset = generateDataset(random, count, dimension);
    for (int i = 0; i < count; i++) {
      int id = startId + i;
      Document document = new Document();
      document.add(new StringField(ID_FIELD, Integer.toString(id), Field.Store.YES));
      document.add(new KnnFloatVectorField(FIELD, dataset[i], EUCLIDEAN));
      writer.addDocument(document);
      if (i % deleteEveryNth != 0) {
        expectedLiveVectorsById.put(id, dataset[i]);
      }
    }
    int deleted = 0;
    for (int i = 0; i < count; i += deleteEveryNth) {
      writer.deleteDocuments(new Term(ID_FIELD, Integer.toString(startId + i)));
      deleted++;
    }
    return deleted;
  }
}
