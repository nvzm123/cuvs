/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/** End-to-end coverage for the opt-in post-ingest flush overlap. */
@SuppressSysoutChecks(bugUrl = "")
public class TestPostIngestOverlapPersistedIndex extends LuceneTestCase {

  private static final String OVERLAP_PROPERTY = "cuvs.lucene.experimentalPostIngestOverlap";
  private static final String ID_FIELD = "id";
  private static final String SORT_FIELD = "sort_value";
  private static final String FIRST_VECTOR_FIELD = "first_vector";
  private static final String SECOND_VECTOR_FIELD = "second_vector";
  private static final int DOCUMENT_COUNT = 192;
  private static final int DIMENSIONS = 32;

  private String originalOverlapProperty;

  @BeforeClass
  public static void checkCuVSSupport() {
    assumeTrue("cuVS is not supported", isSupported());
  }

  @Before
  public void rememberOverlapProperty() {
    originalOverlapProperty = System.getProperty(OVERLAP_PROPERTY);
  }

  @After
  public void restoreOverlapProperty() {
    restoreProperty(originalOverlapProperty);
  }

  @Test
  public void testSortedSparseMultiFieldRoundTripMatchesSerialFlush() throws Exception {
    Corpus corpus = createCorpus();

    try (Directory serialDirectory = newDirectory();
        Directory overlapDirectory = newDirectory()) {
      buildIndex(serialDirectory, corpus, false);
      assertEquals(originalOverlapProperty, System.getProperty(OVERLAP_PROPERTY));
      buildIndex(overlapDirectory, corpus, true);
      assertEquals(originalOverlapProperty, System.getProperty(OVERLAP_PROPERTY));

      TestUtil.checkIndex(serialDirectory);
      TestUtil.checkIndex(overlapDirectory);

      IndexSnapshot serial = readAndValidate(serialDirectory, corpus);
      IndexSnapshot overlapped = readAndValidate(overlapDirectory, corpus);
      assertSnapshotsEqual(serial, overlapped);
    }
  }

  private static Corpus createCorpus() {
    Random random = new Random(0x2653L);
    List<TestDocument> documents = new ArrayList<>(DOCUMENT_COUNT);
    Map<String, float[]> expectedFirstVectors = new HashMap<>();
    Map<String, float[]> expectedSecondVectors = new HashMap<>();

    for (int id = 0; id < DOCUMENT_COUNT; id++) {
      String storedId = Integer.toString(id);
      long sortValue = (id * 37L) % DOCUMENT_COUNT;
      float[] firstVector = id % 3 == 0 ? null : randomVector(random);
      float[] secondVector = id % 4 == 1 ? null : randomVector(random);
      documents.add(new TestDocument(storedId, sortValue, firstVector, secondVector));
      if (firstVector != null) {
        expectedFirstVectors.put(storedId, firstVector);
      }
      if (secondVector != null) {
        expectedSecondVectors.put(storedId, secondVector);
      }
    }
    return new Corpus(documents, expectedFirstVectors, expectedSecondVectors);
  }

  private static float[] randomVector(Random random) {
    float[] vector = new float[DIMENSIONS];
    for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
      vector[dimension] = random.nextFloat();
    }
    return vector;
  }

  private static void buildIndex(Directory directory, Corpus corpus, boolean overlap)
      throws Exception {
    String previousProperty = System.getProperty(OVERLAP_PROPERTY);
    System.setProperty(OVERLAP_PROPERTY, Boolean.toString(overlap));
    try {
      AcceleratedHNSWParams params =
          new AcceleratedHNSWParams.Builder()
              .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
              .withIntermediateGraphDegree(32)
              .withGraphDegree(16)
              .withHNSWLayer(1)
              .build();
      Codec codec = new Lucene101AcceleratedHNSWCodec(params);
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(codec)
              .setUseCompoundFile(false)
              .setMergePolicy(NoMergePolicy.INSTANCE)
              .setIndexSort(new Sort(new SortField(SORT_FIELD, SortField.Type.LONG)))
              .setMaxBufferedDocs(DOCUMENT_COUNT + 1)
              .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (TestDocument testDocument : corpus.documents()) {
          Document document = new Document();
          document.add(new StringField(ID_FIELD, testDocument.id(), Field.Store.YES));
          document.add(new NumericDocValuesField(SORT_FIELD, testDocument.sortValue()));
          document.add(new StoredField(SORT_FIELD, testDocument.sortValue()));
          if (testDocument.firstVector() != null) {
            document.add(
                new KnnFloatVectorField(FIRST_VECTOR_FIELD, testDocument.firstVector(), EUCLIDEAN));
          }
          if (testDocument.secondVector() != null) {
            document.add(
                new KnnFloatVectorField(
                    SECOND_VECTOR_FIELD, testDocument.secondVector(), EUCLIDEAN));
          }
          writer.addDocument(document);
        }
        writer.commit();
      }
    } finally {
      restoreProperty(previousProperty);
    }
  }

  private static IndexSnapshot readAndValidate(Directory directory, Corpus corpus)
      throws Exception {
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals(DOCUMENT_COUNT, reader.numDocs());
      assertEquals(1, reader.leaves().size());
      LeafReader leaf = getOnlyLeafReader(reader);

      List<String> idsInIndexOrder = assertIndexSortOrder(leaf);
      Map<String, float[]> firstVectors =
          readAndValidateVectors(leaf, FIRST_VECTOR_FIELD, corpus.expectedFirstVectors());
      Map<String, float[]> secondVectors =
          readAndValidateVectors(leaf, SECOND_VECTOR_FIELD, corpus.expectedSecondVectors());
      assertGraphSize(leaf, FIRST_VECTOR_FIELD, firstVectors.size());
      assertGraphSize(leaf, SECOND_VECTOR_FIELD, secondVectors.size());

      IndexSearcher searcher = new IndexSearcher(reader);
      assertSelfSearchFindsDocument(
          searcher, FIRST_VECTOR_FIELD, corpus.expectedFirstVectors().get("1"), "1");
      assertSelfSearchFindsDocument(
          searcher, SECOND_VECTOR_FIELD, corpus.expectedSecondVectors().get("0"), "0");
      return new IndexSnapshot(idsInIndexOrder, firstVectors, secondVectors);
    }
  }

  private static List<String> assertIndexSortOrder(LeafReader leaf) throws IOException {
    List<String> ids = new ArrayList<>(leaf.maxDoc());
    long previousSortValue = Long.MIN_VALUE;
    for (int docId = 0; docId < leaf.maxDoc(); docId++) {
      Document document = leaf.storedFields().document(docId);
      long sortValue = document.getField(SORT_FIELD).numericValue().longValue();
      assertTrue("index sort order went backwards", sortValue >= previousSortValue);
      previousSortValue = sortValue;
      ids.add(document.get(ID_FIELD));
    }
    return ids;
  }

  private static Map<String, float[]> readAndValidateVectors(
      LeafReader leaf, String field, Map<String, float[]> expected) throws IOException {
    FloatVectorValues values = leaf.getFloatVectorValues(field);
    assertNotNull("no vector values for field " + field, values);
    assertEquals(expected.size(), values.size());

    Map<String, float[]> actual = new HashMap<>();
    int previousDoc = -1;
    for (int ordinal = 0; ordinal < values.size(); ordinal++) {
      int docId = values.ordToDoc(ordinal);
      assertTrue("vector doc IDs went backwards for field " + field, docId > previousDoc);
      previousDoc = docId;
      String id = leaf.storedFields().document(docId).get(ID_FIELD);
      float[] vector = values.vectorValue(ordinal).clone();
      assertNotNull(
          "unexpected vector for document " + id + " in field " + field, expected.get(id));
      assertArrayEquals(expected.get(id), vector, 0.0f);
      assertNull(
          "duplicate vector for document " + id + " in field " + field, actual.put(id, vector));
    }
    assertEquals(expected.keySet(), actual.keySet());
    return actual;
  }

  private static void assertGraphSize(LeafReader leaf, String field, int expectedSize)
      throws Exception {
    KnnVectorsReader vectorReader = ((CodecReader) leaf).getVectorReader();
    if (vectorReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      vectorReader = fieldsReader.getFieldReader(field);
    }
    HnswGraph graph = ((HnswGraphProvider) vectorReader).getGraph(field);
    assertNotNull(graph);
    assertEquals(expectedSize, graph.size());
  }

  private static void assertSelfSearchFindsDocument(
      IndexSearcher searcher, String field, float[] queryVector, String expectedId)
      throws IOException {
    var hits = searcher.search(new KnnFloatVectorQuery(field, queryVector, 10), 10);
    assertEquals(10, hits.scoreDocs.length);
    Set<String> ids = new HashSet<>();
    for (var hit : hits.scoreDocs) {
      ids.add(searcher.storedFields().document(hit.doc).get(ID_FIELD));
    }
    assertTrue(
        "self-search did not return document " + expectedId + " for field " + field,
        ids.contains(expectedId));
  }

  private static void assertSnapshotsEqual(IndexSnapshot expected, IndexSnapshot actual) {
    assertEquals(expected.idsInIndexOrder(), actual.idsInIndexOrder());
    assertVectorMapsEqual(expected.firstVectors(), actual.firstVectors());
    assertVectorMapsEqual(expected.secondVectors(), actual.secondVectors());
  }

  private static void assertVectorMapsEqual(
      Map<String, float[]> expected, Map<String, float[]> actual) {
    assertEquals(expected.keySet(), actual.keySet());
    for (String id : expected.keySet()) {
      assertArrayEquals(
          "vector differs for document " + id, expected.get(id), actual.get(id), 0.0f);
    }
  }

  private static void restoreProperty(String value) {
    if (value == null) {
      System.clearProperty(OVERLAP_PROPERTY);
    } else {
      System.setProperty(OVERLAP_PROPERTY, value);
    }
  }

  private record TestDocument(
      String id, long sortValue, float[] firstVector, float[] secondVector) {}

  private record Corpus(
      List<TestDocument> documents,
      Map<String, float[]> expectedFirstVectors,
      Map<String, float[]> expectedSecondVectors) {}

  private record IndexSnapshot(
      List<String> idsInIndexOrder,
      Map<String, float[]> firstVectors,
      Map<String, float[]> secondVectors) {}
}
