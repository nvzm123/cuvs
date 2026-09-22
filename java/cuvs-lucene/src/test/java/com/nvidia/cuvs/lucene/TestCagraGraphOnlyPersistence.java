/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_INDEX_EXT;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_META_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType.CAGRA_AND_BRUTE_FORCE;
import static com.nvidia.cuvs.lucene.GPUSearchParams.CagraPersistenceMode.GRAPH_ONLY;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CuVSResources;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.InfoStream;
import org.junit.After;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCagraGraphOnlyPersistence extends LuceneTestCase {

  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector";

  @BeforeClass
  public static void requireGpu() {
    assumeTrue("Requires a GPU", isSupported());
  }

  @After
  public void closeQueryResources() {
    closeCuVSResourcesInstance();
  }

  @Test
  public void testReopenAtAlignedDimension() throws Exception {
    assertIndexReopens(96, graphOnlyParams());
  }

  @Test
  public void testReopenAtUnalignedDimension() throws Exception {
    assertIndexReopens(95, graphOnlyParams());
  }

  @Test
  public void testLegacyV0IndexStillReopens() throws Exception {
    int dimension = 96;
    float[][] vectors = vectors(64, dimension);
    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      writeIndex(directory, vectors, new GPUSearchParams.Builder().build(), infoStream);
      assertLegacyV0MetadataShape(directory, dimension, vectors.length);
      assertNoCagraBuildFallback(infoStream);

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertTop1Identity(newSearcher(reader), vectors, 0, 17, 63);
      }
    }
  }

  @Test
  public void testGraphOnlyMatchesLegacyV0Results() throws Exception {
    float[][] vectors = vectors(64, 96);
    RecordingInfoStream legacyInfo = new RecordingInfoStream();
    RecordingInfoStream graphOnlyInfo = new RecordingInfoStream();
    try (Directory legacyDirectory = newDirectory();
        Directory graphOnlyDirectory = newDirectory()) {
      writeIndex(legacyDirectory, vectors, new GPUSearchParams.Builder().build(), legacyInfo);
      writeIndex(graphOnlyDirectory, vectors, graphOnlyParams(), graphOnlyInfo);
      assertNoCagraBuildFallback(legacyInfo);
      assertNoCagraBuildFallback(graphOnlyInfo);

      try (DirectoryReader legacyReader = DirectoryReader.open(legacyDirectory);
          DirectoryReader graphOnlyReader = DirectoryReader.open(graphOnlyDirectory)) {
        IndexSearcher legacySearcher = newSearcher(legacyReader);
        IndexSearcher graphOnlySearcher = newSearcher(graphOnlyReader);
        for (int id : new int[] {4, 17, 42, 63}) {
          assertEquals(
              "graph-only result ids differ for query " + id,
              resultIds(legacySearcher, vectors[id], 10),
              resultIds(graphOnlySearcher, vectors[id], 10));
        }
      }
    }
  }

  @Test
  public void testConcurrentGraphOnlyReadersHaveIndependentResourceLifetimes() throws Exception {
    float[][] firstVectors = vectors(64, 95);
    float[][] secondVectors = vectors(64, 95);
    RecordingInfoStream firstInfo = new RecordingInfoStream();
    RecordingInfoStream secondInfo = new RecordingInfoStream();
    try (Directory firstDirectory = newDirectory();
        Directory secondDirectory = newDirectory()) {
      writeIndex(firstDirectory, firstVectors, graphOnlyParams(), firstInfo);
      writeIndex(secondDirectory, secondVectors, graphOnlyParams(), secondInfo);
      assertNoCagraBuildFallback(firstInfo);
      assertNoCagraBuildFallback(secondInfo);

      DirectoryReader firstReader = DirectoryReader.open(firstDirectory);
      DirectoryReader secondReader = DirectoryReader.open(secondDirectory);
      try {
        assertTop1Identity(newSearcher(firstReader), firstVectors, 0, 17, 63);
        assertTop1Identity(newSearcher(secondReader), secondVectors, 0, 17, 63);

        firstReader.close();
        firstReader = null;

        assertTop1Identity(newSearcher(secondReader), secondVectors, 0, 17, 63);
      } finally {
        if (firstReader != null) {
          firstReader.close();
        }
        secondReader.close();
      }
    }
  }

  @Test
  public void testReaderCloseDoesNotCloseOrReplaceCallerOwnedQueryResources() throws Exception {
    float[][] vectors = vectors(64, 95);
    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      writeIndex(directory, vectors, graphOnlyParams(), infoStream);
      assertNoCagraBuildFallback(infoStream);

      TrackingCuVSResources callerResources =
          new TrackingCuVSResources(
              ThreadLocalCuVSResourcesProvider.createIndependentCuVSResourcesInstance());
      assertNotNull(callerResources.delegate);
      setCuVSResourcesInstance(callerResources);
      try {
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
          assertTop1Identity(newSearcher(reader), vectors, 0, 17, 63);
        }

        assertSame(callerResources, getCuVSResourcesInstance());
        assertEquals(0, callerResources.closeCalls);
        try (CuVSResources.ScopedAccess ignored = callerResources.access()) {
          // Successfully acquiring the native handle proves the reader did not invalidate it.
        }
      } finally {
        CuVSResources currentResources = getCuVSResourcesInstance();
        closeCuVSResourcesInstance();
        if (currentResources != callerResources) {
          callerResources.close();
        }
      }
    }
  }

  @Test
  public void testCustomReaderResourcesFactoryOwnsReaderLifetime() throws Exception {
    float[][] vectors = vectors(64, 95);
    Path tempDirectory = createTempDir("custom-reader-resources");
    List<TrackingCuVSResources> createdResources = new ArrayList<>();
    CuVSReaderResourcesFactory resourcesFactory =
        () -> {
          TrackingCuVSResources resources =
              new TrackingCuVSResources(CuVSResources.create(tempDirectory));
          createdResources.add(resources);
          return resources;
        };
    CuVS2510GPUSearchCodec codec =
        new CuVS2510GPUSearchCodec(
            graphOnlyParams(), FilterBitsetCacheConfig.DEFAULT, resourcesFactory);

    try (Directory directory = newDirectory();
        IndexWriter writer =
            new IndexWriter(
                directory,
                new IndexWriterConfig()
                    .setUseCompoundFile(false)
                    .setMergePolicy(NoMergePolicy.INSTANCE)
                    .setCodec(codec))) {
      for (int id = 0; id < vectors.length; id++) {
        writer.addDocument(document(id, vectors[id]));
      }
      writer.commit();
      assertEquals("writing must not invoke the reader factory", 0, createdResources.size());

      DirectoryReader reader = DirectoryReader.open(writer);
      try {
        assertEquals(1, reader.leaves().size());
        assertEquals(
            "one retained reader must own one resources instance", 1, createdResources.size());
        TrackingCuVSResources readerResources = createdResources.getFirst();
        assertEquals(0, readerResources.closeCalls);
        assertTrue("reader did not use its custom resources", readerResources.accessCalls > 0);
        assertTrue(
            "reader deserialization did not use the custom temporary directory",
            readerResources.tempDirectoryCalls > 0);
        assertEquals(tempDirectory, readerResources.tempDirectory());
        assertTop1Identity(newSearcher(reader), vectors, 0, 17, 63);
      } finally {
        reader.close();
        reader.close();
      }
    }
    assertEquals(1, createdResources.getFirst().closeCalls);
  }

  @Test
  public void testReaderResourcesFactoryPreservesCheckedFailure() throws Exception {
    float[][] vectors = vectors(64, 95);
    IOException factoryFailure = new IOException("reader resources unavailable");
    CuVSReaderResourcesFactory resourcesFactory =
        () -> {
          throw factoryFailure;
        };

    try (Directory directory = newDirectory();
        IndexWriter writer =
            new IndexWriter(directory, writerConfig(graphOnlyParams(), resourcesFactory))) {
      for (int id = 0; id < vectors.length; id++) {
        writer.addDocument(document(id, vectors[id]));
      }
      writer.commit();

      IOException thrown = expectThrows(IOException.class, () -> DirectoryReader.open(writer));
      assertSame(factoryFailure, thrown);
    }
  }

  @Test
  public void testReaderResourcesFactoryRejectsNullResult() throws Exception {
    float[][] vectors = vectors(64, 95);
    try (Directory directory = newDirectory();
        IndexWriter writer =
            new IndexWriter(directory, writerConfig(graphOnlyParams(), () -> null))) {
      for (int id = 0; id < vectors.length; id++) {
        writer.addDocument(document(id, vectors[id]));
      }
      writer.commit();

      IllegalStateException thrown =
          expectThrows(IllegalStateException.class, () -> DirectoryReader.open(writer));
      assertEquals("readerResourcesFactory returned null", thrown.getMessage());
    }
  }

  @Test
  public void testReaderConstructionFailureClosesFactoryResources() throws Throwable {
    float[][] vectors = vectors(64, 95);
    RuntimeException loadFailure = new RuntimeException("native index load failed");
    TrackingCuVSResources readerResources =
        new TrackingCuVSResources(CuVSResources.create(), loadFailure);

    try (Directory directory = newDirectory();
        IndexWriter writer =
            new IndexWriter(directory, writerConfig(graphOnlyParams(), () -> readerResources))) {
      for (int id = 0; id < vectors.length; id++) {
        writer.addDocument(document(id, vectors[id]));
      }
      writer.commit();

      RuntimeException thrown =
          expectThrows(RuntimeException.class, () -> DirectoryReader.open(writer));
      assertSame(loadFailure, thrown);
      assertTrue(readerResources.accessCalls > 0);
      assertEquals(1, readerResources.closeCalls);
    }
  }

  @Test
  public void testMergeReadersDoNotInvokeReaderResourcesFactory() throws Exception {
    float[][] vectors = vectors(128, 95);
    AtomicInteger factoryCalls = new AtomicInteger();
    CuVSReaderResourcesFactory resourcesFactory =
        () -> {
          factoryCalls.incrementAndGet();
          throw new AssertionError("merge reader invoked retained-reader resources factory");
        };

    try (Directory directory = newDirectory();
        IndexWriter writer =
            new IndexWriter(
                directory,
                writerConfig(graphOnlyParams(), resourcesFactory)
                    .setMergePolicy(NoMergePolicy.INSTANCE))) {
      for (int id = 0; id < vectors.length; id++) {
        writer.addDocument(document(id, vectors[id]));
        if (id == 63) {
          writer.commit();
        }
      }
      writer.commit();
      writer.getConfig().setMergePolicy(new TieredMergePolicy());
      writer.forceMerge(1);
    }
    assertEquals(0, factoryCalls.get());
  }

  @Test
  public void testGraphOnlyReaderCanCloseFromAnotherThread() throws Exception {
    float[][] vectors = vectors(64, 95);
    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      writeIndex(directory, vectors, graphOnlyParams(), infoStream);
      assertNoCagraBuildFallback(infoStream);

      DirectoryReader reader = DirectoryReader.open(directory);
      assertTop1Identity(newSearcher(reader), vectors, 0, 17, 63);
      AtomicReference<Throwable> closeFailure = new AtomicReference<>();
      Thread closingThread =
          new Thread(
              () -> {
                try {
                  reader.close();
                } catch (Throwable t) {
                  closeFailure.set(t);
                }
              },
              "graph-only-reader-close");
      closingThread.start();
      closingThread.join();
      if (closeFailure.get() != null) {
        throw new AssertionError("reader close failed on another thread", closeFailure.get());
      }

      // KnnVectorsReader is Closeable; a second close must not destroy the native handle twice.
      reader.close();
    }
  }

  @Test
  public void testGraphOnlyCagraAndBruteForceReopens() throws Exception {
    GPUSearchParams params =
        new GPUSearchParams.Builder()
            .withIndexType(CAGRA_AND_BRUTE_FORCE)
            .withCagraPersistenceMode(GRAPH_ONLY)
            .withCagraSerializationBufferSize(16 * 1024 * 1024)
            .build();
    assertIndexReopens(96, params);
  }

  @Test
  public void testGraphOnlyReopensFromCompoundFile() throws Exception {
    float[][] vectors = vectors(64, 95);
    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer =
          new IndexWriter(
              directory,
              writerConfig(graphOnlyParams()).setUseCompoundFile(true).setInfoStream(infoStream))) {
        for (int i = 0; i < vectors.length; i++) {
          writer.addDocument(document(i, vectors[i]));
        }
      }

      assertTrue(
          "test setup did not create a compound file: " + List.of(directory.listAll()),
          List.of(directory.listAll()).stream().anyMatch(file -> file.endsWith(".cfs")));
      assertNoCagraBuildFallback(infoStream);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertTop1Identity(newSearcher(reader), vectors, 0, 17, 63);
      }
    }
  }

  @Test
  public void testSparseDocumentsPreserveVectorOrdinalToDocumentMapping() throws Exception {
    int dimension = 95;
    float[][] vectors = vectors(96, dimension);
    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer =
          new IndexWriter(directory, writerConfig(graphOnlyParams()).setInfoStream(infoStream))) {
        for (int id = 0; id < vectors.length; id++) {
          Document document = new Document();
          document.add(new StringField(ID_FIELD, Integer.toString(id), Field.Store.YES));
          if (id % 3 != 1) {
            document.add(new KnnFloatVectorField(VECTOR_FIELD, vectors[id], EUCLIDEAN));
          }
          writer.addDocument(document);
        }
      }

      assertNoCagraBuildFallback(infoStream);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(96, reader.numDocs());
        assertTop1Identity(newSearcher(reader), vectors, 0, 17, 63, 95);
      }
    }
  }

  @Test
  public void testTwoUnalignedVectorFieldsReopen() throws Exception {
    String field95 = "vector95";
    String field97 = "vector97";
    float[][] vectors95 = vectors(64, 95);
    float[][] vectors97 = vectors(64, 97);
    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      try (IndexWriter writer =
          new IndexWriter(directory, writerConfig(graphOnlyParams()).setInfoStream(infoStream))) {
        for (int id = 0; id < vectors95.length; id++) {
          Document document = new Document();
          document.add(new StringField(ID_FIELD, Integer.toString(id), Field.Store.YES));
          document.add(new KnnFloatVectorField(field95, vectors95[id], EUCLIDEAN));
          document.add(new KnnFloatVectorField(field97, vectors97[id], EUCLIDEAN));
          writer.addDocument(document);
        }
      }

      assertNoCagraBuildFallback(infoStream);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);
        assertTop1Identity(searcher, field95, vectors95, 0, 17, 63);
        assertTop1Identity(searcher, field97, vectors97, 0, 17, 63);
      }
    }
  }

  @Test
  public void testMixedLegacyAndGraphOnlySegmentsReopen() throws Exception {
    float[][] vectors = vectors(128, 96);
    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      writeMixedPersistenceSegments(directory, vectors, infoStream);
      assertMixedSegmentsSearch(directory, vectors);
      assertNoCagraBuildFallback(infoStream);
    }
  }

  @Test
  public void testMixedSegmentsForceMergeToGraphOnlyV1() throws Exception {
    assertMixedSegmentsMergeTo(graphOnlyParams(), 1);
  }

  @Test
  public void testMixedSegmentsForceMergeToLegacyV0() throws Exception {
    assertMixedSegmentsMergeTo(new GPUSearchParams.Builder().build(), 0);
  }

  @Test
  public void testDeletedRowsSurviveNativeMergeAndReopen() throws Exception {
    int dimension = 96;
    float[][] vectors = vectors(128, dimension);
    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      IndexWriterConfig segmentConfig =
          writerConfig(graphOnlyParams())
              .setMergePolicy(NoMergePolicy.INSTANCE)
              .setInfoStream(infoStream);
      try (IndexWriter writer = new IndexWriter(directory, segmentConfig)) {
        for (int i = 0; i < vectors.length; i++) {
          writer.addDocument(document(i, vectors[i]));
          if (i == 63) {
            writer.commit();
          }
        }
        writer.commit();
        writer.deleteDocuments(new Term(ID_FIELD, "3"), new Term(ID_FIELD, "70"));
        writer.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals("test setup must create two input segments", 2, reader.leaves().size());
      }

      try (IndexWriter writer =
          new IndexWriter(directory, writerConfig(graphOnlyParams()).setInfoStream(infoStream))) {
        writer.forceMerge(1);
      }

      assertTrue(
          "graph-only inputs fell back from native CAGRA merge: " + infoStream.messages(),
          infoStream.messages().stream()
              .anyMatch(message -> message.contains("Successfully merged 2 CAGRA indexes")));
      assertFalse(
          "a CAGRA build fell back to brute force: " + infoStream.messages(),
          infoStream.messages().stream()
              .anyMatch(message -> message.contains("CAGRA build failed")));

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(1, reader.leaves().size());
        assertEquals(126, reader.numDocs());
        IndexSearcher searcher = newSearcher(reader);
        assertEquals(0, searcher.count(new TermQuery(new Term(ID_FIELD, "3"))));
        assertEquals(0, searcher.count(new TermQuery(new Term(ID_FIELD, "70"))));
        assertTop1Identity(searcher, vectors, 0, 2, 4, 69, 71, 127);
        assertVectorResultsExcludeIds(searcher, vectors[3], Set.of("3", "70"));
        assertVectorResultsExcludeIds(searcher, vectors[70], Set.of("3", "70"));
      }
    }
  }

  private void assertIndexReopens(int dimension, GPUSearchParams params) throws Exception {
    float[][] vectors = vectors(64, dimension);
    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      writeIndex(directory, vectors, params, infoStream);
      assertNoCagraBuildFallback(infoStream);

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(64, reader.numDocs());
        assertTop1Identity(newSearcher(reader), vectors, 0, 17, 63);
      }
    }
  }

  private void assertMixedSegmentsMergeTo(GPUSearchParams targetParams, int expectedVersion)
      throws Exception {
    float[][] vectors = vectors(128, 96);
    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      writeMixedPersistenceSegments(directory, vectors, infoStream);
      assertMixedSegmentsSearch(directory, vectors);

      try (IndexWriter writer =
          new IndexWriter(directory, writerConfig(targetParams).setInfoStream(infoStream))) {
        writer.forceMerge(1);
      }

      assertFormatVersion(directory, expectedVersion);
      assertNoCagraBuildFallback(infoStream);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(1, reader.leaves().size());
        assertEquals(vectors.length, reader.numDocs());
        assertTop1Identity(newSearcher(reader), vectors, 0, 17, 63, 64, 95, 127);
      }
    }
  }

  private void writeMixedPersistenceSegments(
      Directory directory, float[][] vectors, RecordingInfoStream infoStream) throws IOException {
    assertEquals("mixed-version test expects two equal batches", 0, vectors.length % 2);
    int split = vectors.length / 2;
    try (IndexWriter writer =
        new IndexWriter(
            directory,
            writerConfig(new GPUSearchParams.Builder().build())
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setInfoStream(infoStream))) {
      for (int id = 0; id < split; id++) {
        writer.addDocument(document(id, vectors[id]));
      }
    }
    try (IndexWriter writer =
        new IndexWriter(
            directory,
            writerConfig(graphOnlyParams())
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setInfoStream(infoStream))) {
      for (int id = split; id < vectors.length; id++) {
        writer.addDocument(document(id, vectors[id]));
      }
    }
  }

  private void assertMixedSegmentsSearch(Directory directory, float[][] vectors)
      throws IOException {
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals("test setup must retain one v0 and one v1 segment", 2, reader.leaves().size());
      assertTop1Identity(newSearcher(reader), vectors, 0, 17, 63, 64, 95, 127);
    }
  }

  private void writeIndex(
      Directory directory,
      float[][] vectors,
      GPUSearchParams params,
      RecordingInfoStream infoStream)
      throws IOException {
    try (IndexWriter writer =
        new IndexWriter(directory, writerConfig(params).setInfoStream(infoStream))) {
      for (int i = 0; i < vectors.length; i++) {
        writer.addDocument(document(i, vectors[i]));
      }
    }
  }

  private void assertTop1Identity(IndexSearcher searcher, float[][] vectors, int... expectedIds)
      throws IOException {
    assertTop1Identity(searcher, VECTOR_FIELD, vectors, expectedIds);
  }

  private void assertTop1Identity(
      IndexSearcher searcher, String vectorField, float[][] vectors, int... expectedIds)
      throws IOException {
    for (int expectedId : expectedIds) {
      ScoreDoc[] hits =
          searcher.search(new KnnFloatVectorQuery(vectorField, vectors[expectedId], 1), 1)
              .scoreDocs;
      assertEquals("missing top-1 result for query " + expectedId, 1, hits.length);
      assertEquals(
          "wrong top-1 document for query " + expectedId,
          Integer.toString(expectedId),
          searcher.storedFields().document(hits[0].doc).get(ID_FIELD));
    }
  }

  private void assertFormatVersion(Directory directory, int expectedVersion) throws IOException {
    String indexFile = onlyFileWithExtension(directory, CUVS_INDEX_EXT);
    try (IndexInput input = directory.openInput(indexFile, IOContext.DEFAULT)) {
      assertEquals(expectedVersion, readIndexHeader(input, CUVS_INDEX_CODEC_NAME).version());
    }

    String metaFile = onlyFileWithExtension(directory, CUVS_META_CODEC_EXT);
    try (ChecksumIndexInput input = directory.openChecksumInput(metaFile)) {
      assertEquals(expectedVersion, readIndexHeader(input, CUVS_META_CODEC_NAME).version());
    }
  }

  private void assertVectorResultsExcludeIds(
      IndexSearcher searcher, float[] query, Set<String> excludedIds) throws IOException {
    ScoreDoc[] hits =
        searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, query, 32), 32).scoreDocs;
    assertTrue("vector query returned no surviving documents", hits.length > 0);
    for (ScoreDoc hit : hits) {
      String id = searcher.storedFields().document(hit.doc).get(ID_FIELD);
      assertFalse("deleted id " + id + " appeared in vector results", excludedIds.contains(id));
    }
  }

  private List<String> resultIds(IndexSearcher searcher, float[] query, int topK)
      throws IOException {
    List<String> ids = new ArrayList<>();
    for (ScoreDoc hit :
        searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, query, topK), topK).scoreDocs) {
      ids.add(searcher.storedFields().document(hit.doc).get(ID_FIELD));
    }
    return ids;
  }

  private void assertNoCagraBuildFallback(RecordingInfoStream infoStream) {
    assertFalse(
        "a CAGRA build fell back to brute force: " + infoStream.messages(),
        infoStream.messages().stream().anyMatch(message -> message.contains("CAGRA build failed")));
  }

  private void assertLegacyV0MetadataShape(Directory directory, int dimension, int count)
      throws IOException {
    String indexFile = onlyFileWithExtension(directory, CUVS_INDEX_EXT);
    Header indexHeader;
    try (IndexInput input = directory.openInput(indexFile, IOContext.DEFAULT)) {
      indexHeader = readIndexHeader(input, CUVS_INDEX_CODEC_NAME);
      assertEquals("legacy index version", 0, indexHeader.version());
    }

    String metaFile = onlyFileWithExtension(directory, CUVS_META_CODEC_EXT);
    try (ChecksumIndexInput input = directory.openChecksumInput(metaFile)) {
      Header metaHeader = readIndexHeader(input, CUVS_META_CODEC_NAME);
      assertEquals("legacy metadata version", 0, metaHeader.version());
      assertTrue("invalid field number", input.readInt() >= 0);
      assertEquals(FLOAT32.ordinal(), input.readInt());
      assertEquals(CuVS2510GPUVectorsWriter.distFuncToOrd(EUCLIDEAN), input.readInt());
      assertEquals(dimension, input.readInt());
      assertEquals(count, input.readInt());
      assertEquals(indexHeader.length(), input.readVLong());
      long cagraLength = input.readVLong();
      assertTrue("legacy metadata has no CAGRA bytes", cagraLength > 0);
      assertEquals(indexHeader.length() + cagraLength, input.readVLong());
      assertEquals(0L, input.readVLong());
      assertEquals(-1, input.readInt());
      assertEquals(input.length() - CodecUtil.footerLength(), input.getFilePointer());
      CodecUtil.checkFooter(input);
    }
  }

  private String onlyFileWithExtension(Directory directory, String extension) throws IOException {
    Set<String> matches = new HashSet<>();
    String[] files = directory.listAll();
    for (String file : files) {
      if (file.endsWith("." + extension)) {
        matches.add(file);
      }
    }
    assertEquals("expected one ." + extension + " file among " + List.of(files), 1, matches.size());
    return matches.iterator().next();
  }

  private Header readIndexHeader(IndexInput input, String expectedCodec) throws IOException {
    assertEquals(CodecUtil.CODEC_MAGIC, CodecUtil.readBEInt(input));
    assertEquals(expectedCodec, input.readString());
    int version = CodecUtil.readBEInt(input);
    byte[] segmentId = new byte[16];
    input.readBytes(segmentId, 0, segmentId.length);
    int suffixLength = Byte.toUnsignedInt(input.readByte());
    byte[] suffix = new byte[suffixLength];
    input.readBytes(suffix, 0, suffix.length);
    return new Header(version, input.getFilePointer());
  }

  private record Header(int version, long length) {}

  private GPUSearchParams graphOnlyParams() {
    return new GPUSearchParams.Builder()
        .withCagraPersistenceMode(GRAPH_ONLY)
        .withCagraSerializationBufferSize(16 * 1024 * 1024)
        .build();
  }

  private IndexWriterConfig writerConfig(GPUSearchParams params) throws IOException {
    return new IndexWriterConfig()
        .setUseCompoundFile(false)
        .setCodec(TestUtil.alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat(params)));
  }

  private IndexWriterConfig writerConfig(
      GPUSearchParams params, CuVSReaderResourcesFactory readerResourcesFactory) throws Exception {
    return new IndexWriterConfig()
        .setUseCompoundFile(false)
        .setCodec(
            new CuVS2510GPUSearchCodec(
                params, FilterBitsetCacheConfig.DEFAULT, readerResourcesFactory));
  }

  private Document document(int id, float[] vector) {
    Document document = new Document();
    document.add(new StringField(ID_FIELD, Integer.toString(id), Field.Store.YES));
    document.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
    return document;
  }

  private float[][] vectors(int count, int dimension) {
    float[][] vectors = new float[count][dimension];
    for (int row = 0; row < count; row++) {
      for (int column = 0; column < dimension; column++) {
        vectors[row][column] = random().nextFloat();
      }
    }
    return vectors;
  }

  private static final class RecordingInfoStream extends InfoStream {
    private final List<String> messages = Collections.synchronizedList(new ArrayList<>());

    @Override
    public void message(String component, String message) {
      messages.add(component + ": " + message);
    }

    @Override
    public boolean isEnabled(String component) {
      return true;
    }

    @Override
    public void close() {}

    List<String> messages() {
      synchronized (messages) {
        return List.copyOf(messages);
      }
    }
  }

  private static final class TrackingCuVSResources implements CuVSResources {
    private final CuVSResources delegate;
    private final RuntimeException accessFailure;
    private int closeCalls;
    private int accessCalls;
    private int tempDirectoryCalls;

    private TrackingCuVSResources(CuVSResources delegate) {
      this(delegate, null);
    }

    private TrackingCuVSResources(CuVSResources delegate, RuntimeException accessFailure) {
      this.delegate = delegate;
      this.accessFailure = accessFailure;
    }

    @Override
    public ScopedAccess access() {
      if (closeCalls != 0) {
        throw new IllegalStateException("resources are closed");
      }
      accessCalls++;
      if (accessFailure != null) {
        throw accessFailure;
      }
      return delegate.access();
    }

    @Override
    public int deviceId() {
      return delegate.deviceId();
    }

    @Override
    public void close() {
      if (closeCalls++ == 0 && delegate != null) {
        delegate.close();
      }
    }

    @Override
    public Path tempDirectory() {
      tempDirectoryCalls++;
      return delegate.tempDirectory();
    }

    @Override
    public void setWorkspacePool(long initialSizeBytes) {
      delegate.setWorkspacePool(initialSizeBytes);
    }
  }
}
