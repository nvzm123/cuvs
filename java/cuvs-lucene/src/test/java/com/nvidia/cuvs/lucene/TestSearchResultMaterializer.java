/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.junit.Test;

public class TestSearchResultMaterializer {
  private static final String ID_FIELD = "dataset_id";

  @Test
  public void testRestoresRankAcrossLeavesWithoutMutatingHitsOrNarrowingIds() throws Exception {
    // Above double's exact integer range: ID transfer must never pass through a floating type.
    long largeDatasetId = (1L << 53) + 1;
    Map<String, Object> firstResult;
    try (var directory = new ByteBuffersDirectory();
        var writer =
            new IndexWriter(
                directory, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      writer.addDocument(documentWithId(largeDatasetId));
      writer.addDocument(documentWithId(27));
      writer.commit();
      writer.addDocument(documentWithId(-7));
      writer.commit();
      try (var reader = DirectoryReader.open(directory)) {
        assertEquals(
            "The test needs two independent doc-values iterators", 2, reader.leaves().size());
        // Explicit synthetic search order differs from document order, including within the first
        // leaf.
        TopDocs hits = hits(new ScoreDoc(2, 0.9f), new ScoreDoc(1, 0.8f), new ScoreDoc(0, 0.7f));
        ScoreDoc[] originalOrder = hits.scoreDocs.clone();
        firstResult = materialize(reader, hits);
        Map<String, Object> secondResult = materialize(reader, hits);

        assertArrayEquals(new long[] {-7, 27, largeDatasetId}, ids(firstResult));
        assertArrayEquals(new float[] {0.9f, 0.8f, 0.7f}, scores(firstResult), 0f);
        assertArrayEquals(ids(firstResult), ids(secondResult));
        for (int rank = 0; rank < originalOrder.length; rank++) {
          assertSame(originalOrder[rank], hits.scoreDocs[rank]);
        }
        ids(secondResult)[0] = 0;
        assertEquals("Separate calls must own separate output arrays", -7, ids(firstResult)[0]);
      }
    }
    assertArrayEquals(
        "Closing the reader must not invalidate results",
        new long[] {-7, 27, largeDatasetId},
        ids(firstResult));
  }

  @Test
  public void testDeletedDocumentGapsDoNotChangeDatasetIds() throws Exception {
    try (var directory = new ByteBuffersDirectory();
        var writer =
            new IndexWriter(
                directory, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      for (long id : new long[] {101, 202, 303}) {
        Document document = documentWithId(id);
        document.add(new StringField("delete_key", Long.toString(id), Field.Store.NO));
        writer.addDocument(document);
      }
      writer.commit();
      writer.deleteDocuments(new Term("delete_key", "202"));
      writer.commit();
      try (var reader = DirectoryReader.open(directory)) {
        assertEquals(3, reader.maxDoc());
        assertEquals(2, reader.numDocs());
        TopDocs hits = new IndexSearcher(reader).search(new MatchAllDocsQuery(), 3);

        assertArrayEquals(new long[] {101, 303}, ids(materialize(reader, hits)));
      }
    }
  }

  @Test
  public void testEmptyAndSingleHitResultsKeepTheirActualSizes() throws Exception {
    try (var directory = new ByteBuffersDirectory();
        var writer = new IndexWriter(directory, new IndexWriterConfig())) {
      writer.addDocument(documentWithId(42));
      writer.commit();
      try (var reader = DirectoryReader.open(directory)) {
        assertArrayEquals(new long[0], ids(materialize(reader, hits())));
        assertArrayEquals(new float[0], scores(materialize(reader, hits())), 0f);
        assertArrayEquals(new long[] {42}, ids(materialize(reader, hits(new ScoreDoc(0, 1f)))));
      }
    }
  }

  @Test
  public void testMissingIdsAreErrorsRatherThanInternalDocumentIdSubstitutes() throws Exception {
    try (var directory = new ByteBuffersDirectory();
        var writer =
            new IndexWriter(
                directory, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      writer.addDocument(new Document());
      writer.commit();
      writer.addDocument(documentWithId(42));
      writer.addDocument(new Document());
      writer.commit();
      try (var reader = DirectoryReader.open(directory)) {
        assertEquals(2, reader.leaves().size());
        for (int missingIdDocument : new int[] {0, 2}) {
          IllegalStateException error =
              assertThrows(
                  IllegalStateException.class,
                  () -> materialize(reader, hits(new ScoreDoc(missingIdDocument, 1f))));
          assertTrue(error.getMessage(), error.getMessage().contains("no numeric dataset ID"));
        }
        for (int invalidDocument : new int[] {-1, reader.maxDoc()}) {
          assertThrows(
              IllegalArgumentException.class,
              () -> materialize(reader, hits(new ScoreDoc(invalidDocument, 1f))));
        }
      }
    }
  }

  @Test
  public void testRejectsIncompleteRequests() {
    var materializer = new SearchResultMaterializer();
    assertThrows(IllegalArgumentException.class, () -> materializer.apply(null));
    IllegalArgumentException error =
        assertThrows(IllegalArgumentException.class, () -> materializer.apply(Map.of()));
    assertEquals("reader must have type IndexReader", error.getMessage());
  }

  private static Document documentWithId(long id) {
    Document document = new Document();
    document.add(new NumericDocValuesField(ID_FIELD, id));
    return document;
  }

  private static TopDocs hits(ScoreDoc... scoreDocs) {
    return new TopDocs(new TotalHits(scoreDocs.length, TotalHits.Relation.EQUAL_TO), scoreDocs);
  }

  private static Map<String, Object> materialize(IndexReader reader, TopDocs hits) {
    var request = new HashMap<String, Object>();
    request.put(SearchResultMaterializer.READER_KEY, reader);
    request.put(SearchResultMaterializer.TOP_DOCS_KEY, hits);
    request.put(SearchResultMaterializer.ID_FIELD_KEY, ID_FIELD);
    return new SearchResultMaterializer().apply(request);
  }

  private static long[] ids(Map<String, Object> result) {
    return (long[]) result.get(SearchResultMaterializer.DOCUMENT_IDS_KEY);
  }

  private static float[] scores(Map<String, Object> result) {
    return (float[]) result.get(SearchResultMaterializer.SCORES_KEY);
  }
}
