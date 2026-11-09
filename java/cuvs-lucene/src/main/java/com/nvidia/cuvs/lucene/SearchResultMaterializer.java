/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

/**
 * Copies search hits into primitive dataset-ID and score arrays in their original result order.
 *
 * <p>The request contains {@code reader} ({@link IndexReader}), {@code top_docs} ({@link TopDocs})
 * from that reader, and {@code id_field} (a {@link String} naming a numeric doc-values field).
 * The response contains {@code document_ids} ({@code long[]}) and {@code scores} ({@code float[]}).
 * These new arrays remain valid after the reader closes; this class neither closes the reader nor
 * retains it. Missing IDs are errors, not a reason to substitute Lucene's internal document IDs.
 *
 * <p>{@link Function} and {@link Map} let generated bindings call this helper without wrapping a new
 * Java type. Retrieval stays separate from {@link IndexSearcherTimingBridge}'s search-only timer.
 * An I/O failure is exposed as an {@link UncheckedIOException}.
 */
public final class SearchResultMaterializer
    implements Function<Map<String, Object>, Map<String, Object>> {
  public static final String READER_KEY = "reader";
  public static final String TOP_DOCS_KEY = "top_docs";
  public static final String ID_FIELD_KEY = "id_field";
  public static final String DOCUMENT_IDS_KEY = "document_ids";
  public static final String SCORES_KEY = "scores";

  @Override
  public Map<String, Object> apply(Map<String, Object> request) {
    IndexReader reader = requiredValue(request, READER_KEY, IndexReader.class);
    TopDocs topDocs = requiredValue(request, TOP_DOCS_KEY, TopDocs.class);
    String idField = requiredValue(request, ID_FIELD_KEY, String.class);
    try {
      return materialize(reader, topDocs.scoreDocs, idField);
    } catch (IOException error) {
      throw new UncheckedIOException("Reading numeric dataset IDs failed", error);
    }
  }

  private static Map<String, Object> materialize(
      IndexReader reader, ScoreDoc[] scoreDocs, String idField) throws IOException {
    long[] documentIds = new long[scoreDocs.length];
    float[] scores = new float[scoreDocs.length];
    RankedHit[] hitsInDocIdOrder = new RankedHit[scoreDocs.length];
    for (int rank = 0; rank < scoreDocs.length; rank++) {
      ScoreDoc hit = scoreDocs[rank];
      if (hit.doc < 0 || hit.doc >= reader.maxDoc()) {
        throw new IllegalArgumentException("Lucene document " + hit.doc + " is outside the index");
      }
      hitsInDocIdOrder[rank] = new RankedHit(rank, hit);
      scores[rank] = hit.score;
    }
    // NumericDocValues advances only forward. Sorting access must not reorder the returned hits.
    Arrays.sort(hitsInDocIdOrder, Comparator.comparingInt(hit -> hit.scoreDoc().doc));
    int nextHit = 0;
    for (var leaf : reader.leaves()) {
      int docLimit = leaf.docBase + leaf.reader().maxDoc();
      if (nextHit == hitsInDocIdOrder.length
          || hitsInDocIdOrder[nextHit].scoreDoc().doc >= docLimit) {
        continue;
      }
      NumericDocValues ids = leaf.reader().getNumericDocValues(idField);
      if (ids == null) {
        throw new IllegalStateException("Lucene index has no numeric dataset IDs in " + idField);
      }
      while (nextHit < hitsInDocIdOrder.length
          && hitsInDocIdOrder[nextHit].scoreDoc().doc < docLimit) {
        RankedHit hit = hitsInDocIdOrder[nextHit++];
        int localDocId = hit.scoreDoc().doc - leaf.docBase;
        if (!ids.advanceExact(localDocId)) {
          throw new IllegalStateException(
              "Lucene document " + hit.scoreDoc().doc + " has no numeric dataset ID in " + idField);
        }
        documentIds[hit.rank()] = ids.longValue();
      }
    }
    return Map.of(DOCUMENT_IDS_KEY, documentIds, SCORES_KEY, scores);
  }

  private record RankedHit(int rank, ScoreDoc scoreDoc) {}

  private static <T> T requiredValue(Map<String, Object> request, String key, Class<T> type) {
    if (request == null) {
      throw new IllegalArgumentException("request must not be null");
    }
    Object value = request.get(key);
    if (!type.isInstance(value)) {
      throw new IllegalArgumentException(key + " must have type " + type.getSimpleName());
    }
    return type.cast(value);
  }
}
