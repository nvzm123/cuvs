/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.function.BiFunction;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

/** Reads external IDs from the same reader that produced the hits and returns them in score order. */
public final class RankedSearchHits implements BiFunction<IndexReader, TopDocs, byte[]> {
  private static final int RECORD_BYTES = Long.BYTES + Float.BYTES;
  // Use the ordinary per-hit reader for exceptionally large results instead of a large JNI array.
  private static final int MAX_TRANSFER_BYTES = 8 * 1024 * 1024;

  /**
   * Packs one little-endian (signed long ID, float score) per hit. A null result requests the
   * caller's existing per-hit path, which also owns the diagnostic for a missing external ID.
   * An empty result is a zero-length array.
   */
  @Override
  public byte[] apply(IndexReader reader, TopDocs results) {
    ScoreDoc[] hits = results.scoreDocs;
    if (hits.length > MAX_TRANSFER_BYTES / RECORD_BYTES) {
      return null;
    }
    try {
      return encode(reader, hits);
    } catch (IOException error) {
      throw new ReadFailure(error);
    }
  }

  private static byte[] encode(IndexReader reader, ScoreDoc[] hits) throws IOException {
    // Doc values advance in document order. Retain the original rank to write straight into the
    // caller's result order, including tied scores or repeated documents.
    long[] byDocument = new long[hits.length];
    for (int rank = 0; rank < hits.length; rank++) {
      byDocument[rank] = ((long) hits[rank].doc << 32) | rank;
    }
    Arrays.sort(byDocument);
    ByteBuffer output =
        ByteBuffer.allocate(hits.length * RECORD_BYTES).order(ByteOrder.LITTLE_ENDIAN);
    int next = 0;
    for (var leaf : reader.leaves()) {
      if (next == hits.length) {
        break;
      }
      int limit = leaf.docBase + leaf.reader().maxDoc();
      if (documentId(byDocument[next]) >= limit) {
        continue;
      }
      var ids = leaf.reader().getNumericDocValues("id");
      if (ids == null) {
        return null;
      }
      while (next < hits.length && documentId(byDocument[next]) < limit) {
        long documentAndRank = byDocument[next++];
        int localDoc = documentId(documentAndRank) - leaf.docBase;
        if (localDoc < 0 || !ids.advanceExact(localDoc)) {
          return null;
        }
        int rank = (int) documentAndRank;
        int offset = rank * RECORD_BYTES;
        output.putLong(offset, ids.longValue());
        output.putFloat(offset + Long.BYTES, hits[rank].score);
      }
    }
    return next == hits.length ? output.array() : null;
  }

  // Distinguish checked exceptions wrapped by this bridge from unchecked exceptions raised by
  // a custom DocValues reader, which must propagate in their original form.
  private static final class ReadFailure extends UncheckedIOException {
    private static final long serialVersionUID = 1L;

    ReadFailure(IOException cause) {
      super(cause);
    }
  }

  private static int documentId(long documentAndRank) {
    return (int) (documentAndRank >> 32);
  }
}
