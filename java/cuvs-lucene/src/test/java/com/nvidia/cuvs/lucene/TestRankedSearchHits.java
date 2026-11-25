/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.junit.Test;

public class TestRankedSearchHits {
  private static TopDocs results(ScoreDoc... hits) {
    return new TopDocs(new TotalHits(hits.length, TotalHits.Relation.EQUAL_TO), hits);
  }

  private static void add(IndexWriter writer, Long externalId) throws IOException {
    Document document = new Document();
    if (externalId != null) {
      document.add(new NumericDocValuesField("id", externalId));
    }
    writer.addDocument(document);
  }

  @Test
  public void preservesRanksScoresAndLongIdsAcrossLeaves() throws IOException {
    try (var directory = new ByteBuffersDirectory();
        var writer =
            new IndexWriter(
                directory, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      add(writer, Long.MAX_VALUE);
      add(writer, -17L);
      writer.commit();
      add(writer, 9_007_199_254_740_993L);
      add(writer, null);
      writer.commit();
      try (var reader = DirectoryReader.open(directory)) {
        var encoder = new RankedSearchHits();
        assertEquals(2, reader.leaves().size());
        var hits =
            results(
                new ScoreDoc(2, 0.5f),
                new ScoreDoc(0, 0.5f),
                new ScoreDoc(1, -0.0f),
                new ScoreDoc(2, 0.0f));
        var data = ByteBuffer.wrap(encoder.apply(reader, hits)).order(ByteOrder.LITTLE_ENDIAN);
        long[] expected = {9_007_199_254_740_993L, Long.MAX_VALUE, -17L, 9_007_199_254_740_993L};
        for (int i = 0; i < expected.length; ++i) {
          assertEquals(expected[i], data.getLong());
          assertEquals(Float.floatToRawIntBits(hits.scoreDocs[i].score), data.getInt());
        }
        assertFalse(data.hasRemaining());
        assertArrayEquals(new byte[0], encoder.apply(reader, results()));
        for (int invalid : new int[] {-1, 3, 4}) {
          assertNull(encoder.apply(reader, results(new ScoreDoc(invalid, 1f))));
        }
        assertNull(encoder.apply(reader, results(new ScoreDoc[1_000_000])));
      }
    }
  }
}
