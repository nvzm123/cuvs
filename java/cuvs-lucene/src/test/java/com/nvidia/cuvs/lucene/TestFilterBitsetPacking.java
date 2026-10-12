/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.junit.Test;

/** Exercises the production packing helper without creating a native filter handle or GPU index. */
public class TestFilterBitsetPacking {
  private static final int[] SIZES = {0, 1, 2, 63, 64, 65, 127, 128, 129, 193};

  @Test
  public void denseEmptyAndAllAcceptedAtWordBoundaries() {
    for (int size : SIZES) {
      DenseValues vectors = new DenseValues(size);
      FixedBitSet acceptDocs = new FixedBitSet(size);
      assertPacked(vectors, acceptDocs, size);
      if (size != 0) acceptDocs.set(0, size);
      assertPacked(vectors, acceptDocs, size);
      assertEquals(size, FilterBitsetPacking.pack(vectors, acceptDocs, size).cardinality());
    }
  }

  @Test
  public void denseRandomMasksMatchScalarReference() {
    Random random = new Random(47321);
    for (int size : SIZES) {
      for (int repetition = 0; repetition < 20; repetition++) {
        FixedBitSet acceptDocs = new FixedBitSet(size);
        for (int doc = 0; doc < size; doc++) {
          if (random.nextBoolean()) acceptDocs.set(doc);
        }
        assertPacked(new DenseValues(size), acceptDocs, size);
      }
    }
  }

  @Test
  public void denseWordsAreCopiedAndTailBitsAreCleared() {
    for (int size : new int[] {1, 63, 65, 127, 129}) {
      FixedBitSet acceptDocs = new FixedBitSet(size);
      acceptDocs.set(0, size);
      // Simulate an overallocated backing array and nonzero padding outside the logical bits.
      acceptDocs = FixedBitSet.ensureCapacity(acceptDocs, size + 128);
      acceptDocs.getBits()[(size - 1) >>> 6] = -1L;
      long[] original = acceptDocs.getBits().clone();
      FilterBitsetPacking.PackedBits packed =
          FilterBitsetPacking.pack(new DenseValues(size), acceptDocs, size);
      assertEquals(size, packed.cardinality());
      assertEquals((size + 63) / 64, packed.words().length);
      assertEquals((1L << (size & 63)) - 1L, packed.words()[packed.words().length - 1]);
      assertNotSame(acceptDocs.getBits(), packed.words());
      packed.words()[0] = 0;
      assertArrayEquals(original, acceptDocs.getBits());
    }
  }

  @Test
  public void nullAcceptDocsAcceptsEveryOrdinalIncludingSparseAndEmpty() {
    for (int size : SIZES) {
      DenseValues vectors = new DenseValues(size);
      assertPacked(vectors, null, size);
      assertPacked(new MappedValues(size), null, size * 2 + 1);
    }
  }

  @Test
  public void nonBitsetAndLiveDocsIntersectionUseOrdinalBits() {
    int size = 129;
    AtomicInteger reads = new AtomicInteger();
    Bits accepted =
        new Bits() {
          @Override
          public boolean get(int doc) {
            reads.incrementAndGet();
            return doc % 3 == 0 && doc % 5 != 0;
          }

          @Override
          public int length() {
            return size;
          }
        };
    FilterBitsetPacking.PackedBits packed =
        FilterBitsetPacking.pack(new DenseValues(size), accepted, size);
    assertEquals(size, reads.get());
    for (int doc = 0; doc < size; doc++) {
      assertEquals(
          doc % 3 == 0 && doc % 5 != 0, (packed.words()[doc >>> 6] & (1L << (doc & 63))) != 0);
    }
  }

  @Test
  public void equalVectorAndDocumentCountsDoNotReplaceAcceptanceMapping() {
    FixedBitSet acceptDocs = new FixedBitSet(129);
    acceptDocs.set(0);
    acceptDocs.set(128);
    FixedBitSet acceptedOrds = new FixedBitSet(129);
    acceptedOrds.set(64);
    DenseValues vectors =
        new DenseValues(129) {
          @Override
          public Bits getAcceptOrds(Bits docs) {
            return acceptedOrds;
          }
        };
    FilterBitsetPacking.PackedBits packed = FilterBitsetPacking.pack(vectors, acceptDocs, 129);
    assertArrayEquals(new long[] {0, 1, 0}, packed.words());
    assertEquals(1, packed.cardinality());
  }

  @Test
  public void sparseDocumentAcceptanceIsMappedToVectorOrdinals() {
    for (int size : SIZES) {
      int maxDoc = size * 2 + 1;
      MappedValues vectors = new MappedValues(size);
      FixedBitSet docs = new FixedBitSet(maxDoc);
      for (int doc = 0; doc < maxDoc; doc++) {
        if (doc % 3 != 0) docs.set(doc);
      }
      assertPacked(vectors, docs, maxDoc);
    }
  }

  @Test
  public void realLuceneDenseSparseAndDeletedReadersMatchOrdinalReference() throws IOException {
    for (boolean sparse : new boolean[] {false, true}) {
      for (boolean deleted : new boolean[] {false, true}) {
        try (Directory directory = new ByteBuffersDirectory();
            IndexWriter writer =
                new IndexWriter(
                    directory,
                    new IndexWriterConfig()
                        .setCodec(new Lucene101Codec())
                        .setMergePolicy(NoMergePolicy.INSTANCE))) {
          for (int doc = 0; doc < 193; doc++) {
            Document document = new Document();
            document.add(new StringField("id", Integer.toString(doc), Field.Store.NO));
            if (!sparse || doc % 3 == 0) {
              document.add(
                  new KnnFloatVectorField(
                      "vector", new float[] {doc}, VectorSimilarityFunction.EUCLIDEAN));
            }
            writer.addDocument(document);
          }
          writer.commit();
          if (deleted) {
            for (int doc = 0; doc < 193; doc += 11)
              writer.deleteDocuments(new Term("id", Integer.toString(doc)));
          }
          try (DirectoryReader reader = DirectoryReader.open(writer)) {
            assertEquals(1, reader.leaves().size());
            LeafReader leaf = reader.leaves().get(0).reader();
            FloatVectorValues vectors = leaf.getFloatVectorValues("vector");
            FixedBitSet filter = new FixedBitSet(leaf.maxDoc());
            for (int doc = 0; doc < leaf.maxDoc(); doc++) if (doc % 5 < 2) filter.set(doc);
            Bits live = leaf.getLiveDocs();
            Bits accepted =
                live == null
                    ? filter
                    : new Bits() {
                      @Override
                      public boolean get(int doc) {
                        return filter.get(doc) && live.get(doc);
                      }

                      @Override
                      public int length() {
                        return leaf.maxDoc();
                      }
                    };
            assertPacked(vectors, accepted, leaf.maxDoc());
            assertPacked(vectors, live, leaf.maxDoc());
            FilterBitsetPacking.PackedBits packed =
                FilterBitsetPacking.pack(vectors, accepted, leaf.maxDoc());
            for (int ord = 0; ord < vectors.size(); ord++) {
              int doc = vectors.ordToDoc(ord);
              assertEquals(
                  filter.get(doc) && (live == null || live.get(doc)),
                  (packed.words()[ord >>> 6] & (1L << (ord & 63))) != 0);
            }
            assertTrue(packed.cardinality() > 0);
          }
        }
      }
    }
  }

  private static void assertPacked(FloatVectorValues vectors, Bits acceptDocs, int maxDoc) {
    Bits ords = vectors.getAcceptOrds(acceptDocs);
    long[] expected = new long[(vectors.size() + 63) / 64];
    int count = 0;
    for (int ord = 0; ord < vectors.size(); ord++) {
      if (ords == null || ords.get(ord)) {
        expected[ord >>> 6] |= 1L << (ord & 63);
        count++;
      }
    }
    FilterBitsetPacking.PackedBits packed = FilterBitsetPacking.pack(vectors, acceptDocs, maxDoc);
    assertArrayEquals(expected, packed.words());
    assertEquals(count, packed.cardinality());
  }

  private static class DenseValues extends FloatVectorValues {
    final int size;

    DenseValues(int size) {
      this.size = size;
    }

    @Override
    public int dimension() {
      return 1;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public float[] vectorValue(int ord) {
      return new float[] {ord};
    }

    @Override
    public FloatVectorValues copy() {
      return this;
    }

    @Override
    public Bits getAcceptOrds(Bits docs) {
      return docs;
    }
  }

  private static final class MappedValues extends DenseValues {
    MappedValues(int size) {
      super(size);
    }

    @Override
    public int ordToDoc(int ord) {
      return ord * 2 + 1;
    }

    @Override
    public Bits getAcceptOrds(Bits docs) {
      if (docs == null) return null;
      return new Bits() {
        @Override
        public boolean get(int ord) {
          return docs.get(ordToDoc(ord));
        }

        @Override
        public int length() {
          return size;
        }
      };
    }
  }
}
