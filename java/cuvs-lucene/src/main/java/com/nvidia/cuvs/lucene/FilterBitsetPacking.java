/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Arrays;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

/** CPU-only packing of accepted vector ordinals for a segment's filter handle. */
final class FilterBitsetPacking {
  record PackedBits(long[] words, int cardinality) {}

  private FilterBitsetPacking() {}

  static PackedBits pack(FloatVectorValues vectors, Bits acceptDocs, int maxDoc) {
    Bits acceptedOrds = vectors.getAcceptOrds(acceptDocs);
    int numOrds = vectors.size();
    long[] words = new long[(int) (((long) numOrds + 63) / 64)];
    int cardinality;
    // Returning the same object proves that acceptDocs already addresses vector ordinals.
    // Also require a vector for every document: size alone must not be used to infer mapping.
    // Sparse mappings and live-document intersection wrappers retain the scalar path below.
    if (numOrds == maxDoc
        && acceptedOrds == acceptDocs
        && acceptDocs instanceof FixedBitSet fixed) {
      System.arraycopy(fixed.getBits(), 0, words, 0, words.length);
      maskTail(words, numOrds);
      cardinality = 0;
      for (long word : words) {
        cardinality += Long.bitCount(word);
      }
    } else if (acceptedOrds == null) {
      // Lucene uses null to mean that every vector ordinal is accepted.
      Arrays.fill(words, -1L);
      maskTail(words, numOrds);
      cardinality = numOrds;
    } else {
      cardinality = 0;
      for (int ord = 0; ord < numOrds; ord++) {
        if (acceptedOrds.get(ord)) {
          words[ord >>> 6] |= 1L << (ord & 63);
          cardinality++;
        }
      }
    }
    return new PackedBits(words, cardinality);
  }

  private static void maskTail(long[] words, int numOrds) {
    int tail = numOrds & 63;
    if (tail != 0) {
      words[words.length - 1] &= (1L << tail) - 1L;
    }
  }
}
