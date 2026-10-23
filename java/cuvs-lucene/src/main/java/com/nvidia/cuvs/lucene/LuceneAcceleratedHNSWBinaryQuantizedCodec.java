/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.codecs.Codec;

/**
 * cuVS based codec for GPU graph construction and CPU HNSW search. Binary quantization is used
 * only while building the graph; persisted vectors and CPU scoring remain full precision.
 *
 * <p>This codec, named {@code Lucene101AcceleratedHNSWBinaryQuantizedCodec}, wraps {@code
 * Lucene101Codec}, the default codec of Lucene 10.2. {@link CuVSCodecs#acceleratedHNSWBinaryQuantized} returns the codec of
 * this family for the Lucene release in use.
 *
 * @since 26.02
 */
public class LuceneAcceleratedHNSWBinaryQuantizedCodec extends CuVSFilterCodec {

  private static final String NAME = "Lucene101AcceleratedHNSWBinaryQuantizedCodec";

  public LuceneAcceleratedHNSWBinaryQuantizedCodec() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  public LuceneAcceleratedHNSWBinaryQuantizedCodec(String name, Codec delegate) {
    super(
        name,
        delegate,
        () ->
            new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(
                new AcceleratedHNSWParams.Builder().build()));
  }

  public LuceneAcceleratedHNSWBinaryQuantizedCodec(AcceleratedHNSWParams acceleratedHNSWParams) {
    super(
        NAME,
        delegate(NAME, () -> LuceneCompat.lucene101Codec()),
        () -> new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(acceleratedHNSWParams));
  }
}
