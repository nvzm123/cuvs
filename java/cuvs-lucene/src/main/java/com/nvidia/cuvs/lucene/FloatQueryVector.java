/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.function.Function;

/** Copies a little-endian FP32 query from a byte array passed in bulk through a Java binding. */
public final class FloatQueryVector implements Function<byte[], float[]> {
  @Override
  public float[] apply(byte[] bytes) {
    if (bytes.length % Float.BYTES != 0) {
      throw new IllegalArgumentException("Expected whole float32 values");
    }
    float[] values = new float[bytes.length / Float.BYTES];
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values);
    return values;
  }
}
