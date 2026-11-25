/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Test;

public class TestFloatQueryVector {
  @Test
  public void copiesExactFloatBits() {
    int[] bits = {0x80000000, 1, 0xbf000001, 0x7f7fffff};
    ByteBuffer input =
        ByteBuffer.allocate(bits.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int value : bits) {
      input.putInt(value);
    }
    FloatQueryVector bridge = new FloatQueryVector();
    float[] output = bridge.apply(input.array());
    assertEquals(bits.length, output.length);
    for (int i = 0; i < bits.length; i++) {
      assertEquals(bits[i], Float.floatToRawIntBits(output[i]));
    }
    assertArrayEquals(new float[0], bridge.apply(new byte[0]), 0);
    assertThrows(IllegalArgumentException.class, () -> bridge.apply(new byte[3]));
  }
}
