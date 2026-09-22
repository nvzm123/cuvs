/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestGPUIndexCleanup extends LuceneTestCase {

  @Test
  public void testCleanupContinuesAndSuppressesLaterFailures() {
    List<String> closed = new ArrayList<>();
    IOException first = new IOException("first");
    IllegalStateException second = new IllegalStateException("second");

    IOException thrown =
        expectThrows(
            IOException.class,
            () ->
                GPUIndex.closeAll(
                    () -> {
                      closed.add("first");
                      throw first;
                    },
                    () -> {
                      closed.add("middle");
                    },
                    () -> {
                      closed.add("last");
                      throw second;
                    }));

    assertSame(first, thrown);
    assertEquals(List.of("first", "middle", "last"), closed);
    assertArrayEquals(new Throwable[] {second}, thrown.getSuppressed());
  }

  @Test
  public void testFailedGraphLoadKeepsPrimaryFailure() {
    IOException primary = new IOException("graph load");
    IllegalStateException cleanup = new IllegalStateException("dataset cleanup");

    CuVS2510GPUVectorsReader.closeAndSuppress(
        primary,
        () -> {
          throw cleanup;
        });

    assertArrayEquals(new Throwable[] {cleanup}, primary.getSuppressed());
  }
}
