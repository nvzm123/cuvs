/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestCuVS2510GPUVectorsWriterFailureHandling extends LuceneTestCase {

  @Test
  public void testRuntimeFailuresCanFallBackToBruteForce() {
    assertTrue(
        CuVS2510GPUVectorsWriter.canFallbackToBruteForce(
            new RuntimeException("native CAGRA failure")));
  }

  @Test
  public void testIoFailuresCannotFallBackToBruteForce() {
    assertFalse(CuVS2510GPUVectorsWriter.canFallbackToBruteForce(new IOException("write failure")));
  }

  @Test
  public void testErrorsCannotFallBackToBruteForce() {
    assertFalse(
        CuVS2510GPUVectorsWriter.canFallbackToBruteForce(new AssertionError("fatal failure")));
  }

  @Test
  public void testUnexpectedCheckedFailuresCannotFallBackToBruteForce() {
    assertFalse(CuVS2510GPUVectorsWriter.canFallbackToBruteForce(new Exception("checked failure")));
  }

  @Test
  public void testFailedCagraWriteRecordsNoPayload() {
    assertEquals(0, CuVS2510GPUVectorsWriter.completedIndexLength(false, 17, 41));
    assertEquals(24, CuVS2510GPUVectorsWriter.completedIndexLength(true, 17, 41));
  }

  @Test
  public void testIndexCloseFailureClosesDataset() {
    AtomicBoolean datasetClosed = new AtomicBoolean();
    RuntimeException indexFailure = new RuntimeException("index close failure");

    RuntimeException thrown =
        expectThrows(
            RuntimeException.class,
            () ->
                CuVS2510GPUVectorsWriter.closeIndexWithDatasetFallback(
                    () -> {
                      throw indexFailure;
                    },
                    () -> datasetClosed.set(true)));

    assertSame(indexFailure, thrown);
    assertTrue(datasetClosed.get());
  }

  @Test
  public void testDatasetCloseFailureIsSuppressed() {
    RuntimeException indexFailure = new RuntimeException("index close failure");
    RuntimeException datasetFailure = new RuntimeException("dataset close failure");

    RuntimeException thrown =
        expectThrows(
            RuntimeException.class,
            () ->
                CuVS2510GPUVectorsWriter.closeIndexWithDatasetFallback(
                    () -> {
                      throw indexFailure;
                    },
                    () -> {
                      throw datasetFailure;
                    }));

    assertSame(indexFailure, thrown);
    assertArrayEquals(new Throwable[] {datasetFailure}, thrown.getSuppressed());
  }
}
