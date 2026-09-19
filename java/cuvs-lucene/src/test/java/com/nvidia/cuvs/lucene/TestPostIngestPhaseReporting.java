/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.InfoStream;

public class TestPostIngestPhaseReporting extends LuceneTestCase {

  public void testReportsPhaseDimensions() {
    AtomicReference<String> message = new AtomicReference<>();
    InfoStream infoStream =
        new InfoStream() {
          @Override
          public void message(String component, String value) {
            message.set(component + ":" + value);
          }

          @Override
          public boolean isEnabled(String component) {
            return true;
          }

          @Override
          public void close() {}
        };

    Lucene99AcceleratedHNSWVectorsWriter.reportPhase(
        infoStream, "_segment", "flat_vector_flush", System.nanoTime(), 4096L);

    assertNotNull(message.get());
    assertTrue(message.get().contains("benchmark_phase=flat_vector_flush"));
    assertTrue(message.get().contains("segment=_segment"));
    assertTrue(message.get().contains("duration_ns="));
    assertTrue(message.get().contains("bytes=4096"));
  }

  public void testRuntimeFailureDoesNotAffectIndexing() {
    InfoStream throwingInfoStream =
        new InfoStream() {
          @Override
          public void message(String component, String message) {
            throw new IllegalStateException("telemetry failed");
          }

          @Override
          public boolean isEnabled(String component) {
            return true;
          }

          @Override
          public void close() throws IOException {}
        };

    Lucene99AcceleratedHNSWVectorsWriter.reportPhase(
        throwingInfoStream, "_segment", "flush_total", System.nanoTime(), -1L);
  }
}
