/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.junit.Test;

public class TestSequentialFlatVectorHydration extends LuceneTestCase {

  @Test
  public void testConsumesAndFinishesReturnedMergeReader() throws Exception {
    RecordingReader sequentialReader = new RecordingReader();
    RecordingReader searchReader = new RecordingReader();
    searchReader.mergeReader = sequentialReader;

    String result =
        CuVS2510GPUVectorsReader.withSequentialFlatVectors(
            searchReader,
            false,
            reader -> {
              assertSame(sequentialReader, reader);
              return "hydrated";
            },
            ignored -> {});

    assertEquals("hydrated", result);
    assertEquals(1, searchReader.getMergeInstanceCalls);
    assertEquals(1, sequentialReader.finishMergeCalls);
    assertEquals(0, searchReader.finishMergeCalls);
    assertEquals(0, sequentialReader.closeCalls);
  }

  @Test
  public void testReplayFailureRemainsPrimaryWhenFinishAlsoFails() {
    RecordingReader sequentialReader = new RecordingReader();
    sequentialReader.finishFailure = new IOException("finish failed");
    RecordingReader searchReader = new RecordingReader();
    searchReader.mergeReader = sequentialReader;
    IOException replayFailure = new IOException("replay failed");

    IOException thrown =
        expectThrows(
            IOException.class,
            () ->
                CuVS2510GPUVectorsReader.withSequentialFlatVectors(
                    searchReader,
                    false,
                    reader -> {
                      throw replayFailure;
                    },
                    ignored -> {}));

    assertSame(replayFailure, thrown);
    assertArrayEquals(new Throwable[] {sequentialReader.finishFailure}, thrown.getSuppressed());
    assertEquals(1, sequentialReader.finishMergeCalls);
  }

  @Test
  public void testFinishFailurePropagatesAfterSuccessfulReplay() {
    RecordingReader sequentialReader = new RecordingReader();
    sequentialReader.finishFailure = new IOException("finish failed");
    RecordingReader searchReader = new RecordingReader();
    searchReader.mergeReader = sequentialReader;

    RecordingResult result = new RecordingResult();
    IOException thrown =
        expectThrows(
            IOException.class,
            () ->
                CuVS2510GPUVectorsReader.withSequentialFlatVectors(
                    searchReader, false, reader -> result, RecordingResult::close));

    assertSame(sequentialReader.finishFailure, thrown);
    assertTrue(result.closed);
    assertEquals(1, sequentialReader.finishMergeCalls);
  }

  @Test
  public void testAlreadySequentialContextDoesNotResetAdvice() throws Exception {
    RecordingReader sequentialContextReader = new RecordingReader();

    String result =
        CuVS2510GPUVectorsReader.withSequentialFlatVectors(
            sequentialContextReader,
            true,
            reader -> {
              assertSame(sequentialContextReader, reader);
              return "hydrated";
            },
            ignored -> {});

    assertEquals("hydrated", result);
    assertEquals(0, sequentialContextReader.getMergeInstanceCalls);
    assertEquals(0, sequentialContextReader.finishMergeCalls);
  }

  private static final class RecordingReader extends FlatVectorsReader {
    private FlatVectorsReader mergeReader = this;
    private IOException finishFailure;
    private int getMergeInstanceCalls;
    private int finishMergeCalls;
    private int closeCalls;

    private RecordingReader() {
      super(DefaultFlatVectorScorer.INSTANCE);
    }

    @Override
    public FlatVectorsReader getMergeInstance() {
      getMergeInstanceCalls++;
      return mergeReader;
    }

    @Override
    public void finishMerge() throws IOException {
      finishMergeCalls++;
      if (finishFailure != null) {
        throw finishFailure;
      }
    }

    @Override
    public void checkIntegrity() {}

    @Override
    public FloatVectorValues getFloatVectorValues(String field) {
      throw new UnsupportedOperationException();
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) {
      throw new UnsupportedOperationException();
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public long ramBytesUsed() {
      return 0;
    }

    @Override
    public void close() {
      closeCalls++;
    }
  }

  private static final class RecordingResult implements AutoCloseable {
    private boolean closed;

    @Override
    public void close() {
      closed = true;
    }
  }
}
