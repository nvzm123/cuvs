/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestPostIngestFlushCoordinator extends LuceneTestCase {

  public void testFlatAndGraphTasksOverlap() throws Exception {
    CountDownLatch flatStarted = new CountDownLatch(1);
    CountDownLatch graphStarted = new CountDownLatch(1);
    AtomicBoolean flatFinished = new AtomicBoolean();

    PostIngestFlushCoordinator.runOverlapped(
        () -> {
          flatStarted.countDown();
          assertTrue(graphStarted.await(10, TimeUnit.SECONDS));
          flatFinished.set(true);
        },
        () -> {
          graphStarted.countDown();
          assertTrue(flatStarted.await(10, TimeUnit.SECONDS));
        });

    assertTrue(flatFinished.get());
  }

  public void testOverlappedOutputMatchesSerialOutput() throws Exception {
    List<float[]> vectors =
        List.of(new float[] {1.0f, 2.0f}, new float[] {3.0f, 4.0f}, new float[] {5.0f, 6.0f});
    EncodedOutput serial = encodeSerially(vectors);
    EncodedOutput overlapped = encodeOverlapped(vectors);

    assertArrayEquals(serial.flat(), overlapped.flat());
    assertArrayEquals(serial.graph(), overlapped.graph());
  }

  public void testFlatFailureIsPropagatedAfterGraphCompletes() throws Exception {
    IOException flatFailure = new IOException("flat flush failed");
    CountDownLatch flatFailed = new CountDownLatch(1);
    AtomicBoolean graphFinished = new AtomicBoolean();

    IOException thrown =
        expectThrows(
            IOException.class,
            () ->
                PostIngestFlushCoordinator.runOverlapped(
                    () -> {
                      flatFailed.countDown();
                      throw flatFailure;
                    },
                    () -> {
                      assertTrue(flatFailed.await(10, TimeUnit.SECONDS));
                      graphFinished.set(true);
                    }));

    assertSame(flatFailure, thrown);
    assertTrue(graphFinished.get());
  }

  public void testGraphFailureCancelsAndJoinsFlatWorker() throws Exception {
    IOException graphFailure = new IOException("graph build failed");
    IOException flatFailure = new IOException("flat flush failed after cancellation");
    CountDownLatch flatStarted = new CountDownLatch(1);
    CountDownLatch flatInterrupted = new CountDownLatch(1);
    CountDownLatch releaseFlat = new CountDownLatch(1);
    AtomicBoolean flatExited = new AtomicBoolean();
    AtomicBoolean returnedAfterFlatExit = new AtomicBoolean();
    AtomicReference<Throwable> observed = new AtomicReference<>();
    Thread caller =
        Thread.ofPlatform()
            .name("post-ingest-graph-failure-test")
            .unstarted(
                () -> {
                  try {
                    PostIngestFlushCoordinator.runOverlapped(
                        () -> {
                          flatStarted.countDown();
                          try {
                            new CountDownLatch(1).await();
                            fail("flat worker was not cancelled");
                          } catch (InterruptedException expected) {
                            flatInterrupted.countDown();
                            awaitUninterruptibly(releaseFlat);
                            throw flatFailure;
                          } finally {
                            flatExited.set(true);
                          }
                        },
                        () -> {
                          assertTrue(flatStarted.await(10, TimeUnit.SECONDS));
                          throw graphFailure;
                        });
                    fail("expected graph failure");
                  } catch (Throwable thrown) {
                    observed.set(thrown);
                  } finally {
                    returnedAfterFlatExit.set(flatExited.get());
                  }
                });

    caller.start();
    try {
      assertTrue(flatInterrupted.await(10, TimeUnit.SECONDS));
      assertTrue("coordinator returned before flat worker exited", caller.isAlive());
      assertFalse(flatExited.get());
    } finally {
      releaseFlat.countDown();
      caller.join(10_000L);
      if (caller.isAlive()) {
        caller.interrupt();
      }
    }

    assertFalse("overlap caller did not terminate", caller.isAlive());
    assertSame(graphFailure, observed.get());
    assertTrue(flatExited.get());
    assertTrue(returnedAfterFlatExit.get());
    assertTrue(Arrays.asList(graphFailure.getSuppressed()).contains(flatFailure));
  }

  public void testFlatErrorTakesPrecedenceOverGraphException() throws Exception {
    IOException graphFailure = new IOException("graph build failed");
    InternalError flatFailure = new InternalError("flat flush failed fatally");
    CountDownLatch flatStarted = new CountDownLatch(1);

    InternalError thrown =
        expectThrows(
            InternalError.class,
            () ->
                PostIngestFlushCoordinator.runOverlapped(
                    () -> {
                      flatStarted.countDown();
                      try {
                        new CountDownLatch(1).await();
                        fail("flat worker was not cancelled");
                      } catch (InterruptedException expected) {
                        throw flatFailure;
                      }
                    },
                    () -> {
                      assertTrue(flatStarted.await(10, TimeUnit.SECONDS));
                      throw graphFailure;
                    }));

    assertSame(flatFailure, thrown);
    assertTrue(Arrays.asList(flatFailure.getSuppressed()).contains(graphFailure));
  }

  public void testGraphErrorRemainsPrimaryWhenFlatTaskAlsoFails() throws Exception {
    InternalError graphFailure = new InternalError("graph build failed fatally");
    IOException flatFailure = new IOException("flat flush failed after cancellation");
    CountDownLatch flatStarted = new CountDownLatch(1);

    InternalError thrown =
        expectThrows(
            InternalError.class,
            () ->
                PostIngestFlushCoordinator.runOverlapped(
                    () -> {
                      flatStarted.countDown();
                      try {
                        new CountDownLatch(1).await();
                        fail("flat worker was not cancelled");
                      } catch (InterruptedException expected) {
                        throw flatFailure;
                      }
                    },
                    () -> {
                      assertTrue(flatStarted.await(10, TimeUnit.SECONDS));
                      throw graphFailure;
                    }));

    assertSame(graphFailure, thrown);
    assertTrue(Arrays.asList(graphFailure.getSuppressed()).contains(flatFailure));
  }

  public void testCallerInterruptionIsPreserved() throws Exception {
    AtomicReference<Throwable> failure = new AtomicReference<>();
    AtomicBoolean interruptPreserved = new AtomicBoolean();
    AtomicBoolean graphRan = new AtomicBoolean();
    Thread caller =
        Thread.ofPlatform()
            .name("post-ingest-flush-interruption-test")
            .unstarted(
                () -> {
                  Thread.currentThread().interrupt();
                  try {
                    PostIngestFlushCoordinator.runOverlapped(() -> {}, () -> graphRan.set(true));
                    fail("expected interruption failure");
                  } catch (Throwable thrown) {
                    failure.set(thrown);
                    interruptPreserved.set(Thread.currentThread().isInterrupted());
                  }
                });

    caller.start();
    caller.join(10_000L);
    assertFalse("interrupted caller did not terminate", caller.isAlive());
    assertTrue(failure.get() instanceof InterruptedIOException);
    assertFalse(graphRan.get());
    assertTrue(interruptPreserved.get());
  }

  public void testInterruptionWhileAwaitingFlatWorkerCancelsAndJoinsIt() throws Exception {
    CountDownLatch flatStarted = new CountDownLatch(1);
    CountDownLatch graphFinished = new CountDownLatch(1);
    CountDownLatch releaseFlat = new CountDownLatch(1);
    AtomicBoolean flatInterrupted = new AtomicBoolean();
    AtomicBoolean flatExited = new AtomicBoolean();
    AtomicBoolean interruptPreserved = new AtomicBoolean();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    Thread caller =
        Thread.ofPlatform()
            .name("post-ingest-await-interruption-test")
            .unstarted(
                () -> {
                  try {
                    PostIngestFlushCoordinator.runOverlapped(
                        () -> {
                          flatStarted.countDown();
                          try {
                            releaseFlat.await();
                          } catch (InterruptedException expected) {
                            flatInterrupted.set(true);
                          } finally {
                            flatExited.set(true);
                          }
                        },
                        () -> {
                          assertTrue(flatStarted.await(10, TimeUnit.SECONDS));
                          graphFinished.countDown();
                        });
                    fail("expected interruption failure");
                  } catch (Throwable thrown) {
                    failure.set(thrown);
                    interruptPreserved.set(Thread.currentThread().isInterrupted());
                  }
                });

    caller.start();
    try {
      assertTrue(graphFinished.await(10, TimeUnit.SECONDS));
      caller.interrupt();
      caller.join(10_000L);
    } finally {
      releaseFlat.countDown();
      if (caller.isAlive()) {
        caller.interrupt();
      }
    }

    assertFalse("interrupted overlap caller did not terminate", caller.isAlive());
    assertTrue(failure.get() instanceof InterruptedIOException);
    assertTrue(flatInterrupted.get());
    assertTrue(flatExited.get());
    assertTrue(interruptPreserved.get());
  }

  private static EncodedOutput encodeSerially(List<float[]> vectors) throws IOException {
    ByteArrayOutputStream flat = new ByteArrayOutputStream();
    ByteArrayOutputStream graph = new ByteArrayOutputStream();
    writeFlat(vectors, flat);
    writeGraph(vectors, graph);
    return new EncodedOutput(flat.toByteArray(), graph.toByteArray());
  }

  private static EncodedOutput encodeOverlapped(List<float[]> vectors) throws IOException {
    ByteArrayOutputStream flat = new ByteArrayOutputStream();
    ByteArrayOutputStream graph = new ByteArrayOutputStream();
    PostIngestFlushCoordinator.runOverlapped(
        () -> writeFlat(vectors, flat), () -> writeGraph(vectors, graph));
    return new EncodedOutput(flat.toByteArray(), graph.toByteArray());
  }

  private static void writeFlat(List<float[]> vectors, ByteArrayOutputStream output)
      throws IOException {
    for (float[] vector : vectors) {
      for (float value : vector) {
        output.write(Float.floatToRawIntBits(value));
      }
    }
  }

  private static void writeGraph(List<float[]> vectors, ByteArrayOutputStream output)
      throws IOException {
    for (int ordinal = 0; ordinal < vectors.size(); ordinal++) {
      output.write(ordinal);
      output.write(vectors.get(ordinal).length);
    }
  }

  private static void awaitUninterruptibly(CountDownLatch latch) {
    boolean interrupted = false;
    while (latch.getCount() != 0L) {
      try {
        latch.await();
      } catch (InterruptedException expected) {
        interrupted = true;
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private record EncodedOutput(byte[] flat, byte[] graph) {}
}
