/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.*;

import com.nvidia.cuvs.FilterBitsetHandle;
import com.nvidia.cuvs.lucene.FilterBitsetCache.CachedFilterBitset;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.Bits;
import org.junit.Test;

/** CPU-only scheduling, ownership and filter semantics tests; no native handles are created. */
public class TestParallelFilterBitsetBuilder {
  private static final class CountingHandle implements FilterBitsetHandle {
    final AtomicInteger references = new AtomicInteger(1);
    final AtomicInteger releases = new AtomicInteger();
    private final AtomicBoolean initialReleased = new AtomicBoolean();

    @Override
    public boolean tryIncRef() {
      int current;
      do {
        current = references.get();
        if (current == 0) return false;
      } while (!references.compareAndSet(current, current + 1));
      return true;
    }

    @Override
    public void decRef() {
      releases.incrementAndGet();
      if (references.decrementAndGet() < 0) throw new AssertionError("Over-released handle");
    }

    @Override
    public void close() {
      if (initialReleased.compareAndSet(false, true)) decRef();
    }
  }

  private static void await(CountDownLatch latch) throws InterruptedException {
    assertTrue("Latch timed out", latch.await(10, TimeUnit.SECONDS));
  }

  private static void shutdown(ExecutorService executor) throws InterruptedException {
    executor.shutdown();
    assertTrue("Executor did not terminate", executor.awaitTermination(10, TimeUnit.SECONDS));
  }

  @Test
  public void serialDefaultAndThreadBounds() {
    assertEquals(1, ParallelFilterBitsetBuilder.parseThreads(null));
    for (int i = 1; i <= 8; i++) {
      assertEquals(i, ParallelFilterBitsetBuilder.parseThreads(Integer.toString(i)));
    }
    for (String invalid : List.of("0", "-1", "9", "", "eight", "2147483648")) {
      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class,
              () -> ParallelFilterBitsetBuilder.parseThreads(invalid));
      assertTrue(error.getMessage().contains(ParallelFilterBitsetBuilder.THREADS_PROPERTY));
    }
  }

  @Test
  public void completionOrderDoesNotChangeLeafOrderOrNullSlots() throws Exception {
    ExecutorService executor = ParallelFilterBitsetBuilder.newExecutor(3, 10);
    CountDownLatch lastFinished = new CountDownLatch(1);
    try {
      List<Callable<Integer>> jobs =
          Arrays.asList(
              () -> {
                await(lastFinished);
                return 0;
              },
              null,
              () -> {
                await(lastFinished);
                return 2;
              },
              () -> {
                lastFinished.countDown();
                return 3;
              });
      assertEquals(
          Arrays.asList(0, null, 2, 3),
          ParallelFilterBitsetBuilder.runOrdered(
              executor, jobs, ignored -> fail("Unexpected release")));
    } finally {
      lastFinished.countDown();
      shutdown(executor);
    }
  }

  @Test
  public void concurrentQueriesShareBoundedDedicatedBuilders() throws Exception {
    ThreadPoolExecutor builders = ParallelFilterBitsetBuilder.newExecutor(2, 64);
    ExecutorService queries = Executors.newFixedThreadPool(4);
    CountDownLatch bothRunning = new CountDownLatch(2);
    CountDownLatch continueBuilds = new CountDownLatch(1);
    AtomicInteger active = new AtomicInteger();
    AtomicInteger peak = new AtomicInteger();
    Set<String> workerNames = ConcurrentHashMap.newKeySet();
    List<Future<List<Integer>>> results = new ArrayList<>();
    try {
      for (int query = 0; query < 4; query++) {
        results.add(
            queries.submit(
                () -> {
                  List<Callable<Integer>> jobs = new ArrayList<>();
                  for (int leaf = 0; leaf < 6; leaf++) {
                    final int ordinal = leaf;
                    jobs.add(
                        () -> {
                          int current = active.incrementAndGet();
                          peak.accumulateAndGet(current, Math::max);
                          workerNames.add(Thread.currentThread().getName());
                          bothRunning.countDown();
                          try {
                            await(continueBuilds);
                            return ordinal;
                          } finally {
                            active.decrementAndGet();
                          }
                        });
                  }
                  return ParallelFilterBitsetBuilder.runOrdered(builders, jobs, ignored -> {});
                }));
      }
      await(bothRunning);
      assertEquals(2, active.get());
      continueBuilds.countDown();
      for (Future<List<Integer>> result : results) {
        assertEquals(List.of(0, 1, 2, 3, 4, 5), result.get(10, TimeUnit.SECONDS));
      }
      assertEquals(2, peak.get());
      assertEquals(2, workerNames.size());
      assertEquals(0, active.get());
      assertEquals(2, builders.getMaximumPoolSize());
    } finally {
      continueBuilds.countDown();
      shutdown(queries);
      shutdown(builders);
    }
  }

  @Test
  public void fullQueueRunsOnCallerInsteadOfRejecting() throws Exception {
    ThreadPoolExecutor executor = ParallelFilterBitsetBuilder.newExecutor(1, 1);
    CountDownLatch builderRunning = new CountDownLatch(1);
    CountDownLatch releaseBuilder = new CountDownLatch(1);
    Thread caller = Thread.currentThread();
    try {
      executor.execute(
          () -> {
            builderRunning.countDown();
            try {
              await(releaseBuilder);
            } catch (InterruptedException e) {
              throw new AssertionError(e);
            }
          });
      await(builderRunning);
      List<Callable<Integer>> jobs =
          List.of(
              () -> 1,
              () -> {
                assertSame(caller, Thread.currentThread());
                releaseBuilder.countDown();
                return 2;
              });
      assertEquals(
          List.of(1, 2), ParallelFilterBitsetBuilder.runOrdered(executor, jobs, ignored -> {}));
    } finally {
      releaseBuilder.countDown();
      shutdown(executor);
    }
  }

  @Test
  public void shutdownAfterPartialSubmissionDrainsAndReleases() throws Exception {
    ThreadPoolExecutor executor = ParallelFilterBitsetBuilder.newExecutor(1, 4);
    ExecutorService caller = Executors.newSingleThreadExecutor();
    CountDownLatch acquired = new CountDownLatch(1);
    CountDownLatch finishReaderAccess = new CountDownLatch(1);
    AtomicBoolean readerActive = new AtomicBoolean();
    CountingHandle accepted = new CountingHandle();
    AtomicInteger submits = new AtomicInteger();
    Executor shuttingDown =
        task -> {
          executor.execute(task);
          if (submits.incrementAndGet() == 1) executor.shutdown();
        };
    try {
      Future<List<CountingHandle>> result =
          caller.submit(
              () ->
                  ParallelFilterBitsetBuilder.runOrdered(
                      shuttingDown,
                      List.of(
                          () -> {
                            readerActive.set(true);
                            acquired.countDown();
                            try {
                              await(finishReaderAccess);
                              return accepted;
                            } finally {
                              readerActive.set(false);
                            }
                          },
                          () -> {
                            fail("Rejected leaf must never run");
                            return null;
                          }),
                      CountingHandle::decRef));
      await(acquired);
      assertFalse(result.isDone());
      finishReaderAccess.countDown();
      assertTrue(
          assertThrows(ExecutionException.class, () -> result.get(10, TimeUnit.SECONDS)).getCause()
              instanceof RejectedExecutionException);
      assertFalse(readerActive.get());
      assertEquals(0, accepted.references.get());
      assertEquals(1, accepted.releases.get());
    } finally {
      finishReaderAccess.countDown();
      shutdown(caller);
      shutdown(executor);
    }
  }

  @Test
  public void buildFailureDrainsLaterTasksBeforeUnwinding() throws Exception {
    ExecutorService executor = ParallelFilterBitsetBuilder.newExecutor(3, 4);
    ExecutorService caller = Executors.newSingleThreadExecutor();
    CountDownLatch readerAccess = new CountDownLatch(1);
    CountDownLatch stopAccess = new CountDownLatch(1);
    AtomicBoolean readerActive = new AtomicBoolean();
    IOException failure = new IOException("Failed leaf");
    CountingHandle slow = new CountingHandle();
    CountingHandle fast = new CountingHandle();
    try {
      Future<List<CountingHandle>> result =
          caller.submit(
              () ->
                  ParallelFilterBitsetBuilder.runOrdered(
                      executor,
                      List.of(
                          () -> {
                            throw failure;
                          },
                          () -> {
                            readerActive.set(true);
                            readerAccess.countDown();
                            try {
                              await(stopAccess);
                              return slow;
                            } finally {
                              readerActive.set(false);
                            }
                          },
                          () -> fast),
                      CountingHandle::decRef));
      await(readerAccess);
      assertFalse(result.isDone());
      stopAccess.countDown();
      assertSame(
          failure,
          assertThrows(ExecutionException.class, () -> result.get(10, TimeUnit.SECONDS))
              .getCause());
      assertFalse(readerActive.get());
      for (CountingHandle handle : List.of(slow, fast)) {
        assertEquals(0, handle.references.get());
        assertEquals(1, handle.releases.get());
      }
    } finally {
      stopAccess.countDown();
      shutdown(caller);
      shutdown(executor);
    }
  }

  @Test
  public void interruptionDrainsAllTasksAndRestoresCallerStatus() throws Exception {
    ExecutorService executor = ParallelFilterBitsetBuilder.newExecutor(3, 8);
    CountDownLatch allStarted = new CountDownLatch(3);
    CountDownLatch finish = new CountDownLatch(1);
    AtomicInteger readerUsers = new AtomicInteger();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    AtomicBoolean restored = new AtomicBoolean();
    List<CountingHandle> handles =
        List.of(new CountingHandle(), new CountingHandle(), new CountingHandle());
    List<Callable<CountingHandle>> jobs = new ArrayList<>();
    for (CountingHandle handle : handles) {
      jobs.add(
          () -> {
            readerUsers.incrementAndGet();
            allStarted.countDown();
            try {
              await(finish);
              return handle;
            } finally {
              readerUsers.decrementAndGet();
            }
          });
    }
    Thread caller =
        new Thread(
            () -> {
              try {
                ParallelFilterBitsetBuilder.runOrdered(executor, jobs, CountingHandle::decRef);
              } catch (Throwable t) {
                failure.set(t);
              } finally {
                restored.set(Thread.currentThread().isInterrupted());
              }
            },
            "filter-interruption-test");
    try {
      caller.start();
      await(allStarted);
      caller.interrupt();
      assertTrue(caller.isAlive());
      finish.countDown();
      caller.join(10_000);
      assertFalse(caller.isAlive());
      assertTrue(failure.get() instanceof InterruptedIOException);
      assertTrue(restored.get());
      assertEquals(0, readerUsers.get());
      for (CountingHandle handle : handles) {
        assertEquals(0, handle.references.get());
        assertEquals(1, handle.releases.get());
      }
    } finally {
      finish.countDown();
      caller.join(10_000);
      shutdown(executor);
    }
  }

  @Test
  public void alreadyInterruptedCallerDoesNotSubmitWork() {
    AtomicInteger executions = new AtomicInteger();
    Thread.currentThread().interrupt();
    try {
      assertThrows(
          InterruptedIOException.class,
          () ->
              ParallelFilterBitsetBuilder.runOrdered(
                  task -> {
                    executions.incrementAndGet();
                    task.run();
                  },
                  List.of(() -> 1),
                  ignored -> fail("Nothing acquired")));
      assertTrue(Thread.currentThread().isInterrupted());
      assertEquals(0, executions.get());
    } finally {
      Thread.interrupted();
    }
  }

  @Test
  public void interruptionDuringSubmissionStopsAndReleasesCompletedTask() {
    CountingHandle handle = new CountingHandle();
    AtomicInteger submissions = new AtomicInteger();
    Executor executor =
        task -> {
          submissions.incrementAndGet();
          task.run();
          Thread.currentThread().interrupt();
        };
    try {
      assertThrows(
          InterruptedIOException.class,
          () ->
              ParallelFilterBitsetBuilder.runOrdered(
                  executor,
                  List.of(
                      () -> handle,
                      () -> {
                        fail("Interrupted caller must not submit another leaf");
                        return null;
                      }),
                  CountingHandle::decRef));
      assertTrue(Thread.currentThread().isInterrupted());
      assertEquals(1, submissions.get());
      assertEquals(0, handle.references.get());
      assertEquals(1, handle.releases.get());
    } finally {
      Thread.interrupted();
    }
  }

  @Test
  public void singleFlightAndCachedReferencesSurviveParallelFailure() throws Exception {
    FilterBitsetCache cache = new FilterBitsetCache(new FilterBitsetCacheConfig(true, 1024));
    ExecutorService executor = ParallelFilterBitsetBuilder.newExecutor(3, 8);
    CountDownLatch bothAcquiring = new CountDownLatch(2);
    CountingHandle handle = new CountingHandle();
    AtomicInteger builds = new AtomicInteger();
    Object readerKey = new Object();
    IOException failure = new IOException("Other leaf failed");
    Callable<CachedFilterBitset> acquire =
        () -> {
          bothAcquiring.countDown();
          return cache.acquire(
              null,
              readerKey,
              "vector",
              64,
              () -> {
                builds.incrementAndGet();
                try {
                  await(bothAcquiring);
                } catch (InterruptedException e) {
                  throw new IOException(e);
                }
                return new CachedFilterBitset(handle, 7);
              });
        };
    try {
      assertSame(
          failure,
          assertThrows(
              IOException.class,
              () ->
                  ParallelFilterBitsetBuilder.runOrdered(
                      executor,
                      List.of(
                          acquire,
                          acquire,
                          () -> {
                            throw failure;
                          }),
                      cached -> cached.handle().decRef())));
      assertEquals(1, builds.get());
      assertEquals(1, handle.references.get()); // Only the cache-owned reference remains.
      assertEquals(2, handle.releases.get());
      List<CachedFilterBitset> hits =
          ParallelFilterBitsetBuilder.runOrdered(
              executor, List.of(acquire, acquire), cached -> cached.handle().decRef());
      assertSame(handle, hits.get(0).handle());
      assertSame(handle, hits.get(1).handle());
      assertEquals(7, hits.get(0).cardinality());
      assertEquals(3, handle.references.get());
      for (CachedFilterBitset hit : hits) hit.handle().decRef();
      assertEquals(1, builds.get());
      assertEquals(1, handle.references.get());
    } finally {
      shutdown(executor);
      cache.clear();
    }
    assertEquals(0, handle.references.get());
  }

  @Test
  public void additionalFailuresAreSuppressedAndCleanupContinues() {
    IOException failure = new IOException("First failure");
    IllegalStateException other = new IllegalStateException("Other failure");
    IllegalArgumentException releaseFailure = new IllegalArgumentException("Release failure");
    List<Integer> releases = new ArrayList<>();
    IOException actual =
        assertThrows(
            IOException.class,
            () ->
                ParallelFilterBitsetBuilder.<Integer>runOrdered(
                    Runnable::run,
                    List.of(
                        () -> {
                          throw failure;
                        },
                        () -> 1,
                        () -> {
                          throw other;
                        },
                        () -> 2),
                    value -> {
                      releases.add(value);
                      if (value == 1) throw releaseFailure;
                    }));
    assertSame(failure, actual);
    assertArrayEquals(new Throwable[] {other, releaseFailure}, actual.getSuppressed());
    assertEquals(List.of(1, 2), releases);
  }

  @Test
  public void completedSynchronousTaskIsReleasedIfExecutorThrowsAfterRunning() {
    CountingHandle handle = new CountingHandle();
    IllegalStateException failure = new IllegalStateException("Execute failed after completion");
    assertSame(
        failure,
        assertThrows(
            IllegalStateException.class,
            () ->
                ParallelFilterBitsetBuilder.runOrdered(
                    task -> {
                      task.run();
                      throw failure;
                    },
                    List.of(() -> handle),
                    CountingHandle::decRef)));
    assertEquals(0, handle.references.get());
    assertEquals(1, handle.releases.get());
  }

  @Test
  public void taskErrorsPropagateAfterReleasingAcquiredHandles() {
    CountingHandle handle = new CountingHandle();
    AssertionError failure = new AssertionError("Leaf error");
    assertSame(
        failure,
        assertThrows(
            AssertionError.class,
            () ->
                ParallelFilterBitsetBuilder.runOrdered(
                    Runnable::run,
                    List.of(
                        () -> handle,
                        () -> {
                          throw failure;
                        }),
                    CountingHandle::decRef)));
    assertEquals(0, handle.references.get());
  }

  private record PackedMask(long word, int cardinality) {}

  /** Runs production filter evaluation and real ordinal mapping on CPU Lucene leaves. */
  @Test
  public void parallelLeafMasksPreserveSparseOrdinalsDeletesAndNoMatches() throws Exception {
    Method evaluate =
        GPUKnnFloatVectorQuery.class.getDeclaredMethod(
            "evalFilter", Weight.class, LeafReaderContext.class, Bits.class);
    evaluate.setAccessible(true);
    ExecutorService executor = ParallelFilterBitsetBuilder.newExecutor(3, 8);
    try (ByteBuffersDirectory directory = new ByteBuffersDirectory()) {
      try (IndexWriter writer =
          new IndexWriter(
              directory,
              new IndexWriterConfig()
                  .setCodec(new Lucene101Codec())
                  .setMergePolicy(NoMergePolicy.INSTANCE))) {
        for (int leaf = 0; leaf < 3; leaf++) {
          for (int doc = 0; doc < 6; doc++) {
            Document document = new Document();
            document.add(new StringField("id", leaf + "-" + doc, Field.Store.NO));
            document.add(
                new StringField(
                    "category", doc <= 2 || doc == 4 ? "match" : "other", Field.Store.NO));
            if (doc != 1 && doc != 5)
              document.add(new KnnFloatVectorField("vector", new float[] {doc + 1}));
            writer.addDocument(document);
          }
          writer.commit();
        }
        for (int leaf = 0; leaf < 3; leaf++) writer.deleteDocuments(new Term("id", leaf + "-2"));
        writer.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(3, reader.leaves().size());
        IndexSearcher searcher = new IndexSearcher(reader);
        Weight matching =
            searcher.createWeight(
                searcher.rewrite(new TermQuery(new Term("category", "match"))),
                ScoreMode.COMPLETE_NO_SCORES,
                1f);
        Weight noMatches =
            searcher.createWeight(new MatchNoDocsQuery(), ScoreMode.COMPLETE_NO_SCORES, 1f);
        for (Weight weight : Arrays.asList(matching, null, noMatches)) {
          List<Callable<PackedMask>> jobs = new ArrayList<>();
          for (LeafReaderContext leaf : reader.leaves()) {
            jobs.add(
                () -> {
                  Bits liveDocs = leaf.reader().getLiveDocs();
                  assertNotNull(liveDocs);
                  Bits docs =
                      weight == null
                          ? liveDocs
                          : (Bits) evaluate.invoke(null, weight, leaf, liveDocs);
                  FloatVectorValues vectors = leaf.reader().getFloatVectorValues("vector");
                  assertEquals(4, vectors.size());
                  Bits acceptedOrds = vectors.getAcceptOrds(docs);
                  // Reference encoding for comparison only; do not couple this scheduling test
                  // to the private production packing implementation.
                  long word = 0;
                  int cardinality = 0;
                  for (int ordinal = 0; ordinal < vectors.size(); ordinal++) {
                    if (acceptedOrds == null || acceptedOrds.get(ordinal)) {
                      word |= 1L << ordinal;
                      cardinality++;
                    }
                  }
                  return new PackedMask(word, cardinality);
                });
          }
          List<PackedMask> serial = new ArrayList<>();
          for (Callable<PackedMask> job : jobs) serial.add(job.call());
          List<PackedMask> parallel =
              ParallelFilterBitsetBuilder.runOrdered(executor, jobs, ignored -> {});
          assertEquals(serial, parallel);
          PackedMask expected =
              weight == matching
                  ? new PackedMask(9, 2)
                  : weight == null ? new PackedMask(13, 3) : new PackedMask(0, 0);
          assertEquals(List.of(expected, expected, expected), parallel);
        }
      }
    } finally {
      shutdown(executor);
    }
  }
}
