/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.FutureTask;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/** Shared, opt-in CPU executor for acquiring one filter handle per Lucene leaf. */
final class ParallelFilterBitsetBuilder {
  static final String THREADS_PROPERTY = "com.nvidia.cuvs.lucene.filterBuildThreads";
  private static final int QUEUE_CAPACITY = 4096;
  private static final AtomicInteger WORKER_ID = new AtomicInteger();

  private ParallelFilterBitsetBuilder() {}

  private static final class Configuration {
    static final int THREADS = parseThreads(System.getProperty(THREADS_PROPERTY));
    static final ThreadPoolExecutor EXECUTOR =
        THREADS == 1 ? null : newExecutor(THREADS, QUEUE_CAPACITY);
  }

  /** Read once, on the first query that needs a filter; null retains the serial path. */
  static Executor executor() {
    return Configuration.EXECUTOR;
  }

  static int parseThreads(String value) {
    if (value == null) return 1;
    final int threads;
    try {
      threads = Integer.parseInt(value);
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException(THREADS_PROPERTY + " must be an integer from 1 to 8", e);
    }
    if (threads < 1 || threads > 8) {
      throw new IllegalArgumentException(THREADS_PROPERTY + " must be between 1 and 8");
    }
    return threads;
  }

  static ThreadPoolExecutor newExecutor(int threads, int queueCapacity) {
    ThreadPoolExecutor executor =
        new ThreadPoolExecutor(
            threads,
            threads,
            60,
            TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(queueCapacity),
            task -> {
              Thread worker =
                  new Thread(task, "cuvs-lucene-filter-builder-" + WORKER_ID.incrementAndGet());
              worker.setDaemon(true);
              return worker;
            },
            (task, pool) -> {
              // CallerRunsPolicy silently discards tasks after shutdown. Never leave a future
              // incomplete: reject shutdown explicitly, but retain caller-run backpressure when
              // the bounded queue is full.
              if (pool.isShutdown()) {
                throw new RejectedExecutionException("Filter builder executor is shut down");
              }
              task.run();
            });
    executor.allowCoreThreadTimeOut(true);
    return executor;
  }

  /**
   * Transfers acquired values to the caller in leaf order, or releases all acquired values on
   * failure. Null jobs represent unfiltered leaves and retain their position in the result.
   *
   * <p>Never cancel accepted tasks: cancellation can hide an acquired reference or allow a task
   * to keep using a reader after the query returns. Submission failure and interruption therefore
   * drain every accepted task before releasing references and unwinding. The caller's interrupt
   * status is restored after cleanup.
   */
  static <T> List<T> runOrdered(
      Executor executor, List<? extends Callable<T>> jobs, Consumer<T> release) throws IOException {
    Objects.requireNonNull(executor);
    Objects.requireNonNull(release);
    // Allocate bookkeeping before submitting any work that can acquire a resource.
    List<T> results = new ArrayList<>(Collections.nCopies(jobs.size(), null));
    List<FutureTask<T>> futures = new ArrayList<>(jobs.size());
    for (Callable<T> job : jobs) {
      futures.add(job == null ? null : new FutureTask<>(job));
    }
    boolean[] accepted = new boolean[jobs.size()];
    Throwable failure = null;
    boolean interrupted = false;
    try {
      for (int i = 0; i < futures.size(); i++) {
        if (Thread.interrupted()) {
          interrupted = true;
          failure = recordFailure(failure, interruption());
          break;
        }
        FutureTask<T> future = futures.get(i);
        if (future == null) continue;
        try {
          executor.execute(future);
          accepted[i] = true;
        } catch (Throwable t) {
          // A synchronous executor may finish the task before throwing. Preserve its result too.
          accepted[i] = future.isDone();
          failure = recordFailure(failure, t);
          break;
        }
      }
      for (int i = 0; i < futures.size(); i++) {
        if (!accepted[i]) continue;
        FutureTask<T> future = futures.get(i);
        for (; ; ) {
          try {
            results.set(i, future.get());
            break;
          } catch (InterruptedException e) {
            if (!interrupted) failure = recordFailure(failure, interruption());
            interrupted = true;
          } catch (ExecutionException e) {
            failure = recordFailure(failure, e.getCause());
            break;
          }
        }
      }
      if (Thread.interrupted()) {
        if (!interrupted) failure = recordFailure(failure, interruption());
        interrupted = true;
      }
      if (failure != null) {
        for (T result : results) {
          if (result == null) continue;
          try {
            release.accept(result);
          } catch (Throwable t) {
            failure = recordFailure(failure, t);
          }
        }
        if (failure instanceof IOException e) throw e;
        if (failure instanceof RuntimeException e) throw e;
        if (failure instanceof Error e) throw e;
        throw new IOException(failure);
      }
      return results;
    } finally {
      if (interrupted) Thread.currentThread().interrupt();
    }
  }

  private static InterruptedIOException interruption() {
    return new InterruptedIOException("Interrupted during parallel filter construction");
  }

  private static Throwable recordFailure(Throwable failure, Throwable next) {
    if (failure == null) return next;
    if (failure != next) failure.addSuppressed(next);
    return failure;
  }
}
