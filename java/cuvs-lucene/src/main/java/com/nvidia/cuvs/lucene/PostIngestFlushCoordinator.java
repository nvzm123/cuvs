/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/** Coordinates independent flat-vector and graph outputs after document ingestion is complete. */
final class PostIngestFlushCoordinator {

  private PostIngestFlushCoordinator() {}

  /** Runs the flat-vector output on a worker while the caller builds the graph. */
  static void runOverlapped(FlushTask flatWrite, FlushTask graphWrite) throws IOException {
    ExecutorService executor =
        Executors.newSingleThreadExecutor(
            task -> Thread.ofPlatform().name("cuvs-flat-vector-flush").unstarted(task));
    Future<Void> flatFuture = null;
    Throwable failure = null;
    boolean interrupted = false;

    try {
      flatFuture = executor.submit(() -> runTask(flatWrite));

      if (Thread.interrupted()) {
        interrupted = true;
        failure = interruption("Interrupted before CAGRA graph construction", null);
      } else {
        try {
          graphWrite.run();
        } catch (InterruptedException graphInterruption) {
          interrupted = true;
          failure = interruption("Interrupted during CAGRA graph construction", graphInterruption);
        } catch (Throwable graphFailure) {
          failure = graphFailure;
        }
      }

      if (Thread.interrupted()) {
        interrupted = true;
        failure = combine(failure, interruption("Interrupted after graph construction", null));
      }
    } catch (Throwable orchestrationFailure) {
      failure = combine(failure, orchestrationFailure);
    } finally {
      if (failure != null || interrupted) {
        cancelAndInterrupt(executor, flatFuture);
      } else {
        executor.shutdown();
      }

      while (executor.isTerminated() == false) {
        try {
          executor.awaitTermination(1L, TimeUnit.DAYS);
        } catch (InterruptedException awaitInterruption) {
          interrupted = true;
          failure =
              combine(
                  failure,
                  interruption("Interrupted while awaiting flat-vector output", awaitInterruption));
          cancelAndInterrupt(executor, flatFuture);
        }
      }

      if (flatFuture != null) {
        boolean awaitingCompletion = true;
        while (awaitingCompletion) {
          try {
            flatFuture.get();
            awaitingCompletion = false;
          } catch (InterruptedException getInterruption) {
            interrupted = true;
            failure =
                combine(
                    failure,
                    interruption(
                        "Interrupted while collecting flat-vector output", getInterruption));
          } catch (ExecutionException taskFailure) {
            failure = combine(failure, unwrapTaskFailure(taskFailure.getCause()));
            awaitingCompletion = false;
          } catch (CancellationException taskCancelled) {
            // A queued task is cancelled only after another failure has already made it unsafe to
            // start. The existing failure remains the useful cause.
            if (failure == null) {
              failure = taskCancelled;
            }
            awaitingCompletion = false;
          }
        }
      }

      if (Thread.interrupted()) {
        interrupted = true;
        failure = combine(failure, interruption("Interrupted after flat-vector output", null));
      }
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }

    if (failure != null) {
      throw Utils.handleThrowable(failure);
    }
  }

  private static Void runTask(FlushTask task) throws Exception {
    try {
      task.run();
      return null;
    } catch (Exception | Error failure) {
      throw failure;
    } catch (Throwable unexpected) {
      throw new UnexpectedTaskFailure(unexpected);
    }
  }

  private static void cancelAndInterrupt(ExecutorService executor, Future<?> flatFuture) {
    List<Runnable> neverStarted = executor.shutdownNow();
    if (flatFuture != null && neverStarted.contains(flatFuture)) {
      flatFuture.cancel(false);
    }
  }

  private static Throwable unwrapTaskFailure(Throwable failure) {
    return failure instanceof UnexpectedTaskFailure unexpected ? unexpected.getCause() : failure;
  }

  private static InterruptedIOException interruption(String message, Throwable cause) {
    InterruptedIOException interruption = new InterruptedIOException(message);
    if (cause != null) {
      interruption.initCause(cause);
    }
    return interruption;
  }

  private static Throwable combine(Throwable primary, Throwable additional) {
    if (primary == null) {
      return additional;
    }
    if (additional != null && additional != primary) {
      if (additional instanceof Error && primary instanceof Error == false) {
        additional.addSuppressed(primary);
        return additional;
      }
      primary.addSuppressed(additional);
    }
    return primary;
  }

  private static final class UnexpectedTaskFailure extends Exception {
    private UnexpectedTaskFailure(Throwable cause) {
      super(cause);
    }
  }

  @FunctionalInterface
  interface FlushTask {
    void run() throws Throwable;
  }
}
