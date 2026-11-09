/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

/** Preserves Lucene 10.3's optimistic per-leaf collection and wider second-pass searches. */
abstract class CompatSeededKnnCollectorManager implements KnnCollectorManager {
  private final KnnCollectorManager delegate;

  CompatSeededKnnCollectorManager(KnnCollectorManager delegate) {
    this.delegate = delegate;
  }

  abstract KnnCollector seed(KnnCollector collector, LeafReaderContext context) throws IOException;

  @Override
  public KnnCollector newCollector(
      int visitedLimit, KnnSearchStrategy strategy, LeafReaderContext context) throws IOException {
    return seed(delegate.newCollector(visitedLimit, strategy, context), context);
  }

  @Override
  public KnnCollector newOptimisticCollector(
      int visitedLimit, KnnSearchStrategy strategy, LeafReaderContext context, int k)
      throws IOException {
    return seed(delegate.newOptimisticCollector(visitedLimit, strategy, context, k), context);
  }

  @Override
  public boolean isOptimistic() {
    return delegate.isOptimistic();
  }
}
