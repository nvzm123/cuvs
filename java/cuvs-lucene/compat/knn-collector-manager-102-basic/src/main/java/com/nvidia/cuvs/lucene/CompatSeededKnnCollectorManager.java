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

/** Seeds Lucene's own collectors without replacing their result queues or visit limits. */
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
}
