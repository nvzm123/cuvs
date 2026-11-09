/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import org.apache.lucene.search.Query;

/**
 * Binding-compatible factory for {@link BaseLayerKnnFloatVectorQuery}. The request requires {@code
 * field} ({@link String}), {@code target} ({@code float[]}), and {@code k} ({@link Integer}). Optional
 * {@code num_entry_points} is a nonnegative {@link Integer} controlling the sample count; zero
 * disables random entry selection. When omitted, it uses {@link
 * BaseLayerKnnFloatVectorQuery#DEFAULT_NUM_ENTRY_POINTS}. No other entries are accepted. The factory
 * creates unfiltered queries, returns the base Lucene {@link Query} type, and does not load a native
 * library.
 *
 * @since 26.12
 */
public final class BaseLayerKnnFloatVectorQueryFactory
    implements Function<Map<String, Object>, Query> {
  private static final Set<String> REQUEST_KEYS =
      Set.of("field", "target", "k", "num_entry_points");

  /** Creates one unfiltered query from the required arguments and optional sample count. */
  @Override
  public Query apply(Map<String, Object> request) {
    if (request == null) {
      throw new IllegalArgumentException("request must not be null");
    }
    for (Object key : request.keySet()) {
      if (!(key instanceof String) || !REQUEST_KEYS.contains(key)) {
        throw new IllegalArgumentException("unknown query request key: " + key);
      }
    }
    return new BaseLayerKnnFloatVectorQuery(
        requiredValue(request, "field", String.class),
        requiredValue(request, "target", float[].class),
        requiredValue(request, "k", Integer.class),
        null,
        request.containsKey("num_entry_points")
            ? requiredValue(request, "num_entry_points", Integer.class)
            : BaseLayerKnnFloatVectorQuery.DEFAULT_NUM_ENTRY_POINTS);
  }

  private static <T> T requiredValue(Map<String, Object> request, String key, Class<T> type) {
    Object value = request.get(key);
    if (!type.isInstance(value)) {
      throw new IllegalArgumentException(key + " must have type " + type.getSimpleName());
    }
    return type.cast(value);
  }
}
