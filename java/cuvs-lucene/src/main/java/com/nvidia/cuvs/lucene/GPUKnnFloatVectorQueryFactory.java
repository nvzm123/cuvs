/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraSearchParams;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import org.apache.lucene.search.Query;

/**
 * Constructs a GPU vector query through types exposed by standard generated Java bindings.
 *
 * <p>The request requires {@code field} ({@link String}), {@code target} ({@code float[]}), and
 * {@code k} ({@link Integer}, 1 through 1024). Optional integer entries are {@code itopk} (defaults
 * to {@code k}), {@code search_width} (1), {@code max_iterations} (0, automatic), and
 * {@code thread_block_size} (0, automatic). The optional {@code algo} string is one of {@code auto}
 * (default), {@code single_cta}, or {@code multi_cta}. No other entries are accepted.
 *
 * <p>An explicit {@code itopk} must be at least {@code k}; this bridge rejects a request that the
 * query would otherwise silently raise to {@code k}. Other search-parameter limits are enforced by
 * {@link GPUKnnFloatVectorQuery}. In particular, SINGLE_CTA supports at most 512 intermediate
 * results. The 1024 result-count limit is distinct from the {@code itopk} limit: the latter's Java
 * integer range is only representational, not a promise that native CAGRA can execute every
 * accepted combination. Native limits also depend on the index and other search parameters.
 *
 * <p>The factory is stateless and returns the base Lucene {@link Query} type, so callers can use
 * {@link Function#apply(Object)} without generating wrappers for this class or its constructor.
 * Query construction does not allocate native resources or execute a GPU search.
 *
 * @since 26.12
 */
public final class GPUKnnFloatVectorQueryFactory implements Function<Map<String, Object>, Query> {
  public static final String FIELD_KEY = "field";
  public static final String TARGET_KEY = "target";
  public static final String K_KEY = "k";
  public static final String ITOPK_KEY = "itopk";
  public static final String SEARCH_WIDTH_KEY = "search_width";
  public static final String MAX_ITERATIONS_KEY = "max_iterations";
  public static final String THREAD_BLOCK_SIZE_KEY = "thread_block_size";
  public static final String ALGO_KEY = "algo";

  private static final Set<String> REQUEST_KEYS =
      Set.of(
          FIELD_KEY,
          TARGET_KEY,
          K_KEY,
          ITOPK_KEY,
          SEARCH_WIDTH_KEY,
          MAX_ITERATIONS_KEY,
          THREAD_BLOCK_SIZE_KEY,
          ALGO_KEY);

  /**
   * Creates one unfiltered GPU query from the request.
   *
   * @param request required query arguments and optional CAGRA search parameters
   * @return the configured query, exposed as a base Lucene query
   * @throws IllegalArgumentException if the request has unknown keys, missing required values,
   *     incorrect types, invalid search parameters, or non-finite vector values
   */
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
    String field = requiredValue(request, FIELD_KEY, String.class);
    float[] target = requiredValue(request, TARGET_KEY, float[].class);
    int k = requiredValue(request, K_KEY, Integer.class);
    if (k < 1 || k > 1024) {
      throw new IllegalArgumentException("k must be in range [1, 1024], but was: " + k);
    }
    int iTopK = optionalInteger(request, ITOPK_KEY, k);
    if (iTopK < k) {
      throw new IllegalArgumentException("itopk must be at least k (" + k + "), but was: " + iTopK);
    }
    // This is the CagraSearchParams.Builder/native default, not a query-batching control.
    int searchWidth = optionalInteger(request, SEARCH_WIDTH_KEY, 1);
    int maxIterations = optionalInteger(request, MAX_ITERATIONS_KEY, 0);
    int threadBlockSize = optionalInteger(request, THREAD_BLOCK_SIZE_KEY, 0);
    String algo =
        request.containsKey(ALGO_KEY) ? requiredValue(request, ALGO_KEY, String.class) : "auto";
    CagraSearchParams.SearchAlgo searchAlgo =
        switch (algo) {
          case "auto" -> CagraSearchParams.SearchAlgo.AUTO;
          case "single_cta" -> CagraSearchParams.SearchAlgo.SINGLE_CTA;
          case "multi_cta" -> CagraSearchParams.SearchAlgo.MULTI_CTA;
          default ->
              throw new IllegalArgumentException("algo must be auto, single_cta, or multi_cta");
        };

    return new GPUKnnFloatVectorQuery(
        field, target, k, null, iTopK, searchWidth, threadBlockSize, maxIterations, searchAlgo);
  }

  private static int optionalInteger(Map<String, Object> request, String key, int defaultValue) {
    return request.containsKey(key) ? requiredValue(request, key, Integer.class) : defaultValue;
  }

  private static <T> T requiredValue(Map<String, Object> request, String key, Class<T> type) {
    Object value = request.get(key);
    if (!type.isInstance(value)) {
      throw new IllegalArgumentException(key + " must have type " + type.getSimpleName());
    }
    return type.cast(value);
  }
}
