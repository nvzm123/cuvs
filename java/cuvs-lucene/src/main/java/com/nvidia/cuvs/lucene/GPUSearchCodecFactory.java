/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.codecs.Codec;

/**
 * Stateless, binding-compatible factory for codecs that build and search CAGRA indexes on the GPU.
 *
 * <p>An empty request preserves {@link GPUSearchParams.Builder} defaults, including heuristic mode
 * and 32 writer threads. The response contains a single {@code codec} entry. This factory configures
 * index construction, not the parameters of individual queries.
 *
 * <p>Optional controls are {@code build_strategy} ({@code heuristic} or {@code custom}),
 * {@code graph_degree} (1..512), {@code intermediate_graph_degree} (2..512),
 * {@code graph_build_algo} ({@code nn_descent}), {@code nn_descent_n_iters}
 * (1..100, NN-descent only), {@code writer_threads} (1..512), and {@code build_quality}
 * (0..Integer.MAX_VALUE, default 7, heuristic only). Counts must be {@link Integer}; enum names are
 * lowercase strings.
 * Explicit geometry, algorithm or iteration options infer custom mode. Explicit heuristic
 * mode accepts graph_degree but rejects other custom controls it would ignore. Custom graph degree
 * defaults to 64; intermediate degree defaults to {@code max(128, graph_degree)} and must not be
 * below graph_degree.
 *
 * <p>Build quality is a construction-effort setting, not a percentage or a 0..10 score. Native cuVS
 * describes values below 20 as most practical. Below one million vectors, its current dataset
 * heuristic uses {@code 5 + build_quality} maximum NN-descent iterations. The integer type limit
 * is not a recommended tuning range or a guarantee that native construction will succeed.
 *
 * @since 26.12
 */
public final class GPUSearchCodecFactory
    implements Function<Map<String, Object>, Map<String, Object>> {
  public static final String CODEC_KEY = "codec";

  /**
   * Constructs one GPU CAGRA codec.
   *
   * @param request optional build controls; an empty map preserves defaults
   * @return the configured codec under {@code codec}
   * @throws IllegalArgumentException for null, unknown, invalid or incompatible options
   * @throws IllegalStateException if codec construction or vector-format initialization fails
   */
  @Override
  public Map<String, Object> apply(Map<String, Object> request) {
    GPUSearchParams parameters = CagraBuildOptions.gpu(request);
    final Codec codec;
    try {
      codec = CuVSCodecs.gpuSearch(parameters);
      if (codec.knnVectorsFormat() == null) {
        throw new IllegalStateException("GPU CAGRA codec did not initialize a vector format");
      }
    } catch (IllegalStateException error) {
      throw error;
    } catch (Exception | LinkageError error) {
      throw new IllegalStateException("Could not construct the GPU CAGRA codec", error);
    }
    return Map.of(CODEC_KEY, codec);
  }
}
