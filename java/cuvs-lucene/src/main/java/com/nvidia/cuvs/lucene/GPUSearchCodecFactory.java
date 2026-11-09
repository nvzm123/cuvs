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
 * {@code graph_build_algo} ({@code nn_descent} or {@code ivf_pq}), {@code nn_descent_n_iters}
 * (1..100, NN-descent only), {@code writer_threads} (1..512), and {@code build_quality}
 * (0..Integer.MAX_VALUE, default 7, heuristic only). Counts must be {@link Integer}; enum names are
 * lowercase strings.
 * Explicit geometry, algorithm, iteration or IVF-PQ options infer custom mode. Explicit heuristic
 * mode accepts graph_degree but rejects other custom controls it would ignore. Custom graph degree
 * defaults to 64; intermediate degree defaults to {@code max(128, graph_degree)} and must not be
 * below graph_degree.
 *
 * <p>Build quality is a construction-effort setting, not a percentage or a 0..10 score. Native cuVS
 * describes values below 20 as most practical. Its current dataset heuristic uses
 * {@code 5 + build_quality} maximum NN-descent iterations below one million vectors, otherwise
 * {@code round(2 + sqrt(n_lists) / 20 + build_quality)} IVF-PQ probes. The integer type limit is not
 * a recommended tuning range or a guarantee that native construction will succeed.
 *
 * <p>IVF-PQ construction controls require explicit {@code graph_build_algo=ivf_pq}: positive Integer
 * {@code n_lists}, {@code n_probes}, {@code kmeans_n_iters}, {@code max_train_points_per_pq_code};
 * nonnegative Integer {@code pq_dim} (0 selects the native heuristic); Integer {@code pq_bits}
 * (4..8); Float or Double {@code kmeans_trainset_fraction} (finite, in (0,1]) and
 * {@code refinement_rate} (finite and greater than 1 after conversion to float); Boolean
 * {@code force_random_rotation}; {@code codebook_kind} ({@code per_subspace} or {@code per_cluster});
 * {@code lut_dtype} ({@code float32}, {@code float16}, {@code uint8}); and
 * {@code internal_distance_dtype} ({@code float32} or {@code float16}).
 * A float16 internal distance requires a float16 or uint8 LUT; an omitted LUT defaults to float32.
 * The product {@code pq_dim * pq_bits} must be a multiple of 8, and explicit n_probes must not exceed n_lists.
 * Omitted n_lists is the floor of the square root of the field's vector count in the segment being
 * built; omitted n_probes is min(32, n_lists). Omitted pq_dim is one quarter of the vector dimension,
 * rounded upward to a multiple of 32. Other custom defaults include pq_bits=8, kmeans_n_iters=10,
 * and kmeans_trainset_fraction=0.25. Explicit pq_dim=0 still selects cuVS's own PQ heuristic.
 * These binding-factory defaults do not change the native HEURISTIC strategy or existing Java
 * parameter-builder defaults. Shape-dependent values resolve when the segment is built.
 * These IVF-PQ search controls construct the CAGRA graph; they do not configure CAGRA queries.
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
