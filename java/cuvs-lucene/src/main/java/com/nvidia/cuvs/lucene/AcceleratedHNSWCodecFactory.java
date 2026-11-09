/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.codecs.Codec;

/**
 * Constructs an accelerated HNSW codec from one self-contained parameter request.
 *
 * <p>The standard {@link Function} and {@link Map} types form a narrow bridge for generated Java
 * bindings that do not wrap parameterized constructors. The required entries, max_conn and
 * beam_width, must be {@link Integer} values in the inclusive range 1 through 512.
 * The response contains the configured {@code codec} and the applied values under the
 * same parameter keys. Invalid requests throw {@link IllegalArgumentException}; codec construction
 * or initialization failures throw {@link IllegalStateException}.
 *
 * <p>The factory is stateless. In particular, it does not use JVM system properties, so concurrent
 * callers cannot observe or overwrite one another's configuration.
 *
 * <p>Optional build controls are {@code build_strategy} ({@code heuristic} or {@code custom}),
 * {@code graph_degree} (1..512), {@code intermediate_graph_degree} (2..512),
 * {@code graph_build_algo} ({@code nn_descent}), {@code nn_descent_n_iters}
 * (1..100, NN-descent only), {@code hnsw_layers} (1..99), and {@code writer_threads} (1..512).
 * Counts must be Integer values; enum names are lowercase strings.
 *
 * <p>Supplying geometry, algorithm or iteration controls selects custom mode unless an
 * explicit strategy is supplied. Heuristic mode rejects controls it would ignore. Custom mode
 * defaults to graph degree {@code min(64, 2 * max_conn)} and intermediate degree
 * {@code max(128, graph_degree)}; the graph degree must not exceed {@code 2 * max_conn} or the
 * intermediate degree. Heuristic mode retains the SAME_GRAPH_FOOTPRINT default: cuVS derives
 * {@code graph_degree = 2 * max_conn} and {@code intermediate_graph_degree = 3 * max_conn}.
 * This factory does not expose a heuristic selector.
 *
 * @since 26.12
 */
public final class AcceleratedHNSWCodecFactory
    implements Function<Map<String, Object>, Map<String, Object>> {
  public static final String MAX_CONN_KEY = "max_conn";
  public static final String BEAM_WIDTH_KEY = "beam_width";
  public static final String CODEC_KEY = "codec";

  /**
   * Constructs one codec from the complete request.
   *
   * @param request the required {@code max_conn} and {@code beam_width} entries and optional controls
   * @return the codec and its verified applied parameter values
   * @throws IllegalArgumentException if the request is null, incomplete, or contains unknown,
   *     incorrectly typed, out-of-range or incompatible options
   * @throws IllegalStateException if codec construction or vector-format initialization fails
   */
  @Override
  public Map<String, Object> apply(Map<String, Object> request) {
    int maxConn =
        CagraBuildOptions.requiredInteger(
            request,
            MAX_CONN_KEY,
            AcceleratedHNSWParams.MIN_MAX_CONN,
            AcceleratedHNSWParams.MAX_MAX_CONN);
    int beamWidth =
        CagraBuildOptions.requiredInteger(
            request,
            BEAM_WIDTH_KEY,
            AcceleratedHNSWParams.MIN_BEAM_WIDTH,
            AcceleratedHNSWParams.MAX_BEAM_WIDTH);
    AcceleratedHNSWParams parameters = CagraBuildOptions.accelerated(request, maxConn, beamWidth);

    final Codec codec;
    try {
      codec = CuVSCodecs.acceleratedHNSW(parameters);
      if (codec.knnVectorsFormat() == null) {
        throw new IllegalStateException(
            "Accelerated HNSW codec did not initialize a vector format");
      }
    } catch (IllegalStateException error) {
      throw error;
    } catch (Exception | LinkageError error) {
      throw new IllegalStateException("Could not construct the accelerated HNSW codec", error);
    }

    Map<String, Object> response = new HashMap<>();
    response.put(CODEC_KEY, codec);
    response.put(MAX_CONN_KEY, parameters.getMaxConn());
    response.put(BEAM_WIDTH_KEY, parameters.getBeamWidth());
    return response;
  }
}
