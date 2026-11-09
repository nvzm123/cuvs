/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import java.util.Map;
import java.util.Set;

/** Request-local validation shared by the binding-compatible codec factories. */
final class CagraBuildOptions {
  private static final Set<String> COMMON_KEYS =
      Set.of(
          "build_strategy",
          "graph_degree",
          "intermediate_graph_degree",
          "graph_build_algo",
          "nn_descent_n_iters",
          "writer_threads");
  private static final Set<String> IVF_PQ_KEYS =
      Set.of(
          "n_lists",
          "n_probes",
          "kmeans_n_iters",
          "kmeans_trainset_fraction",
          "pq_dim",
          "pq_bits",
          "refinement_rate",
          "max_train_points_per_pq_code",
          "codebook_kind",
          "force_random_rotation",
          "lut_dtype",
          "internal_distance_dtype");
  private static final Set<String> HNSW_KEYS = Set.of("max_conn", "beam_width", "hnsw_layers");
  private static final Set<String> CUSTOM_KEYS =
      Set.of("graph_degree", "intermediate_graph_degree", "graph_build_algo", "nn_descent_n_iters");

  private final Map<String, Object> request;
  private final boolean custom;
  private final CagraGraphBuildAlgo algorithm;

  private CagraBuildOptions(Map<String, Object> request, boolean hnsw) {
    if (request == null) {
      throw new IllegalArgumentException("request must not be null");
    }
    this.request = request;
    for (Object rawKey : request.keySet()) {
      if (!(rawKey instanceof String key)
          || !COMMON_KEYS.contains(key)
              && !IVF_PQ_KEYS.contains(key)
              && !(hnsw ? HNSW_KEYS.contains(key) : "build_quality".equals(key))) {
        throw new IllegalArgumentException("unsupported build parameter: " + rawKey);
      }
    }
    boolean hasCustom =
        request.keySet().stream()
            .anyMatch(key -> CUSTOM_KEYS.contains(key) || IVF_PQ_KEYS.contains(key));
    custom =
        choice(request, "build_strategy", hasCustom ? "custom" : "heuristic", "heuristic", "custom")
            .equals("custom");
    if (!custom) {
      for (String key : request.keySet()) {
        // GPU's dataset heuristic consumes graph_degree; the HNSW heuristic derives it from M.
        if (IVF_PQ_KEYS.contains(key)
            || CUSTOM_KEYS.contains(key) && (hnsw || !key.equals("graph_degree"))) {
          throw new IllegalArgumentException(key + " requires build_strategy=custom");
        }
      }
    } else if (request.containsKey("build_quality")) {
      throw new IllegalArgumentException("build_quality requires build_strategy=heuristic");
    }
    algorithm =
        choice(request, "graph_build_algo", "nn_descent", "nn_descent", "ivf_pq").equals("ivf_pq")
            ? CagraGraphBuildAlgo.IVF_PQ
            : CagraGraphBuildAlgo.NN_DESCENT;
    for (String key : IVF_PQ_KEYS) {
      if (request.containsKey(key) && algorithm != CagraGraphBuildAlgo.IVF_PQ) {
        throw new IllegalArgumentException(key + " requires explicit graph_build_algo=ivf_pq");
      }
    }
    if (request.containsKey("nn_descent_n_iters") && algorithm != CagraGraphBuildAlgo.NN_DESCENT) {
      throw new IllegalArgumentException("nn_descent_n_iters requires graph_build_algo=nn_descent");
    }
  }

  static AcceleratedHNSWParams accelerated(
      Map<String, Object> request, int maxConn, int beamWidth) {
    var options = new CagraBuildOptions(request, true);
    var builder =
        new AcceleratedHNSWParams.Builder()
            .withMaxConn(maxConn)
            .withBeamWidth(beamWidth)
            .withWriterThreads(
                integer(
                    request,
                    "writer_threads",
                    AcceleratedHNSWParams.DEFAULT_WRITER_THREADS,
                    AcceleratedHNSWParams.MIN_WRITER_THREADS,
                    AcceleratedHNSWParams.MAX_WRITER_THREADS))
            .withHNSWLayer(
                integer(
                    request,
                    "hnsw_layers",
                    AcceleratedHNSWParams.DEFAULT_HNSW_LAYERS,
                    AcceleratedHNSWParams.MIN_HNSW_LAYERS,
                    AcceleratedHNSWParams.MAX_HNSW_LAYERS));
    if (options.custom) {
      int graph =
          integer(
              request,
              "graph_degree",
              Math.min(AcceleratedHNSWParams.DEFAULT_GRAPH_DEGREE, 2 * maxConn),
              AcceleratedHNSWParams.MIN_GRAPH_DEG,
              AcceleratedHNSWParams.MAX_GRAPH_DEG);
      if (graph > 2 * maxConn) {
        throw new IllegalArgumentException("graph_degree must not exceed 2 * max_conn");
      }
      int intermediate =
          options.intermediate(
              graph,
              AcceleratedHNSWParams.DEFAULT_INT_GRAPH_DEGREE,
              AcceleratedHNSWParams.MIN_INT_GRAPH_DEG,
              AcceleratedHNSWParams.MAX_INT_GRAPH_DEG);
      builder
          .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
          .withGraphDegree(graph)
          .withIntermediateGraphDegree(intermediate)
          .withCagraGraphBuildAlgo(options.algorithm)
          .withNNDescentNumIterations(
              integer(
                  request,
                  "nn_descent_n_iters",
                  AcceleratedHNSWParams.DEFAULT_NN_DESCENT_NUM_ITERATIONS,
                  AcceleratedHNSWParams.MIN_NN_DESCENT_NUM_ITERATIONS,
                  AcceleratedHNSWParams.MAX_NN_DESCENT_NUM_ITERATIONS));
      if (options.algorithm == CagraGraphBuildAlgo.IVF_PQ) {
        builder.withIvfPqBuildOptions(IvfPqBuildOptions.parse(request));
      }
    }
    return builder.build();
  }

  static GPUSearchParams gpu(Map<String, Object> request) {
    var options = new CagraBuildOptions(request, false);
    int graph =
        integer(
            request,
            "graph_degree",
            GPUSearchParams.DEFAULT_GRAPH_DEGREE,
            GPUSearchParams.MIN_GRAPH_DEG,
            GPUSearchParams.MAX_GRAPH_DEG);
    var builder =
        new GPUSearchParams.Builder()
            .withGraphDegree(graph)
            .withWriterThreads(
                integer(
                    request,
                    "writer_threads",
                    GPUSearchParams.DEFAULT_WRITER_THREADS,
                    GPUSearchParams.MIN_WRITER_THREADS,
                    GPUSearchParams.MAX_WRITER_THREADS));
    if (options.custom) {
      builder
          .withStrategy(GPUSearchParams.Strategy.CUSTOM)
          .withIntermediateGraphDegree(
              options.intermediate(
                  graph,
                  GPUSearchParams.DEFAULT_INT_GRAPH_DEGREE,
                  GPUSearchParams.MIN_INT_GRAPH_DEG,
                  GPUSearchParams.MAX_INT_GRAPH_DEG))
          .withCagraGraphBuildAlgo(options.algorithm)
          .withNNDescentNumIterations(
              integer(
                  request,
                  "nn_descent_n_iters",
                  GPUSearchParams.DEFAULT_NN_DESCENT_NUM_ITERATIONS,
                  GPUSearchParams.MIN_NN_DESCENT_NUM_ITERATIONS,
                  GPUSearchParams.MAX_NN_DESCENT_NUM_ITERATIONS));
      if (options.algorithm == CagraGraphBuildAlgo.IVF_PQ) {
        builder.withIvfPqBuildOptions(IvfPqBuildOptions.parse(request));
      }
    } else {
      builder.withBuildQuality(
          integer(
              request,
              "build_quality",
              GPUSearchParams.DEFAULT_BUILD_QUALITY,
              GPUSearchParams.MIN_BUILD_QUALITY,
              Integer.MAX_VALUE));
    }
    return builder.build();
  }

  private int intermediate(int graph, int defaultValue, int minimum, int maximum) {
    int intermediate =
        integer(
            request, "intermediate_graph_degree", Math.max(defaultValue, graph), minimum, maximum);
    if (intermediate < graph) {
      throw new IllegalArgumentException("intermediate_graph_degree must be at least graph_degree");
    }
    return intermediate;
  }

  static int requiredInteger(Map<String, Object> values, String key, int minimum, int maximum) {
    if (values == null) {
      throw new IllegalArgumentException("request must not be null");
    }
    Object value = values.get(key);
    if (!(value instanceof Integer integer)) {
      throw new IllegalArgumentException(key + " must have type Integer");
    }
    if (integer < minimum || integer > maximum) {
      throw new IllegalArgumentException(
          key + " must be in range [" + minimum + ", " + maximum + "], but was: " + integer);
    }
    return integer;
  }

  static int integer(
      Map<String, Object> values, String key, int defaultValue, int minimum, int maximum) {
    return values.containsKey(key) ? requiredInteger(values, key, minimum, maximum) : defaultValue;
  }

  static double floating(Map<String, Object> values, String key, double defaultValue) {
    if (!values.containsKey(key)) {
      return defaultValue;
    }
    Object value = values.get(key);
    if (!(value instanceof Float || value instanceof Double)
        || !Double.isFinite(((Number) value).doubleValue())) {
      throw new IllegalArgumentException(key + " must have type Float or Double and be finite");
    }
    return ((Number) value).doubleValue();
  }

  static String choice(
      Map<String, Object> values, String key, String defaultValue, String... choices) {
    if (!values.containsKey(key)) {
      return defaultValue;
    }
    Object value = values.get(key);
    for (String choice : choices) {
      if (choice.equals(value)) {
        return choice;
      }
    }
    throw new IllegalArgumentException(key + " must be one of " + String.join(", ", choices));
  }
}
