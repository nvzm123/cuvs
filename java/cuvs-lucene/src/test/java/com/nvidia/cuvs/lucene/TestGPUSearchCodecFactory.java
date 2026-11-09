/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

/** Codec parameter assertions do not need CUDA or invoke a native build heuristic. */
public class TestGPUSearchCodecFactory extends LuceneTestCase {
  @Test
  @SuppressWarnings({"rawtypes", "unchecked"})
  public void testRejectsNonStringRequestKeys() {
    Map malformedRequest = new HashMap<>();
    malformedRequest.put(42, "fake non-string key");
    var error =
        expectThrows(
            IllegalArgumentException.class,
            () -> new GPUSearchCodecFactory().apply(malformedRequest));
    assertEquals("unsupported build parameter: 42", error.getMessage());
  }

  @Test
  public void testEmptyRequestPreservesGpuDefaults() throws Exception {
    var parameters = codecParameters(new GPUSearchCodecFactory().apply(Map.of()));
    assertEquals(GPUSearchParams.Strategy.HEURISTIC, parameters.getStrategy());
    assertEquals(64, parameters.getGraphdegree());
    assertEquals(128, parameters.getIntermediateGraphDegree());
    assertEquals(32, parameters.getWriterThreads());
    assertEquals(7, parameters.getBuildQuality());
    assertEquals(CagraGraphBuildAlgo.NN_DESCENT, parameters.getCagraGraphBuildAlgo());
    assertEquals(20, parameters.getnNDescentNumIterations());
  }

  @Test
  public void testExplicitHeuristicAcceptsGraphDegreeAndBuildQuality() throws Exception {
    var parameters =
        codecParameters(
            new GPUSearchCodecFactory()
                .apply(
                    Map.of(
                        "build_strategy",
                        "heuristic",
                        "graph_degree",
                        24,
                        "build_quality",
                        9,
                        "writer_threads",
                        2)));
    assertEquals(GPUSearchParams.Strategy.HEURISTIC, parameters.getStrategy());
    assertEquals(24, parameters.getGraphdegree());
    assertEquals(9, parameters.getBuildQuality());
    assertEquals(2, parameters.getWriterThreads());
  }

  @Test
  public void testCustomNnDescentControlsReachCagraParameters() throws Exception {
    var parameters =
        codecParameters(
            new GPUSearchCodecFactory()
                .apply(
                    Map.of(
                        "graph_degree",
                        48,
                        "intermediate_graph_degree",
                        96,
                        "nn_descent_n_iters",
                        13,
                        "writer_threads",
                        4)));
    assertEquals(GPUSearchParams.Strategy.CUSTOM, parameters.getStrategy());
    var nativeParameters = CagraIndexParamsFactory.create(parameters, 1000, 16);
    assertEquals(48L, nativeParameters.getGraphDegree());
    assertEquals(96L, nativeParameters.getIntermediateGraphDegree());
    assertEquals(13, nativeParameters.getNNDescentNumIterations());
    assertEquals(4, nativeParameters.getNumWriterThreads());
    assertEquals(CagraGraphBuildAlgo.NN_DESCENT, nativeParameters.getCagraGraphBuildAlgo());
  }

  @Test
  public void testRequestsDoNotLeakConfiguration() throws Exception {
    Function<Map<String, Object>, Map<String, Object>> factory = new GPUSearchCodecFactory();
    var firstRequest = new HashMap<String, Object>(Map.of("graph_build_algo", "nn_descent"));
    firstRequest.put("writer_threads", 5);
    var first = codecParameters(factory.apply(firstRequest));
    firstRequest.put("writer_threads", 9);
    var second = codecParameters(factory.apply(firstRequest));
    var defaults = codecParameters(factory.apply(Map.of()));
    assertEquals(5, first.getWriterThreads());
    assertEquals(9, second.getWriterThreads());
    assertEquals(32, defaults.getWriterThreads());
    assertEquals(GPUSearchParams.Strategy.HEURISTIC, defaults.getStrategy());
    assertEquals(CagraGraphBuildAlgo.NN_DESCENT, defaults.getCagraGraphBuildAlgo());
  }

  @Test
  public void testCustomDegreeRaisesDefaultIntermediateDegree() throws Exception {
    var parameters =
        codecParameters(new GPUSearchCodecFactory().apply(Map.of("graph_degree", 256)));
    assertEquals(256, parameters.getGraphdegree());
    assertEquals(256, parameters.getIntermediateGraphDegree());
    assertEquals(GPUSearchParams.Strategy.CUSTOM, parameters.getStrategy());
  }

  @Test
  public void testRejectsInvalidCommonControlsAndIgnoredCombinations() {
    var factory = new GPUSearchCodecFactory();
    expectThrows(IllegalArgumentException.class, () -> factory.apply(null));
    var nullKey = new HashMap<String, Object>();
    nullKey.put(null, 1);
    expectThrows(IllegalArgumentException.class, () -> factory.apply(nullKey));
    var nullValue = new HashMap<String, Object>();
    nullValue.put("graph_degree", null);
    expectThrows(IllegalArgumentException.class, () -> factory.apply(nullValue));
    for (Map<String, Object> request :
        List.<Map<String, Object>>of(
            Map.of("unknown_build_option", 1),
            Map.of("max_conn", 16),
            Map.of("hnsw_layers", 2),
            Map.of("hnsw_heuristic", "same_graph_footprint"),
            Map.of("graph_degree", 0),
            Map.of("graph_degree", 513),
            Map.of("intermediate_graph_degree", 1),
            Map.of("graph_degree", 16L),
            Map.of("writer_threads", true),
            Map.of("writer_threads", 513),
            Map.of("nn_descent_n_iters", 0),
            Map.of("nn_descent_n_iters", 101),
            Map.of("build_strategy", "CUSTOM"),
            Map.of("graph_build_algo", "unknown"),
            Map.of("build_quality", -1),
            Map.of("build_quality", 7.0),
            Map.of("build_strategy", "heuristic", "intermediate_graph_degree", 128),
            Map.of("build_strategy", "heuristic", "graph_build_algo", "nn_descent"),
            Map.of("graph_degree", 32, "intermediate_graph_degree", 16),
            Map.of("build_strategy", "custom", "build_quality", 7))) {
      expectThrows(IllegalArgumentException.class, () -> factory.apply(request));
    }
  }

  private static GPUSearchParams codecParameters(Map<String, Object> response) throws Exception {
    assertEquals(1, response.size());
    Codec codec = (Codec) response.get(GPUSearchCodecFactory.CODEC_KEY);
    assertEquals(CuVSCodecs.gpuSearch().getClass(), codec.getClass());
    var format = codec.knnVectorsFormat();
    assertTrue(format instanceof CuVS2510GPUVectorsFormat);
    // The authoritative state lives in the returned codec, not in an input echo.
    var field = CuVS2510GPUVectorsFormat.class.getDeclaredField("gpuSearchParams");
    field.setAccessible(true);
    return (GPUSearchParams) field.get(format);
  }
}
