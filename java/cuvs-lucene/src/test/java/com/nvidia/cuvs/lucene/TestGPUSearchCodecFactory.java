/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CodebookGen;
import com.nvidia.cuvs.CagraIndexParams.CudaDataType;
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
    assertEquals(1024, parameters.getCuVSIvfPqParams().getIndexParams().getnLists());
    assertEquals(20, parameters.getCuVSIvfPqParams().getSearchParams().getnProbes());
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
  public void testAllIvfPqControlsReachCodecAndCagraParameters() throws Exception {
    var request = ivfRequest();
    request.putAll(
        Map.of(
            "n_lists",
            64,
            "n_probes",
            8,
            "kmeans_n_iters",
            15,
            "kmeans_trainset_fraction",
            0.25,
            "pq_dim",
            8,
            "pq_bits",
            6,
            "max_train_points_per_pq_code",
            128,
            "refinement_rate",
            3.5));
    request.putAll(
        Map.of(
            "codebook_kind",
            "per_cluster",
            "force_random_rotation",
            true,
            "lut_dtype",
            "uint8",
            "internal_distance_dtype",
            "float16"));

    var parameters = codecParameters(new GPUSearchCodecFactory().apply(request));
    assertEquals(GPUSearchParams.Strategy.CUSTOM, parameters.getStrategy());
    var nativeParameters = CagraIndexParamsFactory.create(parameters, 1000, 16);
    assertEquals(CagraGraphBuildAlgo.IVF_PQ, nativeParameters.getCagraGraphBuildAlgo());
    var ivf = nativeParameters.getCuVSIvfPqParams();
    assertEquals(64, ivf.getIndexParams().getnLists());
    assertEquals(8, ivf.getSearchParams().getnProbes());
    assertEquals(15, ivf.getIndexParams().getKmeansNIters());
    assertEquals(0.25, ivf.getIndexParams().getKmeansTrainsetFraction(), 0d);
    assertEquals(8, ivf.getIndexParams().getPqDim());
    assertEquals(6, ivf.getIndexParams().getPqBits());
    assertEquals(128, ivf.getIndexParams().getMaxTrainPointsPerPqCode());
    assertEquals(3.5f, ivf.getRefinementRate(), 0f);
    assertEquals(CodebookGen.PER_CLUSTER, ivf.getIndexParams().getCodebookKind());
    assertTrue(ivf.getIndexParams().isForceRandomRotation());
    assertEquals(CudaDataType.CUDA_R_8U, ivf.getSearchParams().getLutDtype());
    assertEquals(CudaDataType.CUDA_R_16F, ivf.getSearchParams().getInternalDistanceDtype());
  }

  @Test
  public void testRequestsDoNotLeakConfiguration() throws Exception {
    Function<Map<String, Object>, Map<String, Object>> factory = new GPUSearchCodecFactory();
    var firstRequest = ivfRequest();
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
            Map.of("build_strategy", "heuristic", "graph_build_algo", "ivf_pq"),
            Map.of("graph_degree", 32, "intermediate_graph_degree", 16),
            Map.of("build_strategy", "custom", "build_quality", 7),
            Map.of("graph_build_algo", "ivf_pq", "nn_descent_n_iters", 10),
            Map.of("n_probes", 8),
            Map.of("graph_build_algo", "nn_descent", "pq_dim", 8))) {
      expectThrows(IllegalArgumentException.class, () -> factory.apply(request));
    }
  }

  @Test
  public void testRejectsInvalidIvfPqTypesBoundsAndRelationships() {
    var factory = new GPUSearchCodecFactory();
    for (Map<String, Object> options :
        List.<Map<String, Object>>of(
            Map.of("n_lists", 0),
            Map.of("n_lists", 16, "n_probes", 17),
            Map.of("n_probes", 0),
            Map.of("kmeans_n_iters", 0),
            Map.of("pq_dim", -1),
            Map.of("pq_bits", 3),
            Map.of("pq_bits", 9),
            Map.of("pq_dim", 1, "pq_bits", 5),
            Map.of("max_train_points_per_pq_code", 0),
            Map.of("kmeans_trainset_fraction", 0.0),
            Map.of("kmeans_trainset_fraction", 1.1),
            Map.of("kmeans_trainset_fraction", Double.NaN),
            Map.of("kmeans_trainset_fraction", 1),
            Map.of("refinement_rate", 1.0),
            Map.of("refinement_rate", 0.5),
            Map.of("refinement_rate", Double.POSITIVE_INFINITY),
            Map.of("refinement_rate", Double.MAX_VALUE),
            Map.of("refinement_rate", Math.nextUp(1.0)),
            Map.of("refinement_rate", 2),
            Map.of("force_random_rotation", "true"),
            Map.of("codebook_kind", "unknown"),
            Map.of("lut_dtype", "float64"),
            Map.of("internal_distance_dtype", "float16"),
            Map.of("internal_distance_dtype", "float16", "lut_dtype", "float32"),
            Map.of("internal_distance_dtype", "uint8"))) {
      var request = ivfRequest();
      request.putAll(options);
      expectThrows(IllegalArgumentException.class, () -> factory.apply(request));
    }
  }

  @Test
  public void testHalfPrecisionDistancesAcceptHalfPrecisionLookupTable() throws Exception {
    var request = ivfRequest();
    request.put("internal_distance_dtype", "float16");
    request.put("lut_dtype", "float16");

    var parameters = codecParameters(new GPUSearchCodecFactory().apply(request));
    var search =
        CagraIndexParamsFactory.create(parameters, 1024, 128)
            .getCuVSIvfPqParams()
            .getSearchParams();

    assertEquals(CudaDataType.CUDA_R_16F, search.getInternalDistanceDtype());
    assertEquals(CudaDataType.CUDA_R_16F, search.getLutDtype());
  }

  @Test
  public void testIvfPqBoundaryValuesAndFloatInputs() throws Exception {
    var request = ivfRequest();
    request.putAll(
        Map.of(
            "n_lists",
            1,
            "n_probes",
            1,
            "kmeans_n_iters",
            1,
            "pq_dim",
            0,
            "pq_bits",
            4,
            "max_train_points_per_pq_code",
            1,
            "kmeans_trainset_fraction",
            1.0f,
            "refinement_rate",
            Math.nextUp(1.0f)));
    request.putAll(
        Map.of(
            "lut_dtype",
            "float16",
            "internal_distance_dtype",
            "float32",
            "force_random_rotation",
            false,
            "codebook_kind",
            "per_subspace"));
    var parameters = codecParameters(new GPUSearchCodecFactory().apply(request));
    var ivf = CagraIndexParamsFactory.create(parameters, 1024, 128).getCuVSIvfPqParams();
    assertEquals(1, ivf.getIndexParams().getnLists());
    assertEquals(1, ivf.getSearchParams().getnProbes());
    assertEquals(0, ivf.getIndexParams().getPqDim());
    assertEquals(4, ivf.getIndexParams().getPqBits());
    assertEquals(1.0, ivf.getIndexParams().getKmeansTrainsetFraction(), 0d);
    assertEquals(Math.nextUp(1.0f), ivf.getRefinementRate(), 0f);
    assertEquals(CudaDataType.CUDA_R_16F, ivf.getSearchParams().getLutDtype());
    assertEquals(CudaDataType.CUDA_R_32F, ivf.getSearchParams().getInternalDistanceDtype());
    assertEquals(CodebookGen.PER_SUBSPACE, ivf.getIndexParams().getCodebookKind());
    assertFalse(ivf.getIndexParams().isForceRandomRotation());
  }

  private static Map<String, Object> ivfRequest() {
    return new HashMap<>(Map.of("graph_build_algo", "ivf_pq"));
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
