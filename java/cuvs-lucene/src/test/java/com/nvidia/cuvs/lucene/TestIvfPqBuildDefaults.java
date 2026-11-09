/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSIvfPqParams;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

/** Checks actual build parameters without allocating vectors or requiring CUDA. */
public class TestIvfPqBuildDefaults extends LuceneTestCase {
  @Test
  public void testBothCustomFactoriesUseDocumentedDefaults() {
    var request = Map.<String, Object>of("graph_build_algo", "ivf_pq");
    var gpu = CagraBuildOptions.gpu(request);
    var hnsw = CagraBuildOptions.accelerated(request, 16, 100);
    for (boolean useHnsw : new boolean[] {false, true}) {
      // These are shape arguments only; no dataset is allocated.
      var parameters =
          (useHnsw
                  ? CagraIndexParamsFactory.create(hnsw, 4096, 128)
                  : CagraIndexParamsFactory.create(gpu, 4096, 128))
              .getCuVSIvfPqParams();
      assertEquals(1024, parameters.getIndexParams().getnLists());
      assertEquals(20, parameters.getSearchParams().getnProbes());
      assertEquals(0, parameters.getIndexParams().getPqDim());
      assertEquals(8, parameters.getIndexParams().getPqBits());
      assertEquals(20, parameters.getIndexParams().getKmeansNIters());
      assertEquals(0.5, parameters.getIndexParams().getKmeansTrainsetFraction(), 0d);
      assertEquals(1.0f, parameters.getRefinementRate(), 0f);
    }
    // The factory-specific refinement policy must not change defaults for other cuVS Java users.
    assertEquals(2.0f, new CuVSIvfPqParams.Builder().build().getRefinementRate(), 0f);
  }

  @Test
  public void testBothCustomFactoriesPreserveExplicitRefinementRates() {
    for (Number rate : new Number[] {1.0f, 1.0d, Math.nextUp(1.0f), 2.5d}) {
      var request = Map.<String, Object>of("graph_build_algo", "ivf_pq", "refinement_rate", rate);
      var gpu = CagraBuildOptions.gpu(request);
      var hnsw = CagraBuildOptions.accelerated(request, 16, 100);

      assertEquals(
          rate.floatValue(),
          CagraIndexParamsFactory.create(gpu, 4096, 128).getCuVSIvfPqParams().getRefinementRate(),
          0f);
      assertEquals(
          rate.floatValue(),
          CagraIndexParamsFactory.create(hnsw, 4096, 128).getCuVSIvfPqParams().getRefinementRate(),
          0f);
    }
  }

  @Test
  public void testDefaultProbesAreNotSilentlyClampedToAnExplicitSmallerListCount() {
    var invalid = Map.<String, Object>of("graph_build_algo", "ivf_pq", "n_lists", 19);
    expectThrows(IllegalArgumentException.class, () -> CagraBuildOptions.gpu(invalid));
    expectThrows(
        IllegalArgumentException.class, () -> CagraBuildOptions.accelerated(invalid, 16, 100));

    var boundary = IvfPqBuildOptions.parse(Map.of("n_lists", 20));
    assertEquals(20, boundary.getIndexParams().getnLists());
    assertEquals(20, boundary.getSearchParams().getnProbes());
  }

  @Test
  public void testBothFactoriesRejectProbesAboveDefaultListsBeforeSegmentConstruction() {
    var request = Map.<String, Object>of("graph_build_algo", "ivf_pq", "n_probes", 1025);
    expectThrows(IllegalArgumentException.class, () -> CagraBuildOptions.gpu(request));
    expectThrows(
        IllegalArgumentException.class, () -> CagraBuildOptions.accelerated(request, 16, 100));
  }

  @Test
  public void testExplicitOverridesAreCopiedAndNativeAutoPqDimensionIsRetained() {
    var request = new HashMap<String, Object>();
    request.put("graph_build_algo", "ivf_pq");
    request.put("n_lists", 7);
    request.put("n_probes", 1);
    request.put("pq_dim", 0);
    request.put("pq_bits", 4);
    request.put("kmeans_n_iters", 3);
    request.put("kmeans_trainset_fraction", 0.75);
    var parameters = CagraBuildOptions.gpu(request);
    request.put("n_lists", 99);
    request.put("kmeans_trainset_fraction", 0.5);

    var actual = CagraIndexParamsFactory.create(parameters, 4096, 128).getCuVSIvfPqParams();
    assertEquals(7, actual.getIndexParams().getnLists());
    assertEquals(1, actual.getSearchParams().getnProbes());
    assertEquals(0, actual.getIndexParams().getPqDim());
    assertEquals(4, actual.getIndexParams().getPqBits());
    assertEquals(3, actual.getIndexParams().getKmeansNIters());
    assertEquals(0.75, actual.getIndexParams().getKmeansTrainsetFraction(), 0d);
  }
}
