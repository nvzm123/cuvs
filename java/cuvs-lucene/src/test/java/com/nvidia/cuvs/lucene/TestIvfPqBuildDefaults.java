/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

/** Checks resolved build parameters without allocating vectors or requiring CUDA. */
public class TestIvfPqBuildDefaults extends LuceneTestCase {
  @Test
  public void testBothCustomFactoriesResolveDefaultsFromEachSegmentShape() {
    var request = Map.<String, Object>of("graph_build_algo", "ivf_pq");
    var gpu = CagraBuildOptions.gpu(request);
    var hnsw = CagraBuildOptions.accelerated(request, 16, 100);
    for (boolean useHnsw : new boolean[] {false, true}) {
      // Shape-only examples: no million-row dataset is allocated by this test.
      var millionVectorSegment =
          (useHnsw
                  ? CagraIndexParamsFactory.create(hnsw, 1_000_000, 96)
                  : CagraIndexParamsFactory.create(gpu, 1_000_000, 96))
              .getCuVSIvfPqParams();
      assertEquals(1000, millionVectorSegment.getIndexParams().getnLists());
      assertEquals(32, millionVectorSegment.getSearchParams().getnProbes());
      assertEquals(32, millionVectorSegment.getIndexParams().getPqDim());
      assertEquals(8, millionVectorSegment.getIndexParams().getPqBits());
      assertEquals(10, millionVectorSegment.getIndexParams().getKmeansNIters());
      assertEquals(0.25, millionVectorSegment.getIndexParams().getKmeansTrainsetFraction(), 0d);

      // Reuse the same configuration: a second segment must not retain the first one's list count.
      var smallerSegment =
          (useHnsw
                  ? CagraIndexParamsFactory.create(hnsw, 4096, 128)
                  : CagraIndexParamsFactory.create(gpu, 4096, 128))
              .getCuVSIvfPqParams();
      assertEquals(64, smallerSegment.getIndexParams().getnLists());
      assertEquals(32, smallerSegment.getSearchParams().getnProbes());
    }
  }

  @Test
  public void testListCountFloorsSquareRootAndCapsOnlyDefaultProbes() {
    var options = IvfPqBuildOptions.parse(Map.of());
    // Each row is {segment vector count, expected lists, expected default probes}.
    for (int[] example : new int[][] {{1, 1, 1}, {1023, 31, 31}, {1024, 32, 32}, {1025, 32, 32}}) {
      var resolved = options.resolve(example[0], 128);
      assertEquals(example[1], resolved.getIndexParams().getnLists());
      assertEquals(example[2], resolved.getSearchParams().getnProbes());
    }
    var explicitProbes = IvfPqBuildOptions.parse(Map.of("n_probes", 32));
    var error =
        expectThrows(IllegalArgumentException.class, () -> explicitProbes.resolve(1023, 128));
    assertEquals("n_probes must not exceed n_lists for this segment", error.getMessage());
  }

  @Test
  public void testQuarterDimensionRoundsUpAcrossAlignmentBoundaries() {
    var options = IvfPqBuildOptions.parse(Map.of());
    // Each row is {original vector dimension, expected PQ dimension}.
    for (int[] example :
        new int[][] {{1, 32}, {96, 32}, {127, 32}, {128, 32}, {129, 64}, {768, 192}}) {
      var resolved = options.resolve(4096, example[0]);
      assertEquals(example[1], resolved.getIndexParams().getPqDim());
    }
  }

  @Test
  public void testExplicitOverridesAreCopiedAndNativeAutoPqDimensionIsRetained() {
    var request = new HashMap<String, Object>();
    request.put("graph_build_algo", "ivf_pq");
    request.put("n_lists", 7);
    request.put("pq_dim", 0);
    request.put("pq_bits", 4);
    request.put("kmeans_n_iters", 3);
    request.put("kmeans_trainset_fraction", 0.75);
    var parameters = CagraBuildOptions.gpu(request);
    request.put("n_lists", 99);
    request.put("kmeans_trainset_fraction", 0.5);

    var resolved = CagraIndexParamsFactory.create(parameters, 4096, 128).getCuVSIvfPqParams();
    assertEquals(7, resolved.getIndexParams().getnLists());
    assertEquals(7, resolved.getSearchParams().getnProbes());
    assertEquals(0, resolved.getIndexParams().getPqDim());
    assertEquals(4, resolved.getIndexParams().getPqBits());
    assertEquals(3, resolved.getIndexParams().getKmeansNIters());
    assertEquals(0.75, resolved.getIndexParams().getKmeansTrainsetFraction(), 0d);
  }

  @Test
  public void testLegacyJavaCustomBuilderKeepsItsExistingDefaults() {
    var legacy =
        new GPUSearchParams.Builder()
            .withStrategy(GPUSearchParams.Strategy.CUSTOM)
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.IVF_PQ)
            .build();
    var resolved = CagraIndexParamsFactory.create(legacy, 4096, 128).getCuVSIvfPqParams();
    assertSame(legacy.getCuVSIvfPqParams(), resolved);
    assertEquals(1024, resolved.getIndexParams().getnLists());
    assertEquals(20, resolved.getSearchParams().getnProbes());
    assertEquals(0, resolved.getIndexParams().getPqDim());
    assertEquals(20, resolved.getIndexParams().getKmeansNIters());
    assertEquals(0.5, resolved.getIndexParams().getKmeansTrainsetFraction(), 0d);
  }
}
