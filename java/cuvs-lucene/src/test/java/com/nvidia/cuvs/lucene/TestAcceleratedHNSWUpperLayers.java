/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.QuantizationType;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntFunction;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestAcceleratedHNSWUpperLayers extends LuceneTestCase {

  @Test
  public void testParametersResolvedForEachSampledLayer() throws Throwable {
    assumeTrue("cuVS not supported", isSupported());
    try {
      int dimensions = 16;
      int[][] adjacency = new int[256][4];
      for (int i = 0; i < adjacency.length; i++) {
        for (int j = 0; j < adjacency[i].length; j++) {
          adjacency[i][j] = (i + j + 1) % adjacency.length;
        }
      }
      List<Integer> resolvedRows = new ArrayList<>();
      CagraIndexParams firstLayerParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(8)
              .withIntermediateGraphDegree(16)
              .build();
      CagraIndexParams secondLayerParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(2)
              .withIntermediateGraphDegree(16)
              .build();
      IntFunction<CagraIndexParams> resolve =
          rows -> {
            resolvedRows.add(rows);
            return switch (rows) {
              case 128 -> firstLayerParams;
              case 64 -> secondLayerParams;
              default -> throw new AssertionError("Unexpected sampled layer size: " + rows);
            };
          };
      int maxConn = 4;
      try (CuVSMatrix base = CuVSMatrix.ofArray(adjacency);
          CuVSMatrix vectors =
              CuVSMatrix.ofArray(
                  TestUtils.generateDataset(random(), adjacency.length, dimensions))) {
        AcceleratedHNSWUtils.createMultiLayerHnswGraph(
            dimensions, base, vectors, 1, resolve, QuantizationType.NONE, maxConn);
        assertTrue("A base-only graph needs no upper-layer parameters", resolvedRows.isEmpty());

        GPUBuiltHnswGraph graph =
            AcceleratedHNSWUtils.createMultiLayerHnswGraph(
                dimensions, base, vectors, 3, resolve, QuantizationType.NONE, maxConn);
        assertEquals(
            "Resolve each upper layer using its own sample size", List.of(128, 64), resolvedRows);
        assertEquals(3, graph.numLevels());
        for (int level = 1; level < graph.numLevels(); level++) {
          int expectedDegree = level == 1 ? maxConn : 2;
          var nodes = graph.getNodesOnLevel(level);
          while (nodes.hasNext()) {
            int degree = graph.getNeighbors(level, nodes.nextInt()).size();
            assertEquals("Use the parameters resolved for layer " + level, expectedDegree, degree);
            assertTrue("Upper-layer degree exceeds maxConn", degree <= maxConn);
          }
        }
      }
    } finally {
      closeCuVSResourcesInstance();
    }
  }

  @Test
  public void testLegacyListOverloadDescriptorIsPresent() throws Exception {
    Method method =
        AcceleratedHNSWUtils.class.getMethod(
            "createMultiLayerHnswGraph",
            FieldInfo.class,
            int.class,
            int.class,
            CuVSMatrix.class,
            List.class,
            int.class,
            CagraIndexParams.class,
            QuantizationType.class);

    assertEquals(GPUBuiltHnswGraph.class, method.getReturnType());

    Method matrixOverload =
        AcceleratedHNSWUtils.class.getDeclaredMethod(
            "createMultiLayerHnswGraph",
            int.class,
            CuVSMatrix.class,
            CuVSMatrix.class,
            int.class,
            CagraIndexParams.class,
            QuantizationType.class,
            int.class);
    assertFalse(Modifier.isPublic(matrixOverload.getModifiers()));
  }
}
