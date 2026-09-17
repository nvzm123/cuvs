/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.RowView;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.junit.Test;

/** Verifies serial and parallel CAGRA-adjacency materialization are equivalent and bounded. */
public class TestWriterThreadsGraphMaterialization extends LuceneTestCase {

  private static final int NUM_NODES = GPUBuiltHnswGraph.PARALLEL_MIN_NODES + 1000;
  private static final int DEGREE = 12;
  private static final int NUM_THREADS = 4;

  @Test
  public void parallelMaterializationMatchesSerial() throws Exception {
    int[][] adjacency = randomAdjacency(NUM_NODES, DEGREE, new Random(1));

    try (CuVSMatrix matrix = new ArrayMatrix(adjacency)) {
      GPUBuiltHnswGraph serial = newSingleLayerGraph(matrix, 1);
      GPUBuiltHnswGraph parallel = newSingleLayerGraph(matrix, NUM_THREADS);
      assertGraphsEqual(serial, parallel);
    }
  }

  @Test
  public void negativeSentinelsAreFilteredInSerialAndParallelMaterialization() throws Exception {
    int[][] adjacency = randomAdjacency(NUM_NODES, DEGREE, new Random(2));
    Arrays.fill(adjacency[0], -1);
    adjacency[0][1] = 1;
    adjacency[0][2] = NUM_NODES - 1;

    try (CuVSMatrix matrix = new ArrayMatrix(adjacency)) {
      GPUBuiltHnswGraph serial = newSingleLayerGraph(matrix, 1);
      GPUBuiltHnswGraph parallel = newSingleLayerGraph(matrix, NUM_THREADS);

      assertArrayEquals(new int[] {1, NUM_NODES - 1}, arcsOf(serial, 0, 0));
      assertEquals(2, serial.getNeighbors(0, 0).size());
      assertGraphsEqual(serial, parallel);
    }
  }

  @Test
  public void positiveOutOfRangeOrdinalsFailFastInSerialAndParallelMaterialization()
      throws Exception {
    int[][] adjacency = randomAdjacency(NUM_NODES, DEGREE, new Random(3));
    adjacency[0][0] = NUM_NODES;

    try (CuVSMatrix matrix = new ArrayMatrix(adjacency)) {
      IOException serial = expectThrows(IOException.class, () -> newSingleLayerGraph(matrix, 1));
      assertTrue(serial.getMessage().contains("outside graph size " + NUM_NODES));

      IOException parallel =
          expectThrows(IOException.class, () -> newSingleLayerGraph(matrix, NUM_THREADS));
      assertTrue(parallel.getMessage().contains("outside graph size " + NUM_NODES));
    }
  }

  @Test
  public void higherLayerNeighborsUseFullGraphOrdinalDomain() throws Exception {
    int graphSize = 8;
    int[][] level0Adjacency = new int[graphSize][1];
    for (int i = 0; i < graphSize; i++) {
      level0Adjacency[i][0] = i;
    }
    int[] higherLayerNodes = new int[] {2, 7};
    int[][] higherLayerAdjacency = new int[][] {{7, -1}, {2, -1}};

    try (CuVSMatrix level0 = new ArrayMatrix(level0Adjacency);
        CuVSMatrix higher = new ArrayMatrix(higherLayerAdjacency)) {
      List<int[]> layerNodes = new ArrayList<>();
      layerNodes.add(null);
      layerNodes.add(higherLayerNodes);
      GPUBuiltHnswGraph graph =
          new GPUBuiltHnswGraph(
              graphSize, /* dimensions= */ 4, layerNodes, List.of(level0, higher), NUM_THREADS);

      assertArrayEquals(new int[] {7}, arcsOf(graph, 1, 2));
      assertArrayEquals(new int[] {2}, arcsOf(graph, 1, 7));
      graph.seek(1, 2);
      assertEquals(1, graph.neighborCount());
    }
  }

  @Test
  public void graphCopyBudgetHandlesExpectedDatasetSizesAndOverflow() {
    assertTrue(GPUBuiltHnswGraph.fitsParallelGraphCopyBudget(25_000_000L, 32));
    assertFalse(GPUBuiltHnswGraph.fitsParallelGraphCopyBudget(100_000_000L, 32));
    assertFalse(GPUBuiltHnswGraph.fitsParallelGraphCopyBudget(Long.MAX_VALUE, Long.MAX_VALUE));
  }

  @Test
  public void oversizedDeviceAdjacencyUsesSerialFallback() throws Exception {
    int[][] adjacency = randomAdjacency(NUM_NODES, 1, new Random(0));
    long oversizedColumns =
        GPUBuiltHnswGraph.MAX_PARALLEL_GRAPH_COPY_BYTES / Integer.BYTES / NUM_NODES + 1;
    try (CuVSMatrix matrix = new ArrayDeviceMatrix(adjacency, oversizedColumns)) {
      GPUBuiltHnswGraph graph = newSingleLayerGraph(matrix, NUM_THREADS);
      assertEquals(NUM_NODES, graph.size());
    }
  }

  private static GPUBuiltHnswGraph newSingleLayerGraph(CuVSMatrix layer0Adjacency, int numThreads)
      throws IOException {
    return new GPUBuiltHnswGraph(
        NUM_NODES,
        /* dimensions= */ 4,
        Arrays.asList((int[]) null),
        List.of(layer0Adjacency),
        numThreads);
  }

  private static void assertGraphsEqual(HnswGraph a, HnswGraph b) throws Exception {
    assertEquals(a.numLevels(), b.numLevels());
    for (int level = 0; level < a.numLevels(); level++) {
      int[] nodes = NodesIterator.getSortedNodes(a.getNodesOnLevel(level));
      for (int node : nodes) {
        assertArrayEquals(
            "node " + node + " at level " + level + " has different neighbors",
            arcsOf(a, level, node),
            arcsOf(b, level, node));
      }
    }
  }

  private static int[] arcsOf(HnswGraph graph, int level, int node) throws Exception {
    graph.seek(level, node);
    List<Integer> arcs = new ArrayList<>();
    for (int n = graph.nextNeighbor(); n != NO_MORE_DOCS; n = graph.nextNeighbor()) {
      arcs.add(n);
    }
    return arcs.stream().mapToInt(Integer::intValue).toArray();
  }

  private static int[][] randomAdjacency(int numNodes, int degree, Random random) {
    int[][] adjacency = new int[numNodes][degree];
    for (int[] row : adjacency) {
      for (int j = 0; j < degree; j++) {
        row[j] = random.nextInt(numNodes);
      }
    }
    return adjacency;
  }

  private static class ArrayMatrix implements CuVSMatrix {
    private final int[][] rows;

    ArrayMatrix(int[][] rows) {
      this.rows = rows;
    }

    @Override
    public long size() {
      return rows.length;
    }

    @Override
    public long columns() {
      return rows.length == 0 ? 0 : rows[0].length;
    }

    @Override
    public DataType dataType() {
      return DataType.INT;
    }

    @Override
    public RowView getRow(long row) {
      return new ArrayRow(rows[Math.toIntExact(row)]);
    }

    @Override
    public void toArray(int[][] target) {
      for (int i = 0; i < rows.length; i++) {
        System.arraycopy(rows[i], 0, target[i], 0, rows[i].length);
      }
    }

    @Override
    public void toArray(float[][] target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toArray(byte[][] target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toHost(CuVSHostMatrix target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public CuVSHostMatrix toHost() {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toDevice(CuVSDeviceMatrix target, CuVSResources resources) {
      throw new UnsupportedOperationException();
    }

    @Override
    public CuVSDeviceMatrix toDevice(CuVSResources resources) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  /** Reports an oversized device shape and fails if the guarded host-copy path is reached. */
  private static final class ArrayDeviceMatrix extends ArrayMatrix implements CuVSDeviceMatrix {
    private final long reportedColumns;

    ArrayDeviceMatrix(int[][] rows, long reportedColumns) {
      super(rows);
      this.reportedColumns = reportedColumns;
    }

    @Override
    public long columns() {
      return reportedColumns;
    }

    @Override
    public CuVSHostMatrix toHost() {
      throw new AssertionError("oversized device adjacency must not be copied to host");
    }
  }

  private static final class ArrayRow implements RowView {
    private final int[] values;

    ArrayRow(int[] values) {
      this.values = values;
    }

    @Override
    public long size() {
      return values.length;
    }

    @Override
    public int getAsInt(long index) {
      return values[Math.toIntExact(index)];
    }

    @Override
    public float getAsFloat(long index) {
      throw new UnsupportedOperationException();
    }

    @Override
    public byte getAsByte(long index) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toArray(int[] target) {
      System.arraycopy(values, 0, target, 0, values.length);
    }

    @Override
    public void toArray(float[] target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toArray(byte[] target) {
      throw new UnsupportedOperationException();
    }
  }
}
