/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodecFactory.BEAM_WIDTH_KEY;
import static com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodecFactory.CODEC_KEY;
import static com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodecFactory.MAX_CONN_KEY;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.HnswHeuristicType;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

/** Tests the binding-compatible accelerated-HNSW codec factory contract. */
public class TestLucene101AcceleratedHNSWCodecFactory extends LuceneTestCase {

  private static final String ACCELERATED_CODEC_NAME = "Lucene101AcceleratedHNSWCodec";
  private static final String ACCELERATED_FORMAT_NAME = "Lucene99AcceleratedHNSWVectorsFormat";

  @Test
  public void testCreatesCodecAndReportsAppliedConfiguration() throws Exception {
    var factory = new Lucene101AcceleratedHNSWCodecFactory();

    Map<String, Object> response = factory.apply(request(16, 80));

    Codec codec = (Codec) response.get(CODEC_KEY);
    assertTrue(Modifier.isFinal(factory.getClass().getModifiers()));
    assertTrue(Modifier.isPublic(factory.getClass().getConstructors()[0].getModifiers()));
    assertEquals(16, response.get(MAX_CONN_KEY));
    assertEquals(80, response.get(BEAM_WIDTH_KEY));
    assertEquals(ACCELERATED_CODEC_NAME, codec.getName());
    assertNotNull(codec.knnVectorsFormat());
    assertEquals(ACCELERATED_FORMAT_NAME, codec.knnVectorsFormat().getName());
    assertCodecParameters(codec, 16, 80);
    assertEquals(
        Lucene101AcceleratedHNSWCodec.class, Codec.forName(ACCELERATED_CODEC_NAME).getClass());
  }

  @Test
  public void testCreatesExpectedCagraBuildParameters() {
    assumeTrue("cuVS not supported", isSupported());
    AcceleratedHNSWParams parameters =
        Lucene101AcceleratedHNSWCodecFactory.createParameters(16, 80);

    CagraIndexParams cagraParameters = CagraIndexParamsFactory.create(parameters, 10_000_000, 96);

    assertEquals(AcceleratedHNSWParams.Strategy.HEURISTIC, parameters.getStrategy());
    assertEquals(HnswHeuristicType.SAME_GRAPH_FOOTPRINT, parameters.getHnswHeuristicType());
    assertEquals(AcceleratedHNSWParams.DEFAULT_WRITER_THREADS, parameters.getWriterThreads());
    assertEquals(32, cagraParameters.getGraphDegree());
    assertEquals(48, cagraParameters.getIntermediateGraphDegree());
    assertEquals(
        AcceleratedHNSWParams.DEFAULT_WRITER_THREADS, cagraParameters.getNumWriterThreads());
    assertEquals(CagraGraphBuildAlgo.IVF_PQ, cagraParameters.getCagraGraphBuildAlgo());
  }

  @Test
  public void testRejectsMissingWrongTypeAndOutOfRangeParameters() {
    var factory = new Lucene101AcceleratedHNSWCodecFactory();

    assertInvalidRequest(factory, null, "request must not be null");
    assertInvalidRequest(factory, new HashMap<>(), "max_conn must have type Integer");

    var wrongType = new HashMap<String, Object>();
    wrongType.put(MAX_CONN_KEY, 16L);
    wrongType.put(BEAM_WIDTH_KEY, 80);
    assertInvalidRequest(factory, wrongType, "max_conn must have type Integer");

    var unexpected = request(16, 80);
    unexpected.put("writer_threads", 2);
    assertInvalidRequest(factory, unexpected, "request must contain only");

    assertInvalidRequest(
        factory, request(AcceleratedHNSWParams.MIN_MAX_CONN - 1, 80), "max_conn must be in range");
    assertInvalidRequest(
        factory,
        request(16, AcceleratedHNSWParams.MAX_BEAM_WIDTH + 1),
        "beam_width must be in range");
  }

  @Test
  public void testConcurrentRequestsKeepIndependentParameterPairs() throws Exception {
    Function<Map<String, Object>, Map<String, Object>> factory =
        new Lucene101AcceleratedHNSWCodecFactory();
    var executor = Executors.newFixedThreadPool(4);
    try {
      List<Future<Map<String, Object>>> responses = new ArrayList<>();
      for (int i = 0; i < 32; i++) {
        int maxConn = i % 2 == 0 ? 16 : 24;
        int beamWidth = i % 2 == 0 ? 80 : 96;
        responses.add(executor.submit(() -> factory.apply(request(maxConn, beamWidth))));
      }

      for (int i = 0; i < responses.size(); i++) {
        Map<String, Object> response = responses.get(i).get();
        assertEquals(i % 2 == 0 ? 16 : 24, response.get(MAX_CONN_KEY));
        assertEquals(i % 2 == 0 ? 80 : 96, response.get(BEAM_WIDTH_KEY));
        assertEquals(ACCELERATED_CODEC_NAME, ((Codec) response.get(CODEC_KEY)).getName());
        assertCodecParameters(
            (Codec) response.get(CODEC_KEY), i % 2 == 0 ? 16 : 24, i % 2 == 0 ? 80 : 96);
      }
    } finally {
      executor.shutdownNow();
      assertTrue(
          "codec-factory executor did not terminate",
          executor.awaitTermination(10, TimeUnit.SECONDS));
    }
  }

  private static void assertCodecParameters(Codec codec, int maxConn, int beamWidth)
      throws ReflectiveOperationException {
    // Inspect the returned format, not the response map: echoed input cannot prove configuration.
    // Keep this reflection test-only rather than adding a production accessor for the test.
    var format = codec.knnVectorsFormat();
    assertTrue(format instanceof Lucene99AcceleratedHNSWVectorsFormat);
    var parametersField =
        Lucene99AcceleratedHNSWVectorsFormat.class.getDeclaredField("acceleratedHNSWParams");
    parametersField.setAccessible(true);
    var parameters = (AcceleratedHNSWParams) parametersField.get(format);
    assertEquals(maxConn, parameters.getMaxConn());
    assertEquals(beamWidth, parameters.getBeamWidth());
  }

  private static Map<String, Object> request(int maxConn, int beamWidth) {
    var request = new HashMap<String, Object>();
    request.put(MAX_CONN_KEY, maxConn);
    request.put(BEAM_WIDTH_KEY, beamWidth);
    return request;
  }

  private static void assertInvalidRequest(
      Lucene101AcceleratedHNSWCodecFactory factory,
      Map<String, Object> request,
      String expectedMessage) {
    IllegalArgumentException error =
        expectThrows(IllegalArgumentException.class, () -> factory.apply(request));
    assertTrue(error.getMessage().contains(expectedMessage));
  }
}
