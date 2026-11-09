/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraSearchParams;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.search.Query;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

/** Tests query construction and binding validation without loading a GPU or native search plan. */
public class TestGPUKnnFloatVectorQueryFactory extends LuceneTestCase {

  @Test
  public void testDefaultRequestConstructsGpuQueryThroughStandardFunction() throws Exception {
    // Match the binding's public no-argument construction and standard-interface invocation.
    Function<Map<String, Object>, Query> factory =
        GPUKnnFloatVectorQueryFactory.class.getConstructor().newInstance();
    var request = request(10);

    Query result = factory.apply(request);

    assertTrue(result instanceof GPUKnnFloatVectorQuery);
    var query = (GPUKnnFloatVectorQuery) result;
    assertEquals("vector", query.getField());
    assertEquals(10, query.getK());
    assertArrayEquals(new float[] {1f, 2f}, query.getTargetCopy(), 0f);
    assertNull(query.getFilter());
    assertSearchParameters(query, 10, 1, 0, 0, CagraSearchParams.SearchAlgo.AUTO);
    assertEquals(3, request.size());
  }

  @Test
  public void testForwardsEveryExplicitSearchParameter() {
    var request = request(20);
    request.put("itopk", 96);
    request.put("search_width", 4);
    request.put("max_iterations", 17);
    request.put("thread_block_size", 128);
    request.put("algo", "multi_cta");

    var query = (GPUKnnFloatVectorQuery) new GPUKnnFloatVectorQueryFactory().apply(request);

    assertSearchParameters(query, 96, 4, 128, 17, CagraSearchParams.SearchAlgo.MULTI_CTA);
  }

  @Test
  public void testAlgorithmNamesAndThreadBlockSizes() {
    var algorithms =
        Map.of(
            "auto", CagraSearchParams.SearchAlgo.AUTO,
            "single_cta", CagraSearchParams.SearchAlgo.SINGLE_CTA,
            "multi_cta", CagraSearchParams.SearchAlgo.MULTI_CTA);
    for (var entry : algorithms.entrySet()) {
      var request = request(1);
      request.put("algo", entry.getKey());
      assertSearchParameters(
          (GPUKnnFloatVectorQuery) new GPUKnnFloatVectorQueryFactory().apply(request),
          1,
          1,
          0,
          0,
          entry.getValue());
    }
    for (int blockSize : new int[] {0, 64, 128, 256, 512, 1024}) {
      var request = request(1);
      request.put("thread_block_size", blockSize);
      assertSearchParameters(
          (GPUKnnFloatVectorQuery) new GPUKnnFloatVectorQueryFactory().apply(request),
          1,
          1,
          blockSize,
          0,
          CagraSearchParams.SearchAlgo.AUTO);
    }
  }

  @Test
  public void testJavaRepresentationBoundariesDoNotClaimNativeSupport() {
    var request = request(1024);
    request.put("itopk", Integer.MAX_VALUE);
    request.put("search_width", GPUKnnFloatVectorQuery.MAX_SEARCH_WIDTH);
    request.put("max_iterations", Integer.MAX_VALUE);
    request.put("algo", "multi_cta");

    var query = (GPUKnnFloatVectorQuery) new GPUKnnFloatVectorQueryFactory().apply(request);

    // Construction does not build a native plan or establish that these combinations can run.
    assertSearchParameters(
        query,
        Integer.MAX_VALUE,
        GPUKnnFloatVectorQuery.MAX_SEARCH_WIDTH,
        0,
        Integer.MAX_VALUE,
        CagraSearchParams.SearchAlgo.MULTI_CTA);
  }

  @Test
  public void testRejectsMissingNullAndWrongTypeValues() {
    assertInvalidRequest(null, "request must not be null");
    for (String key : List.of("field", "target", "k")) {
      var request = request(1);
      request.remove(key);
      assertInvalidRequest(request, key + " must have type");
    }
    for (String key :
        List.of(
            "field",
            "target",
            "k",
            "itopk",
            "search_width",
            "max_iterations",
            "thread_block_size",
            "algo")) {
      var request = request(1);
      request.put(key, null);
      assertInvalidRequest(request, key + " must have type");
    }
    for (String key :
        List.of("k", "itopk", "search_width", "max_iterations", "thread_block_size")) {
      for (Object value : List.of(1L, 1.0, "1", true)) {
        var request = request(1);
        request.put(key, value);
        assertInvalidRequest(request, key + " must have type Integer");
      }
    }
    for (var entry :
        Map.<String, Object>of(
                "field",
                1,
                "target",
                new double[] {1, 2},
                "algo",
                CagraSearchParams.SearchAlgo.AUTO)
            .entrySet()) {
      var request = request(1);
      request.put(entry.getKey(), entry.getValue());
      assertInvalidRequest(request, entry.getKey() + " must have type");
    }
  }

  @Test
  @SuppressWarnings({"rawtypes", "unchecked"})
  public void testRejectsUnknownAndNonStringKeys() {
    var request = request(1);
    request.put("num_candidates", 2);
    assertInvalidRequest(request, "unknown query request key: num_candidates");
    request = request(1);
    request.put(null, 2);
    assertInvalidRequest(request, "unknown query request key: null");
    Map rawRequest = request(1);
    rawRequest.put(7, 2);
    assertInvalidRequest(rawRequest, "unknown query request key: 7");
  }

  @Test
  public void testRejectsInvalidSearchBoundsAndAlgorithmNames() {
    for (int k : new int[] {-1, 0, 1025, Integer.MAX_VALUE}) {
      assertInvalidRequest(request(k), "k must be in range");
    }
    for (int iTopK : new int[] {-1, 0, 9}) {
      var request = request(10);
      request.put("itopk", iTopK);
      assertInvalidRequest(request, "itopk must be at least k");
    }
    for (int width : new int[] {-1, 0, GPUKnnFloatVectorQuery.MAX_SEARCH_WIDTH + 1}) {
      var request = request(1);
      request.put("search_width", width);
      assertInvalidRequest(request, "searchWidth");
    }
    for (int blockSize : new int[] {-1, 1, 32, 192, 1025}) {
      var request = request(1);
      request.put("thread_block_size", blockSize);
      assertInvalidRequest(request, "threadBlockSize");
    }
    var negativeIterations = request(1);
    negativeIterations.put("max_iterations", -1);
    assertInvalidRequest(negativeIterations, "maxIterations");
    for (String algo : List.of("AUTO", "", "multi_kernel", "invalid")) {
      var request = request(1);
      request.put("algo", algo);
      assertInvalidRequest(request, "algo must be auto, single_cta, or multi_cta");
    }
  }

  @Test
  public void testSingleCtaLimitIncludesDefaultAndExplicitItopk() {
    var boundary = request(512);
    boundary.put("algo", "single_cta");
    assertSearchParameters(
        (GPUKnnFloatVectorQuery) new GPUKnnFloatVectorQueryFactory().apply(boundary),
        512,
        1,
        0,
        0,
        CagraSearchParams.SearchAlgo.SINGLE_CTA);

    var defaultTooLarge = request(513);
    defaultTooLarge.put("algo", "single_cta");
    assertInvalidRequest(defaultTooLarge, "effective iTopK must not exceed 512");
    var explicitTooLarge = request(1);
    explicitTooLarge.put("algo", "single_cta");
    explicitTooLarge.put("itopk", 513);
    assertInvalidRequest(explicitTooLarge, "effective iTopK must not exceed 512");
  }

  @Test
  public void testRejectsNonFiniteVectorsLikeLuceneQuery() {
    for (float value : new float[] {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
      var request = request(1);
      request.put("target", new float[] {value});
      expectThrows(
          IllegalArgumentException.class, () -> new GPUKnnFloatVectorQueryFactory().apply(request));
    }
  }

  private static void assertSearchParameters(
      GPUKnnFloatVectorQuery query,
      int iTopK,
      int searchWidth,
      int threadBlockSize,
      int maxIterations,
      CagraSearchParams.SearchAlgo algo) {
    // Inspect what the constructed query forwards to search, not echoed request data.
    var collector = (GPUPerLeafCuVSKnnCollector) query.newPerLeafCollector(query.getK(), 1000);
    assertEquals(iTopK, collector.getiTopK());
    assertEquals(searchWidth, collector.getSearchWidth());
    assertEquals(threadBlockSize, collector.getThreadBlockSize());
    assertEquals(maxIterations, collector.getMaxIterations());
    assertEquals(algo, collector.getSearchAlgo());
  }

  private static Map<String, Object> request(int k) {
    var request = new HashMap<String, Object>();
    request.put("field", "vector");
    request.put("target", new float[] {1f, 2f});
    request.put("k", k);
    return request;
  }

  private static void assertInvalidRequest(Map<String, Object> request, String expectedMessage) {
    var error =
        expectThrows(
            IllegalArgumentException.class,
            () -> new GPUKnnFloatVectorQueryFactory().apply(request));
    assertTrue(error.getMessage(), error.getMessage().contains(expectedMessage));
  }
}
