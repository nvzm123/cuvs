/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.index.IndexWriterConfig;
import org.junit.Test;

public class TestIndexWriterConfigPerThreadHardLimitBridge {

  @Test
  public void testAppliesAndVerifiesLimitAbovePublicSetterCap() {
    var config = new IndexWriterConfig();
    assertThrows(IllegalArgumentException.class, () -> config.setRAMPerThreadHardLimitMB(61_440));
    Function<Map<String, Object>, Map<String, Object>> bridge =
        new IndexWriterConfigPerThreadHardLimitBridge();
    var request = request(config, 61_440);

    Map<String, Object> response = bridge.apply(request);

    assertEquals(61_440, config.getRAMPerThreadHardLimitMB());
    assertSame(config, response.get(IndexWriterConfigPerThreadHardLimitBridge.CONFIG_KEY));
    assertEquals(
        61_440,
        response.get(IndexWriterConfigPerThreadHardLimitBridge.PER_THREAD_HARD_LIMIT_MB_KEY));
  }

  @Test
  public void testRejectsNullRequest() {
    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () -> new IndexWriterConfigPerThreadHardLimitBridge().apply(null));

    assertEquals("request must not be null", error.getMessage());
  }

  @Test
  public void testRejectsMissingOrIncorrectConfig() {
    var missingConfig = new HashMap<String, Object>();
    missingConfig.put(IndexWriterConfigPerThreadHardLimitBridge.PER_THREAD_HARD_LIMIT_MB_KEY, 1);
    assertInvalidRequest(missingConfig, "config must have type LiveIndexWriterConfig");

    var incorrectConfig = new HashMap<String, Object>();
    incorrectConfig.put(IndexWriterConfigPerThreadHardLimitBridge.CONFIG_KEY, new Object());
    incorrectConfig.put(IndexWriterConfigPerThreadHardLimitBridge.PER_THREAD_HARD_LIMIT_MB_KEY, 1);
    assertInvalidRequest(incorrectConfig, "config must have type LiveIndexWriterConfig");
  }

  @Test
  public void testRejectsMissingOrNonIntegerLimit() {
    var config = new IndexWriterConfig();
    var missingLimit = new HashMap<String, Object>();
    missingLimit.put(IndexWriterConfigPerThreadHardLimitBridge.CONFIG_KEY, config);
    assertInvalidRequest(missingLimit, "per_thread_hard_limit_mb must have type Integer");

    var nonIntegerLimit = new HashMap<String, Object>();
    nonIntegerLimit.put(IndexWriterConfigPerThreadHardLimitBridge.CONFIG_KEY, config);
    nonIntegerLimit.put(
        IndexWriterConfigPerThreadHardLimitBridge.PER_THREAD_HARD_LIMIT_MB_KEY, 61_440L);
    assertInvalidRequest(nonIntegerLimit, "per_thread_hard_limit_mb must have type Integer");
  }

  @Test
  public void testRejectsNonPositiveLimitsWithoutChangingConfig() {
    var config = new IndexWriterConfig();
    int originalLimit = config.getRAMPerThreadHardLimitMB();

    assertInvalidRequest(request(config, 0), "per_thread_hard_limit_mb must be positive");
    assertEquals(originalLimit, config.getRAMPerThreadHardLimitMB());

    assertInvalidRequest(request(config, -1), "per_thread_hard_limit_mb must be positive");
    assertEquals(originalLimit, config.getRAMPerThreadHardLimitMB());
  }

  private static Map<String, Object> request(IndexWriterConfig config, int limit) {
    var request = new HashMap<String, Object>();
    request.put(IndexWriterConfigPerThreadHardLimitBridge.CONFIG_KEY, config);
    request.put(IndexWriterConfigPerThreadHardLimitBridge.PER_THREAD_HARD_LIMIT_MB_KEY, limit);
    return request;
  }

  private static void assertInvalidRequest(Map<String, Object> request, String expectedMessage) {
    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () -> new IndexWriterConfigPerThreadHardLimitBridge().apply(request));
    assertEquals(expectedMessage, error.getMessage());
  }
}
