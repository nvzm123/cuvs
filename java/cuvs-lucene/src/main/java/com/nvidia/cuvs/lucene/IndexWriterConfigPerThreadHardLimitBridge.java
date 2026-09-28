/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.index.LiveIndexWriterConfig;

/**
 * Sets Lucene's per-thread indexing-memory limit without the public setter's 2048 MiB cap.
 *
 * <p>The standard {@link Function} and {@link Map} types form a narrow bridge for generated Java
 * bindings that do not wrap this class directly. The request must contain
 * {@code config} (a {@code LiveIndexWriterConfig}) and {@code per_thread_hard_limit_mb} (a positive
 * {@link Integer}). The response contains the same config and the verified applied limit under
 * those keys.
 *
 * <p>This bridge deliberately depends on Lucene's non-public field name. It fails if the field
 * cannot be found, made accessible, written, or verified so that a caller never silently continues
 * with Lucene's lower default limit.
 */
public final class IndexWriterConfigPerThreadHardLimitBridge
    implements Function<Map<String, Object>, Map<String, Object>> {
  public static final String CONFIG_KEY = "config";
  public static final String PER_THREAD_HARD_LIMIT_MB_KEY = "per_thread_hard_limit_mb";

  private static final String PER_THREAD_HARD_LIMIT_MB_FIELD = "perThreadHardLimitMB";

  @Override
  public Map<String, Object> apply(Map<String, Object> request) {
    LiveIndexWriterConfig config = requiredValue(request, CONFIG_KEY, LiveIndexWriterConfig.class);
    Integer requestedLimit = requiredValue(request, PER_THREAD_HARD_LIMIT_MB_KEY, Integer.class);
    if (requestedLimit <= 0) {
      throw new IllegalArgumentException(PER_THREAD_HARD_LIMIT_MB_KEY + " must be positive");
    }

    setAndVerify(config, requestedLimit);

    Map<String, Object> response = new HashMap<>();
    response.put(CONFIG_KEY, config);
    response.put(PER_THREAD_HARD_LIMIT_MB_KEY, config.getRAMPerThreadHardLimitMB());
    return response;
  }

  private static void setAndVerify(LiveIndexWriterConfig config, int requestedLimit) {
    Field limitField;
    try {
      limitField = LiveIndexWriterConfig.class.getDeclaredField(PER_THREAD_HARD_LIMIT_MB_FIELD);
    } catch (NoSuchFieldException | SecurityException error) {
      throw new IllegalStateException(
          "Unable to locate LiveIndexWriterConfig." + PER_THREAD_HARD_LIMIT_MB_FIELD, error);
    }

    if (limitField.getType() != int.class) {
      throw new IllegalStateException(
          "LiveIndexWriterConfig." + PER_THREAD_HARD_LIMIT_MB_FIELD + " is not an int field");
    }

    try {
      limitField.setAccessible(true);
      limitField.setInt(config, requestedLimit);
    } catch (IllegalAccessException | RuntimeException error) {
      throw new IllegalStateException(
          "Unable to set LiveIndexWriterConfig." + PER_THREAD_HARD_LIMIT_MB_FIELD, error);
    }

    int actualLimit = config.getRAMPerThreadHardLimitMB();
    if (actualLimit != requestedLimit) {
      throw new IllegalStateException(
          "Failed to verify LiveIndexWriterConfig."
              + PER_THREAD_HARD_LIMIT_MB_FIELD
              + ": requested "
              + requestedLimit
              + " but read "
              + actualLimit);
    }
  }

  private static <T> T requiredValue(
      Map<String, Object> values, String key, Class<T> expectedType) {
    if (values == null) {
      throw new IllegalArgumentException("request must not be null");
    }
    Object value = values.get(key);
    if (!expectedType.isInstance(value)) {
      throw new IllegalArgumentException(key + " must have type " + expectedType.getSimpleName());
    }
    return expectedType.cast(value);
  }
}
