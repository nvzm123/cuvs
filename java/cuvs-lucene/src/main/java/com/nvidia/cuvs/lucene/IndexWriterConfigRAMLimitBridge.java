/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LiveIndexWriterConfig;

/**
 * Applies and verifies Lucene's per-thread indexing-memory limit.
 *
 * <p>Lucene 10.2 accepts limits below 2048 MiB through its public setter. Larger limits require an
 * explicit opt-in and are applied reflectively to Lucene's protected {@code perThreadHardLimitMB}
 * field. That unsupported override is intended only for controlled cuVS Bench vector-only builds
 * that disable automatic merges and validate their final segment topology. It must not be treated
 * as a general replacement for Lucene's safety limit.
 *
 * <p>The standard {@link Function} and {@link Map} types provide a narrow bridge for generated Java
 * bindings that do not expose reflection.
 *
 * <p>An {@link IndexWriterConfig} must be supplied under {@code config}. The request also requires a
 * positive {@link Integer} under {@code per_thread_hard_limit_mb} and a {@link Boolean} under
 * {@code allow_unsupported_lucene_ram_limit}. The response returns the same config and verified
 * limit. The {@code application_mode} entry reports the application strategy. Its value is either
 * {@code public_setter} or {@code unsupported_field_override}.
 *
 * <p>The reflective path deliberately depends on Lucene's field name and type. It rejects the
 * request if the field cannot be found, made accessible, written, or verified through Lucene's
 * public getter, so callers never silently continue with a different limit.
 */
public final class IndexWriterConfigRAMLimitBridge
    implements Function<Map<String, Object>, Map<String, Object>> {
  public static final String CONFIG_KEY = "config";
  public static final String PER_THREAD_HARD_LIMIT_MB_KEY = "per_thread_hard_limit_mb";
  public static final String ALLOW_UNSUPPORTED_LUCENE_RAM_LIMIT_KEY =
      "allow_unsupported_lucene_ram_limit";
  public static final String APPLICATION_MODE_KEY = "application_mode";
  public static final String PUBLIC_SETTER_MODE = "public_setter";
  public static final String UNSUPPORTED_FIELD_OVERRIDE_MODE = "unsupported_field_override";

  private static final int FIRST_UNSUPPORTED_LIMIT_MB = 2048;
  private static final String PER_THREAD_HARD_LIMIT_MB_FIELD = "perThreadHardLimitMB";
  private static final Set<String> REQUEST_KEYS =
      Set.of(CONFIG_KEY, PER_THREAD_HARD_LIMIT_MB_KEY, ALLOW_UNSUPPORTED_LUCENE_RAM_LIMIT_KEY);

  @Override
  public Map<String, Object> apply(Map<String, Object> request) {
    validateRequestKeys(request);
    IndexWriterConfig config = requiredValue(request, CONFIG_KEY, IndexWriterConfig.class);
    Integer requestedLimit = requiredValue(request, PER_THREAD_HARD_LIMIT_MB_KEY, Integer.class);
    Boolean allowUnsupported =
        requiredValue(request, ALLOW_UNSUPPORTED_LUCENE_RAM_LIMIT_KEY, Boolean.class);

    String applicationMode = applyAndVerify(config, requestedLimit, allowUnsupported);

    Map<String, Object> response = new HashMap<>();
    response.put(CONFIG_KEY, config);
    response.put(PER_THREAD_HARD_LIMIT_MB_KEY, config.getRAMPerThreadHardLimitMB());
    response.put(APPLICATION_MODE_KEY, applicationMode);
    return response;
  }

  static String applyAndVerify(
      IndexWriterConfig config, int requestedLimit, boolean allowUnsupported) {
    if (config == null) {
      throw new IllegalArgumentException(CONFIG_KEY + " must not be null");
    }
    if (requestedLimit <= 0) {
      throw new IllegalArgumentException(PER_THREAD_HARD_LIMIT_MB_KEY + " must be positive");
    }

    String applicationMode;
    if (requestedLimit < FIRST_UNSUPPORTED_LIMIT_MB) {
      config.setRAMPerThreadHardLimitMB(requestedLimit);
      applicationMode = PUBLIC_SETTER_MODE;
    } else {
      if (!allowUnsupported) {
        throw new IllegalArgumentException(
            PER_THREAD_HARD_LIMIT_MB_KEY
                + " values of 2048 MiB or greater require "
                + ALLOW_UNSUPPORTED_LUCENE_RAM_LIMIT_KEY
                + "=true");
      }
      setUnsupportedLimit(config, requestedLimit);
      applicationMode = UNSUPPORTED_FIELD_OVERRIDE_MODE;
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
    return applicationMode;
  }

  private static void setUnsupportedLimit(LiveIndexWriterConfig config, int requestedLimit) {
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
  }

  private static void validateRequestKeys(Map<String, Object> request) {
    if (request == null) {
      throw new IllegalArgumentException("request must not be null");
    }
    if (!request.keySet().equals(REQUEST_KEYS)) {
      throw new IllegalArgumentException("request must contain exactly " + REQUEST_KEYS);
    }
  }

  private static <T> T requiredValue(
      Map<String, Object> values, String key, Class<T> expectedType) {
    Object value = values.get(key);
    if (!expectedType.isInstance(value)) {
      throw new IllegalArgumentException(key + " must have type " + expectedType.getSimpleName());
    }
    return expectedType.cast(value);
  }
}
