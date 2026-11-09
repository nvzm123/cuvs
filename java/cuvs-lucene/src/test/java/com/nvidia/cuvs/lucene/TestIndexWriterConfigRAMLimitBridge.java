/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.util.Version;
import org.junit.Test;

public class TestIndexWriterConfigRAMLimitBridge {

  @Test
  public void testUsesPublicSetterForSupportedLimit() {
    var config = new IndexWriterConfig();

    Map<String, Object> response = bridge().apply(request(config, 2047, false));

    assertApplied(response, config, 2047, IndexWriterConfigRAMLimitBridge.PUBLIC_SETTER_MODE);
  }

  @Test
  public void testUnsupportedLimitIsAppliedOnlyOnTheQualifiedLuceneRelease() {
    var config = new IndexWriterConfig();
    assertThrows(IllegalArgumentException.class, () -> config.setRAMPerThreadHardLimitMB(6144));

    if (LuceneCompat.LUCENE_MINOR != 2) {
      int originalLimit = config.getRAMPerThreadHardLimitMB();
      assertThrows(
          UnsupportedOperationException.class, () -> bridge().apply(request(config, 6144, true)));
      assertEquals(originalLimit, config.getRAMPerThreadHardLimitMB());
      return;
    }
    Map<String, Object> response = bridge().apply(request(config, 6144, true));

    assertApplied(
        response, config, 6144, IndexWriterConfigRAMLimitBridge.UNSUPPORTED_FIELD_OVERRIDE_MODE);
  }

  @Test
  public void testRejectsUnsupportedLimitWithoutOptIn() {
    var config = new IndexWriterConfig();
    int originalLimit = config.getRAMPerThreadHardLimitMB();

    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class, () -> bridge().apply(request(config, 2048, false)));

    assertEquals(
        "per_thread_hard_limit_mb values of 2048 MiB or greater require "
            + "allow_unsupported_lucene_ram_limit=true",
        error.getMessage());
    assertEquals(originalLimit, config.getRAMPerThreadHardLimitMB());
  }

  @Test
  @SuppressWarnings("unchecked")
  public void testEarlierArtifactCannotOverrideAnUnqualifiedRuntime() throws Exception {
    assumeTrue("Mixed-version check runs in the newer Lucene module", Version.LATEST.minor > 2);
    Path currentClasses =
        Path.of(
            IndexWriterConfigRAMLimitBridge.class
                .getProtectionDomain()
                .getCodeSource()
                .getLocation()
                .toURI());
    Path earlierClasses = currentClasses.resolve("../../../lucene-10.2/target/classes").normalize();
    assumeTrue(
        "Run the full Maven reactor to build the earlier Lucene module first",
        Files.isDirectory(earlierClasses));
    String bridgeName = IndexWriterConfigRAMLimitBridge.class.getName();
    String guardName = LuceneVersionGuard.class.getName();

    // Load the actual 10.2 bridge and guard, but use this module's real newer Lucene types.
    // This reproduces a mismatched application classpath without changing global JVM state.
    try (var earlierArtifact =
        new URLClassLoader(
            new URL[] {earlierClasses.toUri().toURL()}, getClass().getClassLoader()) {
          @Override
          protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            if (!name.equals(bridgeName) && !name.equals(guardName)) {
              return super.loadClass(name, resolve);
            }
            synchronized (getClassLoadingLock(name)) {
              Class<?> loaded = findLoadedClass(name);
              if (loaded == null) {
                loaded = findClass(name);
              }
              if (resolve) {
                resolveClass(loaded);
              }
              return loaded;
            }
          }
        }) {
      var earlierBridge =
          (Function<Map<String, Object>, Map<String, Object>>)
              earlierArtifact.loadClass(bridgeName).getConstructor().newInstance();
      var config = new IndexWriterConfig();
      int originalLimit = config.getRAMPerThreadHardLimitMB();

      var error =
          assertThrows(
              IllegalStateException.class, () -> earlierBridge.apply(request(config, 6144, true)));

      assertTrue(error.getMessage(), error.getMessage().contains("built for Lucene 10.2.x"));
      assertEquals(originalLimit, config.getRAMPerThreadHardLimitMB());
    }
  }

  @Test
  public void testRejectsIncompleteOrUnexpectedRequest() {
    var request = request(new IndexWriterConfig(), 2047, false);
    request.remove(IndexWriterConfigRAMLimitBridge.CONFIG_KEY);
    assertInvalidRequest(request, "request must contain exactly");

    request = request(new IndexWriterConfig(), 2047, false);
    request.put("unexpected", true);
    assertInvalidRequest(request, "request must contain exactly");
  }

  @Test
  public void testRejectsIncorrectRequestTypes() {
    var wrongConfig = request(new IndexWriterConfig(), 2047, false);
    wrongConfig.put(IndexWriterConfigRAMLimitBridge.CONFIG_KEY, new Object());
    assertInvalidRequest(wrongConfig, "config must have type IndexWriterConfig");

    var wrongLimit = request(new IndexWriterConfig(), 2047, false);
    wrongLimit.put(IndexWriterConfigRAMLimitBridge.PER_THREAD_HARD_LIMIT_MB_KEY, 6144L);
    assertInvalidRequest(wrongLimit, "per_thread_hard_limit_mb must have type Integer");

    var wrongOptIn = request(new IndexWriterConfig(), 2047, false);
    wrongOptIn.put(IndexWriterConfigRAMLimitBridge.ALLOW_UNSUPPORTED_LUCENE_RAM_LIMIT_KEY, "true");
    assertInvalidRequest(wrongOptIn, "allow_unsupported_lucene_ram_limit must have type Boolean");
  }

  @Test
  public void testRejectsNonPositiveLimitWithoutChangingConfig() {
    var config = new IndexWriterConfig();
    int originalLimit = config.getRAMPerThreadHardLimitMB();

    assertInvalidRequest(request(config, 0, false), "per_thread_hard_limit_mb must be positive");

    assertEquals(originalLimit, config.getRAMPerThreadHardLimitMB());
  }

  private static Function<Map<String, Object>, Map<String, Object>> bridge() {
    return new IndexWriterConfigRAMLimitBridge();
  }

  private static Map<String, Object> request(
      IndexWriterConfig config, int limit, boolean allowUnsupported) {
    var request = new HashMap<String, Object>();
    request.put(IndexWriterConfigRAMLimitBridge.CONFIG_KEY, config);
    request.put(IndexWriterConfigRAMLimitBridge.PER_THREAD_HARD_LIMIT_MB_KEY, limit);
    request.put(
        IndexWriterConfigRAMLimitBridge.ALLOW_UNSUPPORTED_LUCENE_RAM_LIMIT_KEY, allowUnsupported);
    return request;
  }

  private static void assertApplied(
      Map<String, Object> response,
      IndexWriterConfig config,
      int expectedLimit,
      String expectedMode) {
    assertEquals(expectedLimit, config.getRAMPerThreadHardLimitMB());
    assertSame(config, response.get(IndexWriterConfigRAMLimitBridge.CONFIG_KEY));
    assertEquals(
        expectedLimit, response.get(IndexWriterConfigRAMLimitBridge.PER_THREAD_HARD_LIMIT_MB_KEY));
    assertEquals(expectedMode, response.get(IndexWriterConfigRAMLimitBridge.APPLICATION_MODE_KEY));
  }

  private static void assertInvalidRequest(Map<String, Object> request, String expectedMessage) {
    IllegalArgumentException error =
        assertThrows(IllegalArgumentException.class, () -> bridge().apply(request));
    org.junit.Assert.assertTrue(error.getMessage(), error.getMessage().contains(expectedMessage));
  }
}
