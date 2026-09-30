/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.junit.Test;

/** Verifies the ownership boundaries of the standard thin JAR after Maven packages it. */
public class ThinJarContentsIT {

  private static final String THIN_JAR_PROPERTY = "cuvs.lucene.thinJar";
  private static final String CUVS_LUCENE_PACKAGE = "com/nvidia/cuvs/lucene/";
  private static final String CODEC_SERVICE = "META-INF/services/org.apache.lucene.codecs.Codec";
  private static final String FORMAT_SERVICE =
      "META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat";

  private static final Set<String> REQUIRED_PRODUCTION_CLASSES =
      Set.of(
          CUVS_LUCENE_PACKAGE + "CuvsBenchFbinIndexingBridge.class",
          CUVS_LUCENE_PACKAGE + "IndexSearcherTimingBridge.class",
          CUVS_LUCENE_PACKAGE + "IndexWriterConfigRAMLimitBridge.class",
          CUVS_LUCENE_PACKAGE + "Lucene101AcceleratedHNSWCodecFactory.class");

  private static final Set<String> REMOVED_PRODUCTION_CLASSES =
      Set.of(
          CUVS_LUCENE_PACKAGE + "IndexWriterConfigPerThreadHardLimitBridge.class",
          CUVS_LUCENE_PACKAGE + "Lucene101ConfiguredHNSWCodec.class");

  private static final Map<String, String> EXPECTED_CODEC_PROVIDERS =
      Map.of(
          "CuVS2510GPUSearchCodec", "com.nvidia.cuvs.lucene.CuVS2510GPUSearchCodec",
          "Lucene101AcceleratedHNSWBinaryQuantizedCodec",
              "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedCodec",
          "Lucene101AcceleratedHNSWCodec", "com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec",
          "Lucene101AcceleratedHNSWScalarQuantizedCodec",
              "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWScalarQuantizedCodec");

  private static final Map<String, String> EXPECTED_VECTOR_FORMAT_PROVIDERS =
      Map.of(
          "CuVS2510GPUVectorsFormat", "com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat",
          "Lucene99AcceleratedHNSWBinaryQuantizedVectorsFormat",
              "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat",
          "Lucene99AcceleratedHNSWScalarQuantizedVectorsFormat",
              "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWScalarQuantizedVectorsFormat",
          "Lucene99AcceleratedHNSWVectorsFormat",
              "com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat",
          "Lucene99HnswScalarQuantizedVectorsFormat",
              "org.apache.lucene.codecs.lucene99.Lucene99HnswScalarQuantizedVectorsFormat",
          "Lucene99HnswVectorsFormat",
              "org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat");

  private static final Map<String, Set<String>> EXPECTED_SERVICES =
      Map.of(
          CODEC_SERVICE,
          Set.copyOf(EXPECTED_CODEC_PROVIDERS.values()),
          FORMAT_SERVICE,
          Set.copyOf(EXPECTED_VECTOR_FORMAT_PROVIDERS.values()));

  @Test
  public void testStandardThinJarContainsOnlyProductionOwnedPayloads() throws Exception {
    Path thinJar = requireConfiguredThinJar();
    assertFailsafeUsesPackagedThinJar(thinJar);

    try (JarFile jar = new JarFile(thinJar.toFile())) {
      Set<String> entries = readFileEntries(jar);

      assertRequiredProductionClassesArePackaged(entries);
      assertRemovedProductionClassesStayRemoved(entries);
      assertNoDependencyClassesAreBundled(entries);
      assertNoCompiledTestOutputIsPackaged(thinJar, entries);
      assertNoSourceJavadocOrNativePayloadIsPackaged(entries);
    }
  }

  @Test
  public void testPackagedLuceneSpiDescriptorsAreCompleteAndResolvable() throws Exception {
    Path thinJar = requireConfiguredThinJar();
    assertFailsafeUsesPackagedThinJar(thinJar);

    try (JarFile jar = new JarFile(thinJar.toFile())) {
      assertExactLuceneServices(jar, readFileEntries(jar));
    }

    assertSpiProvidersResolve(EXPECTED_CODEC_PROVIDERS, Codec.availableCodecs(), Codec::forName);
    assertSpiProvidersResolve(
        EXPECTED_VECTOR_FORMAT_PROVIDERS,
        KnnVectorsFormat.availableKnnVectorsFormats(),
        KnnVectorsFormat::forName);
  }

  private static Path requireConfiguredThinJar() {
    String configuredJar = System.getProperty(THIN_JAR_PROPERTY);
    assertNotNull("Missing system property " + THIN_JAR_PROPERTY, configuredJar);
    Path thinJar = Path.of(configuredJar).toAbsolutePath().normalize();
    assertTrue("Thin JAR does not exist: " + thinJar, Files.isRegularFile(thinJar));
    assertFalse(
        "Expected the standard thin JAR, not the dependency assembly: " + thinJar,
        thinJar.getFileName().toString().contains("jar-with-dependencies"));
    return thinJar;
  }

  private static void assertFailsafeUsesPackagedThinJar(Path thinJar) {
    Path mainClasses = thinJar.getParent().resolve("classes").toAbsolutePath().normalize();
    String currentClasspath =
        System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
    List<Path> classpathEntries =
        Pattern.compile(Pattern.quote(File.pathSeparator))
            .splitAsStream(currentClasspath)
            .map(Path::of)
            .map(Path::toAbsolutePath)
            .map(Path::normalize)
            .toList();

    assertTrue(
        "Failsafe classpath does not contain the packaged thin JAR: " + thinJar,
        classpathEntries.contains(thinJar));
    assertFalse(
        "Failsafe classpath contains compiled main classes instead of the packaged thin JAR",
        classpathEntries.contains(mainClasses));
  }

  private static Set<String> readFileEntries(JarFile jar) {
    return jar.stream()
        .filter(entry -> !entry.isDirectory())
        .map(JarEntry::getName)
        .collect(Collectors.toUnmodifiableSet());
  }

  private static void assertRequiredProductionClassesArePackaged(Set<String> entries) {
    for (String requiredClass : REQUIRED_PRODUCTION_CLASSES) {
      assertTrue(
          "Thin JAR is missing production class " + requiredClass, entries.contains(requiredClass));
    }
  }

  private static void assertRemovedProductionClassesStayRemoved(Set<String> entries) {
    for (String removedClass : REMOVED_PRODUCTION_CLASSES) {
      assertFalse(
          "Thin JAR contains removed class " + removedClass, entries.contains(removedClass));
    }
  }

  private static void assertNoDependencyClassesAreBundled(Set<String> entries) {
    for (String entry : entries) {
      assertFalse(
          "Thin JAR bundles a Lucene dependency class: " + entry,
          entry.endsWith(".class") && entry.contains("org/apache/lucene/"));
      assertFalse(
          "Thin JAR bundles a base cuvs-java class: " + entry,
          entry.endsWith(".class")
              && entry.contains("com/nvidia/cuvs/")
              && !entry.contains(CUVS_LUCENE_PACKAGE));
    }
  }

  private static void assertNoCompiledTestOutputIsPackaged(Path thinJar, Set<String> entries)
      throws IOException {
    Path testClasses = thinJar.getParent().resolve("test-classes");
    assertTrue(
        "Compiled test output does not exist: " + testClasses, Files.isDirectory(testClasses));

    try (Stream<Path> testOutput = Files.walk(testClasses)) {
      testOutput
          .filter(Files::isRegularFile)
          .map(testClasses::relativize)
          .map(Path::toString)
          .map(path -> path.replace(File.separatorChar, '/'))
          .forEach(
              testEntry ->
                  assertFalse(
                      "Thin JAR contains compiled test output: " + testEntry,
                      entries.contains(testEntry)));
    }
  }

  private static void assertNoSourceJavadocOrNativePayloadIsPackaged(Set<String> entries) {
    for (String entry : entries) {
      String lowerEntry = entry.toLowerCase();
      assertFalse("Thin JAR contains Java source: " + entry, lowerEntry.endsWith(".java"));
      assertFalse("Thin JAR contains Javadoc HTML: " + entry, lowerEntry.endsWith(".html"));
      assertFalse(
          "Thin JAR contains a Javadoc index: " + entry,
          lowerEntry.endsWith("element-list")
              || lowerEntry.endsWith("package-list")
              || lowerEntry.endsWith("member-search-index.js")
              || lowerEntry.endsWith("type-search-index.js"));
      assertFalse(
          "Thin JAR contains a native library: " + entry,
          lowerEntry.endsWith(".so")
              || lowerEntry.contains(".so.")
              || lowerEntry.endsWith(".dll")
              || lowerEntry.endsWith(".dylib")
              || lowerEntry.endsWith(".jnilib")
              || lowerEntry.endsWith(".a")
              || lowerEntry.endsWith(".lib"));
    }
  }

  private static void assertExactLuceneServices(JarFile jar, Set<String> entries) throws Exception {
    Set<String> serviceDescriptors =
        entries.stream()
            .filter(name -> name.startsWith("META-INF/services/org.apache.lucene."))
            .collect(Collectors.toUnmodifiableSet());
    assertEquals(
        "Unexpected Lucene service descriptors", EXPECTED_SERVICES.keySet(), serviceDescriptors);

    for (Map.Entry<String, Set<String>> expectedService : EXPECTED_SERVICES.entrySet()) {
      List<String> providers = readProviders(jar, expectedService.getKey());
      assertEquals(
          "Duplicate providers in " + expectedService.getKey(),
          Set.copyOf(providers).size(),
          providers.size());
      assertEquals(
          "Unexpected providers in " + expectedService.getKey(),
          expectedService.getValue(),
          Set.copyOf(providers));
      assertProviderTypesMatchService(expectedService.getKey(), providers, entries);
    }
  }

  private static void assertProviderTypesMatchService(
      String serviceDescriptor, List<String> providers, Set<String> entries) throws Exception {
    Class<?> serviceType =
        CODEC_SERVICE.equals(serviceDescriptor) ? Codec.class : KnnVectorsFormat.class;

    for (String provider : providers) {
      Class<?> providerType =
          Class.forName(provider, false, Thread.currentThread().getContextClassLoader());
      assertTrue(
          provider + " does not implement " + serviceType.getName(),
          serviceType.isAssignableFrom(providerType));
      assertNotNull(
          provider + " has no public no-argument constructor", providerType.getConstructor());

      String providerEntry = provider.replace('.', '/') + ".class";
      if (provider.startsWith("com.nvidia.cuvs.lucene.")) {
        assertTrue(
            "Thin JAR is missing provider class " + provider, entries.contains(providerEntry));
      } else {
        assertFalse(
            "Thin JAR bundles dependency provider class " + provider,
            entries.contains(providerEntry));
      }
    }
  }

  private static <T> void assertSpiProvidersResolve(
      Map<String, String> expectedProviders, Set<String> availableNames, SpiResolver<T> resolver) {
    assertTrue(
        "Missing SPI names: " + findMissingNames(expectedProviders.keySet(), availableNames),
        availableNames.containsAll(expectedProviders.keySet()));
    for (Map.Entry<String, String> expectedProvider : expectedProviders.entrySet()) {
      Object resolved = resolver.resolve(expectedProvider.getKey());
      assertNotNull("SPI name does not resolve: " + expectedProvider.getKey(), resolved);
      assertEquals(
          "Unexpected provider for SPI name " + expectedProvider.getKey(),
          expectedProvider.getValue(),
          resolved.getClass().getName());
    }
  }

  private static Set<String> findMissingNames(Set<String> expected, Set<String> available) {
    return expected.stream()
        .filter(name -> !available.contains(name))
        .collect(Collectors.toUnmodifiableSet());
  }

  private static List<String> readProviders(JarFile jar, String descriptor) throws IOException {
    JarEntry entry = jar.getJarEntry(descriptor);
    assertNotNull("Missing service descriptor " + descriptor, entry);
    try (BufferedReader reader =
        new BufferedReader(
            new InputStreamReader(jar.getInputStream(entry), StandardCharsets.UTF_8))) {
      return reader
          .lines()
          .map(ThinJarContentsIT::stripComment)
          .filter(line -> !line.isEmpty())
          .toList();
    }
  }

  private static String stripComment(String line) {
    int commentStart = line.indexOf('#');
    return (commentStart < 0 ? line : line.substring(0, commentStart)).trim();
  }

  @FunctionalInterface
  private interface SpiResolver<T> {
    T resolve(String name);
  }
}
