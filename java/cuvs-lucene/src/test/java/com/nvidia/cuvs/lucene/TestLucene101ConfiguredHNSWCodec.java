/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.Lucene101ConfiguredHNSWCodec.BEAM_WIDTH_PROPERTY;
import static com.nvidia.cuvs.lucene.Lucene101ConfiguredHNSWCodec.MAX_CONN_PROPERTY;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.HnswHeuristicType;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/** Tests the no-argument, property-configured accelerated HNSW codec contract. */
public class TestLucene101ConfiguredHNSWCodec extends LuceneTestCase {

  private static final String ACCELERATED_CODEC_NAME = "Lucene101AcceleratedHNSWCodec";
  private static final String ACCELERATED_FORMAT_NAME = "Lucene99AcceleratedHNSWVectorsFormat";
  private static final String LEGACY_WRITER_THREADS_PROPERTY =
      "com.nvidia.cuvs.lucene.hnsw.writerThreads";

  private final List<Lucene101ConfiguredHNSWCodec> configuredCodecs = new ArrayList<>();
  private String previousMaxConn;
  private String previousBeamWidth;
  private String previousLegacyWriterThreads;

  @Before
  public void saveAndClearConfigurationProperties() {
    previousMaxConn = System.clearProperty(MAX_CONN_PROPERTY);
    previousBeamWidth = System.clearProperty(BEAM_WIDTH_PROPERTY);
    previousLegacyWriterThreads = System.clearProperty(LEGACY_WRITER_THREADS_PROPERTY);
  }

  @After
  public void restoreConfigurationProperties() {
    for (Lucene101ConfiguredHNSWCodec codec : configuredCodecs) {
      codec.configuredParameters().getMergeExec().shutdownNow();
    }
    restoreProperty(MAX_CONN_PROPERTY, previousMaxConn);
    restoreProperty(BEAM_WIDTH_PROPERTY, previousBeamWidth);
    restoreProperty(LEGACY_WRITER_THREADS_PROPERTY, previousLegacyWriterThreads);
  }

  @Test
  public void testRequestedBuildConfigurationIsSnapshotted() throws Exception {
    setConfiguration(16, 80);

    Lucene101ConfiguredHNSWCodec codec = newConfiguredCodec();
    AcceleratedHNSWParams parameters = codec.configuredParameters();

    assertTrue(Modifier.isFinal(codec.getClass().getModifiers()));
    assertTrue(Modifier.isPublic(codec.getClass().getConstructor().getModifiers()));
    assertEquals(AcceleratedHNSWParams.Strategy.HEURISTIC, parameters.getStrategy());
    assertEquals(HnswHeuristicType.SAME_GRAPH_FOOTPRINT, parameters.getHnswHeuristicType());
    assertEquals(16, parameters.getMaxConn());
    assertEquals(80, parameters.getBeamWidth());
    assertEquals(AcceleratedHNSWParams.DEFAULT_WRITER_THREADS, parameters.getWriterThreads());
    assertEquals("Lucene101ConfiguredHNSWCodec[maxConn=16, beamWidth=80]", codec.toString());
  }

  @Test
  public void testRequestedBuildConfigurationDerivesRequestedGraphDegrees() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    setConfiguration(16, 80);

    Lucene101ConfiguredHNSWCodec codec = newConfiguredCodec();
    CagraIndexParams cagraParameters =
        CagraIndexParamsFactory.create(codec.configuredParameters(), 10_000_000, 96);

    assertEquals(32, cagraParameters.getGraphDegree());
    assertEquals(48, cagraParameters.getIntermediateGraphDegree());
    assertEquals(
        AcceleratedHNSWParams.DEFAULT_WRITER_THREADS, cagraParameters.getNumWriterThreads());
    assertEquals(CagraGraphBuildAlgo.IVF_PQ, cagraParameters.getCagraGraphBuildAlgo());
  }

  @Test
  public void testPropertiesAreRequiredAndValidated() {
    assertMissingProperty(MAX_CONN_PROPERTY);

    System.setProperty(MAX_CONN_PROPERTY, "16");
    assertMissingProperty(BEAM_WIDTH_PROPERTY);

    assertInvalidProperty(MAX_CONN_PROPERTY, "sixteen", "must be an integer");
    assertInvalidProperty(BEAM_WIDTH_PROPERTY, "eighty", "must be an integer");
    assertInvalidProperty(
        MAX_CONN_PROPERTY,
        Integer.toString(AcceleratedHNSWParams.MIN_MAX_CONN - 1),
        "must be in range");
    assertInvalidProperty(
        BEAM_WIDTH_PROPERTY,
        Integer.toString(AcceleratedHNSWParams.MAX_BEAM_WIDTH + 1),
        "must be in range");
  }

  @Test
  public void testEachInstanceRetainsItsOwnPropertySnapshot() throws Exception {
    setConfiguration(16, 80);
    Lucene101ConfiguredHNSWCodec first = newConfiguredCodec();

    setConfiguration(24, 96);
    Lucene101ConfiguredHNSWCodec second = newConfiguredCodec();

    assertEquals(16, first.configuredParameters().getMaxConn());
    assertEquals(80, first.configuredParameters().getBeamWidth());
    assertEquals(
        AcceleratedHNSWParams.DEFAULT_WRITER_THREADS,
        first.configuredParameters().getWriterThreads());
    assertEquals(24, second.configuredParameters().getMaxConn());
    assertEquals(96, second.configuredParameters().getBeamWidth());
    assertEquals(
        AcceleratedHNSWParams.DEFAULT_WRITER_THREADS,
        second.configuredParameters().getWriterThreads());
  }

  @Test
  public void testLegacyWriterThreadsPropertyDoesNotOverrideDefault() throws Exception {
    setConfiguration(16, 80);
    System.setProperty(LEGACY_WRITER_THREADS_PROPERTY, "16");

    Lucene101ConfiguredHNSWCodec codec = newConfiguredCodec();

    assertEquals(
        AcceleratedHNSWParams.DEFAULT_WRITER_THREADS,
        codec.configuredParameters().getWriterThreads());
  }

  @Test
  public void testConfiguredCodecPreservesNamesWithoutReplacingSpiProvider() throws Exception {
    setConfiguration(16, 80);

    Lucene101ConfiguredHNSWCodec codec = newConfiguredCodec();

    assertEquals(ACCELERATED_CODEC_NAME, codec.getName());
    assertNotNull(codec.knnVectorsFormat());
    assertEquals(ACCELERATED_FORMAT_NAME, codec.knnVectorsFormat().getName());
    assertEquals(
        Lucene101AcceleratedHNSWCodec.class, Codec.forName(ACCELERATED_CODEC_NAME).getClass());
  }

  private Lucene101ConfiguredHNSWCodec newConfiguredCodec() throws Exception {
    Lucene101ConfiguredHNSWCodec codec = new Lucene101ConfiguredHNSWCodec();
    configuredCodecs.add(codec);
    return codec;
  }

  private static void assertMissingProperty(String missingProperty) {
    IllegalStateException error =
        expectThrows(IllegalStateException.class, Lucene101ConfiguredHNSWCodec::new);
    assertTrue(error.getMessage().contains(missingProperty));
  }

  private static void assertInvalidProperty(
      String propertyName, String invalidValue, String expectedMessage) {
    setValidConfigurationForOtherProperty(propertyName);
    System.setProperty(propertyName, invalidValue);

    IllegalArgumentException error =
        expectThrows(IllegalArgumentException.class, Lucene101ConfiguredHNSWCodec::new);

    assertTrue(error.getMessage().contains(propertyName));
    assertTrue(error.getMessage().contains(invalidValue));
    assertTrue(error.getMessage().contains(expectedMessage));
  }

  private static void setValidConfigurationForOtherProperty(String propertyName) {
    if (MAX_CONN_PROPERTY.equals(propertyName)) {
      System.setProperty(BEAM_WIDTH_PROPERTY, "80");
    } else {
      System.setProperty(MAX_CONN_PROPERTY, "16");
    }
  }

  private static void setConfiguration(int maxConn, int beamWidth) {
    System.setProperty(MAX_CONN_PROPERTY, Integer.toString(maxConn));
    System.setProperty(BEAM_WIDTH_PROPERTY, Integer.toString(beamWidth));
  }

  private static void restoreProperty(String name, String value) {
    if (value == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, value);
    }
  }
}
