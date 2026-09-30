/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.HnswHeuristicType;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.codecs.Codec;

/**
 * Constructs an accelerated HNSW codec from one self-contained parameter request.
 *
 * <p>The standard {@link Function} and {@link Map} types form a narrow bridge for generated Java
 * bindings that do not wrap parameterized constructors. The request has exactly two entries,
 * max_conn and beam_width. Both must be {@link Integer} values in the inclusive range 1 through
 * 512. The response contains the configured {@code codec} and the verified applied values under the
 * same parameter keys. Invalid requests throw {@link IllegalArgumentException}; codec construction
 * or initialization failures throw {@link IllegalStateException}.
 *
 * <p>The factory is stateless. In particular, it does not use JVM system properties, so concurrent
 * callers cannot observe or overwrite one another's configuration.
 *
 * @since 26.12
 */
public final class Lucene101AcceleratedHNSWCodecFactory
    implements Function<Map<String, Object>, Map<String, Object>> {
  public static final String MAX_CONN_KEY = "max_conn";
  public static final String BEAM_WIDTH_KEY = "beam_width";
  public static final String CODEC_KEY = "codec";

  /**
   * Constructs one codec from the complete request.
   *
   * @param request exactly the {@code max_conn} and {@code beam_width} integer entries
   * @return the codec and its verified applied parameter values
   * @throws IllegalArgumentException if the request is null, incomplete, has extra keys, contains
   *     non-integer values, or contains values outside the inclusive range 1 through 512
   * @throws IllegalStateException if codec construction or vector-format initialization fails
   */
  @Override
  public Map<String, Object> apply(Map<String, Object> request) {
    int maxConn =
        requiredInteger(
            request,
            MAX_CONN_KEY,
            AcceleratedHNSWParams.MIN_MAX_CONN,
            AcceleratedHNSWParams.MAX_MAX_CONN);
    int beamWidth =
        requiredInteger(
            request,
            BEAM_WIDTH_KEY,
            AcceleratedHNSWParams.MIN_BEAM_WIDTH,
            AcceleratedHNSWParams.MAX_BEAM_WIDTH);
    if (request.size() != 2) {
      throw new IllegalArgumentException(
          "request must contain only " + MAX_CONN_KEY + " and " + BEAM_WIDTH_KEY);
    }
    AcceleratedHNSWParams parameters = createParameters(maxConn, beamWidth);

    final Codec codec;
    try {
      codec = new Lucene101AcceleratedHNSWCodec(parameters);
      if (codec.knnVectorsFormat() == null) {
        throw new IllegalStateException(
            "Accelerated HNSW codec did not initialize a vector format");
      }
    } catch (IllegalStateException error) {
      throw error;
    } catch (Exception | LinkageError error) {
      throw new IllegalStateException("Could not construct the accelerated HNSW codec", error);
    }

    Map<String, Object> response = new HashMap<>();
    response.put(CODEC_KEY, codec);
    response.put(MAX_CONN_KEY, parameters.getMaxConn());
    response.put(BEAM_WIDTH_KEY, parameters.getBeamWidth());
    return response;
  }

  static AcceleratedHNSWParams createParameters(int maxConn, int beamWidth) {
    return new AcceleratedHNSWParams.Builder()
        .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
        .withHnswHeuristicType(HnswHeuristicType.SAME_GRAPH_FOOTPRINT)
        .withMaxConn(maxConn)
        .withBeamWidth(beamWidth)
        .build();
  }

  private static int requiredInteger(
      Map<String, Object> values, String key, int minimum, int maximum) {
    if (values == null) {
      throw new IllegalArgumentException("request must not be null");
    }
    Object value = values.get(key);
    if (!(value instanceof Integer integer)) {
      throw new IllegalArgumentException(key + " must have type Integer");
    }
    if (integer < minimum || integer > maximum) {
      throw new IllegalArgumentException(
          key + " must be in range [" + minimum + ", " + maximum + "], but was: " + integer);
    }
    return integer;
  }
}
