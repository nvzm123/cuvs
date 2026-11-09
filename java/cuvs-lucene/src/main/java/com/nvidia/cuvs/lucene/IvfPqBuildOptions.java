/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.CagraBuildOptions.choice;
import static com.nvidia.cuvs.lucene.CagraBuildOptions.floating;
import static com.nvidia.cuvs.lucene.CagraBuildOptions.integer;
import static com.nvidia.cuvs.lucene.CagraBuildOptions.requiredInteger;

import com.nvidia.cuvs.CagraIndexParams.CodebookGen;
import com.nvidia.cuvs.CagraIndexParams.CudaDataType;
import com.nvidia.cuvs.CuVSIvfPqIndexParams;
import com.nvidia.cuvs.CuVSIvfPqParams;
import com.nvidia.cuvs.CuVSIvfPqSearchParams;
import java.util.Map;

/** Binding-factory settings; omitted shape-dependent values resolve per field/segment, not corpus. */
record IvfPqBuildOptions(
    Integer lists,
    Integer probes,
    Integer pqDimensions,
    int pqBits,
    int trainingIterations,
    double trainingFraction,
    int maxTrainingPointsPerCode,
    float refinementRate,
    CodebookGen codebook,
    boolean forceRotation,
    CudaDataType lookupTableType,
    CudaDataType distanceType) {
  private static final int DEFAULT_PROBES = 32;
  private static final int PQ_DIMENSION_ALIGNMENT = 32;
  private static final int DIMENSIONS_PER_PQ_SUBSPACE = 4;

  static IvfPqBuildOptions parse(Map<String, Object> request) {
    Integer lists = optionalInteger(request, "n_lists", 1);
    Integer probes = optionalInteger(request, "n_probes", 1);
    if (lists != null && probes != null && probes > lists) {
      throw new IllegalArgumentException("n_probes must not exceed n_lists");
    }
    Integer dimensions = optionalInteger(request, "pq_dim", 0);
    int bits = integer(request, "pq_bits", 8, 4, 8);
    if (dimensions != null && (long) dimensions * bits % 8 != 0) {
      throw new IllegalArgumentException("pq_dim * pq_bits must be a multiple of 8");
    }
    double fraction = floating(request, "kmeans_trainset_fraction", 0.25);
    if (fraction <= 0 || fraction > 1) {
      throw new IllegalArgumentException("kmeans_trainset_fraction must be in (0, 1]");
    }
    float refinement = (float) floating(request, "refinement_rate", 2.0);
    // The native CAGRA C API leaves its heuristic unchanged for rates <= 1, including 1.
    if (!Float.isFinite(refinement) || refinement <= 1) {
      throw new IllegalArgumentException(
          "refinement_rate must be finite and greater than 1 as a float");
    }
    boolean rotation = false;
    if (request.containsKey("force_random_rotation")) {
      if (!(request.get("force_random_rotation") instanceof Boolean value)) {
        throw new IllegalArgumentException("force_random_rotation must have type Boolean");
      }
      rotation = value;
    }
    var lookupTableType =
        dtype(choice(request, "lut_dtype", "float32", "float32", "float16", "uint8"));
    var distanceType =
        dtype(choice(request, "internal_distance_dtype", "float32", "float32", "float16"));
    // Native IVF-PQ search requires distance storage at least as wide as each LUT entry.
    if (distanceType == CudaDataType.CUDA_R_16F && lookupTableType == CudaDataType.CUDA_R_32F) {
      throw new IllegalArgumentException(
          "internal_distance_dtype=float16 requires lut_dtype=float16 or uint8");
    }
    return new IvfPqBuildOptions(
        lists,
        probes,
        dimensions,
        bits,
        integer(request, "kmeans_n_iters", 10, 1, Integer.MAX_VALUE),
        fraction,
        integer(request, "max_train_points_per_pq_code", 256, 1, Integer.MAX_VALUE),
        refinement,
        choice(request, "codebook_kind", "per_subspace", "per_subspace", "per_cluster")
                .equals("per_subspace")
            ? CodebookGen.PER_SUBSPACE
            : CodebookGen.PER_CLUSTER,
        rotation,
        lookupTableType,
        distanceType);
  }

  CuVSIvfPqParams resolve(long vectorCount, long vectorDimension) {
    if (vectorCount < 1
        || vectorCount > Integer.MAX_VALUE
        || vectorDimension < 1
        || vectorDimension > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "IVF-PQ segment shape must have positive int-sized extents");
    }
    int resolvedLists = lists != null ? lists : (int) Math.sqrt(vectorCount);
    int resolvedProbes = probes != null ? probes : Math.min(DEFAULT_PROBES, resolvedLists);
    if (resolvedProbes > resolvedLists) {
      throw new IllegalArgumentException("n_probes must not exceed n_lists for this segment");
    }
    // Round D/4 upward to a whole group of 32 PQ subspaces, without integer-division truncation.
    // This changes the temporary PQ representation, never the input vectors or Lucene field shape.
    int defaultDimensions =
        (int) Math.ceilDiv(vectorDimension, DIMENSIONS_PER_PQ_SUBSPACE * PQ_DIMENSION_ALIGNMENT)
            * PQ_DIMENSION_ALIGNMENT;
    int resolvedDimensions = pqDimensions != null ? pqDimensions : defaultDimensions;
    var index =
        new CuVSIvfPqIndexParams.Builder()
            .withNLists(resolvedLists)
            .withKmeansNIters(trainingIterations)
            .withKmeansTrainsetFraction(trainingFraction)
            .withPqDim(resolvedDimensions)
            .withPqBits(pqBits)
            .withMaxTrainPointsPerPqCode(maxTrainingPointsPerCode)
            .withForceRandomRotation(forceRotation)
            .withCodebookKind(codebook)
            .build();
    var search =
        new CuVSIvfPqSearchParams.Builder()
            .withNProbes(resolvedProbes)
            .withLutDtype(lookupTableType)
            .withInternalDistanceDtype(distanceType)
            .build();
    return new CuVSIvfPqParams.Builder()
        .withCuVSIvfPqIndexParams(index)
        .withCuVSIvfPqSearchParams(search)
        .withRefinementRate(refinementRate)
        .build();
  }

  private static Integer optionalInteger(Map<String, Object> request, String key, int minimum) {
    return request.containsKey(key)
        ? requiredInteger(request, key, minimum, Integer.MAX_VALUE)
        : null;
  }

  private static CudaDataType dtype(String name) {
    return switch (name) {
      case "float16" -> CudaDataType.CUDA_R_16F;
      case "uint8" -> CudaDataType.CUDA_R_8U;
      default -> CudaDataType.CUDA_R_32F;
    };
  }
}
