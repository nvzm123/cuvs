/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.CagraBuildOptions.choice;
import static com.nvidia.cuvs.lucene.CagraBuildOptions.floating;
import static com.nvidia.cuvs.lucene.CagraBuildOptions.integer;

import com.nvidia.cuvs.CagraIndexParams.CodebookGen;
import com.nvidia.cuvs.CagraIndexParams.CudaDataType;
import com.nvidia.cuvs.CuVSIvfPqIndexParams;
import com.nvidia.cuvs.CuVSIvfPqParams;
import com.nvidia.cuvs.CuVSIvfPqSearchParams;
import java.util.Map;

/** Validate binding overrides, using native heuristic refinement with Java defaults otherwise. */
final class IvfPqBuildOptions {
  private IvfPqBuildOptions() {}

  static CuVSIvfPqParams parse(Map<String, Object> request) {
    // Match native heuristic refinement without changing the shared cuVS Java builder's defaults.
    var defaults = new CuVSIvfPqParams.Builder().withRefinementRate(1.0f).build();
    var indexDefaults = defaults.getIndexParams();
    int lists = integer(request, "n_lists", indexDefaults.getnLists(), 1, Integer.MAX_VALUE);
    int probes =
        integer(request, "n_probes", defaults.getSearchParams().getnProbes(), 1, Integer.MAX_VALUE);
    if (probes > lists) {
      throw new IllegalArgumentException("n_probes must not exceed n_lists");
    }
    int dimensions = integer(request, "pq_dim", indexDefaults.getPqDim(), 0, Integer.MAX_VALUE);
    int bits = integer(request, "pq_bits", indexDefaults.getPqBits(), 4, 8);
    if ((long) dimensions * bits % 8 != 0) {
      throw new IllegalArgumentException("pq_dim * pq_bits must be a multiple of 8");
    }
    double fraction =
        floating(request, "kmeans_trainset_fraction", indexDefaults.getKmeansTrainsetFraction());
    if (fraction <= 0 || fraction > 1) {
      throw new IllegalArgumentException("kmeans_trainset_fraction must be in (0, 1]");
    }
    double requestedRefinement = floating(request, "refinement_rate", defaults.getRefinementRate());
    float refinement = (float) requestedRefinement;
    // Exactly 1 retains native heuristic refinement. A larger request must survive float rounding.
    if (!Float.isFinite(refinement)
        || requestedRefinement < 1
        || (requestedRefinement > 1 && refinement == 1)) {
      throw new IllegalArgumentException(
          "refinement_rate must be 1 or remain finite and greater than 1 as a float");
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
    var codebook =
        choice(request, "codebook_kind", "per_subspace", "per_subspace", "per_cluster")
                .equals("per_subspace")
            ? CodebookGen.PER_SUBSPACE
            : CodebookGen.PER_CLUSTER;
    var index =
        new CuVSIvfPqIndexParams.Builder()
            .withNLists(lists)
            .withKmeansNIters(
                integer(
                    request,
                    "kmeans_n_iters",
                    indexDefaults.getKmeansNIters(),
                    1,
                    Integer.MAX_VALUE))
            .withKmeansTrainsetFraction(fraction)
            .withPqDim(dimensions)
            .withPqBits(bits)
            .withMaxTrainPointsPerPqCode(
                integer(
                    request,
                    "max_train_points_per_pq_code",
                    indexDefaults.getMaxTrainPointsPerPqCode(),
                    1,
                    Integer.MAX_VALUE))
            .withForceRandomRotation(rotation)
            .withCodebookKind(codebook)
            .build();
    var search =
        new CuVSIvfPqSearchParams.Builder()
            .withNProbes(probes)
            .withLutDtype(lookupTableType)
            .withInternalDistanceDtype(distanceType)
            .build();
    return new CuVSIvfPqParams.Builder()
        .withCuVSIvfPqIndexParams(index)
        .withCuVSIvfPqSearchParams(search)
        .withRefinementRate(refinement)
        .build();
  }

  private static CudaDataType dtype(String name) {
    return switch (name) {
      case "float16" -> CudaDataType.CUDA_R_16F;
      case "uint8" -> CudaDataType.CUDA_R_8U;
      default -> CudaDataType.CUDA_R_32F;
    };
  }
}
