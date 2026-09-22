/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSResources;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Provides a mechanism to create ThreadLocal based CuVSResource instances.
 *
 * @since 26.02
 */
public class ThreadLocalCuVSResourcesProvider {

  private static final Logger log =
      Logger.getLogger(ThreadLocalCuVSResourcesProvider.class.getName());
  private static final ThreadLocal<CuVSResources> cuVSResources;

  static {
    cuVSResources = ThreadLocal.withInitial(() -> cuVSResourcesOrNull(true));
  }

  /**
   * Gets the caller-owned resources used by the accessing thread for index construction and
   * serialization, merge work, and query execution.
   *
   * <p>Retained native indexes loaded by {@link CuVS2510GPUVectorsReader} use independently owned
   * resources from the {@link CuVSReaderResourcesFactory} configured on the vectors format or
   * codec. Readers neither retain nor close this thread-local instance.
   *
   * @return an instance of CuVSResources
   */
  public static CuVSResources getCuVSResourcesInstance() {
    return cuVSResources.get();
  }

  /**
   * Creates an independently owned resources instance for long-lived index allocations.
   *
   * <p>The caller owns the returned instance and must close it. This is intended for native
   * allocations whose lifetime is tied to a long-lived object rather than to the current thread.
   * It intentionally does not reserve the per-query workspace pool configured by {@link
   * #WORKSPACE_POOL_SIZE_PROPERTY}.
   *
   * <p>This method backs the default {@link CuVSReaderResourcesFactory}. Applications that need a
   * custom temporary directory, memory tracking, or other reader-specific resource configuration
   * should supply their own factory to {@link CuVS2510GPUVectorsFormat} or {@link
   * CuVS2510GPUSearchCodec}. Thread-local resources installed through {@link
   * #setCuVSResourcesInstance(CuVSResources)} remain construction, serialization, query, and merge
   * resources; readers do not take ownership of them.
   *
   * @return a new resources instance, or {@code null} when cuVS is unavailable
   */
  static CuVSResources createIndependentCuVSResourcesInstance() {
    return cuVSResourcesOrNull(false);
  }

  /** Creates independently owned reader resources or fails when cuVS is unavailable. */
  static CuVSResources createRequiredIndependentCuVSResourcesInstance() {
    CuVSResources resources = createIndependentCuVSResourcesInstance();
    if (resources == null) {
      throw new UnsupportedOperationException("cuVS is not supported");
    }
    return resources;
  }

  /**
   * Sets the caller-owned resources used by the current thread for index construction and
   * serialization, merge work, and query execution.
   *
   * <p>This does not configure the resources that own retained reader indexes. Supply a {@link
   * CuVSReaderResourcesFactory} to {@link CuVS2510GPUVectorsFormat} or {@link
   * CuVS2510GPUSearchCodec} when reader-specific resource configuration is required.
   *
   * @param resources the instance of CuVSResources to set
   */
  public static void setCuVSResourcesInstance(CuVSResources resources) {
    cuVSResources.set(resources);
  }

  /** System property controlling the workspace pool size per resources handle (in bytes). */
  public static final String WORKSPACE_POOL_SIZE_PROPERTY = "com.nvidia.cuvs.workspacePoolSize";

  private static final long RMM_ALIGNMENT_BYTES = 256;

  private static CuVSResources cuVSResourcesOrNull(boolean configureWorkspacePool) {
    CuVSResources resources = null;
    try {
      // Resolve configuration before allocating resources so malformed input cannot leak a newly
      // created native handle and pinned host buffer.
      long poolBytes =
          configureWorkspacePool
              ? resolveWorkspacePoolBytes(System.getProperty(WORKSPACE_POOL_SIZE_PROPERTY))
              : 0;
      resources = CuVSResources.create();
      if (poolBytes > 0) {
        resources.setWorkspacePool(poolBytes);
      }
      return resources;
    } catch (UnsupportedOperationException uoe) {
      closeAfterFailedInitialization(resources, uoe);
      log.log(
          Level.WARNING,
          "cuVS is not supported on this platform or java version: " + uoe.getMessage());
    } catch (Throwable t) {
      Throwable failure = t;
      if (t instanceof ExceptionInInitializerError ex && ex.getCause() != null) {
        failure = ex.getCause();
      }
      closeAfterFailedInitialization(resources, failure);
      log.log(Level.WARNING, "Exception occurred during creation of cuVS resources. " + failure);
    }
    return null;
  }

  /**
   * Resolves a raw workspace-pool property value to a 256-byte-aligned size. Zero or an absent
   * value disables the per-resources pool. Invalid, negative, or unalignable values warn and also
   * disable it.
   */
  static long resolveWorkspacePoolBytes(String raw) {
    if (raw == null) return 0;

    final long requestedBytes;
    try {
      requestedBytes = Long.parseLong(raw.trim());
    } catch (NumberFormatException invalid) {
      warnInvalidWorkspacePoolSize(raw);
      return 0;
    }

    if (requestedBytes == 0) return 0;
    if (requestedBytes < 0 || requestedBytes > Long.MAX_VALUE - (RMM_ALIGNMENT_BYTES - 1)) {
      warnInvalidWorkspacePoolSize(raw);
      return 0;
    }

    return (requestedBytes + (RMM_ALIGNMENT_BYTES - 1)) & ~(RMM_ALIGNMENT_BYTES - 1);
  }

  private static void warnInvalidWorkspacePoolSize(String raw) {
    log.warning(
        "Invalid "
            + WORKSPACE_POOL_SIZE_PROPERTY
            + " value \""
            + raw
            + "\"; expected a non-negative byte count that can be aligned to 256 bytes. "
            + "Continuing without a workspace pool.");
  }

  private static void closeAfterFailedInitialization(CuVSResources resources, Throwable failure) {
    if (resources == null) return;
    try {
      resources.close();
    } catch (Throwable closeFailure) {
      failure.addSuppressed(closeFailure);
    }
  }

  /**
   * Attempts to close the thread's {@link CuVSResources} instance.
   */
  public static void closeCuVSResourcesInstance() {
    CuVSResources r = cuVSResources.get();
    try {
      if (r != null) {
        r.close();
      }
    } finally {
      cuVSResources.remove();
    }
  }

  /**
   * Checks if cuVS is supported and throws {@link UnsupportedOperationException} otherwise.
   *
   * @throws UnsupportedOperationException
   */
  public static void assertIsSupported() throws UnsupportedOperationException {
    if (cuVSResources.get() == null) {
      throw new UnsupportedOperationException("cuVS is not supported");
    }
  }

  /**
   * Checks if cuVS is supported.
   *
   * @return true if cuVS is supported else false
   */
  public static boolean isSupported() {
    return cuVSResources.get() != null;
  }
}
