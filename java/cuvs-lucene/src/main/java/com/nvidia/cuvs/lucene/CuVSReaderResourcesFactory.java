/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSResources;

/**
 * Creates independently owned cuVS resources for long-lived vector readers.
 *
 * <p>Each successful invocation must return a new, non-null {@link CuVSResources} instance that is
 * not shared with another reader or with a query thread. Ownership transfers to the reader, which
 * closes the resources after closing every native index and matrix allocated from them. If creation
 * fails before an instance is returned, the factory remains responsible for cleaning up any
 * partially created resources.
 *
 * <p>The factory is invoked during construction of each non-merge vector reader. Readers opened for
 * Lucene merges do not retain GPU indexes and therefore do not invoke it. Implementations must
 * permit concurrent invocations because Lucene may open readers concurrently.
 *
 * @since 26.12
 */
@FunctionalInterface
public interface CuVSReaderResourcesFactory {

  /**
   * Creates resources whose ownership will transfer to one vector reader.
   *
   * @return a new, non-null resources instance
   * @throws Throwable if the resources cannot be created
   */
  CuVSResources create() throws Throwable;
}
