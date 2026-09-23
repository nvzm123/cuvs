/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.OutputStream;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import org.junit.Test;

public class CagraIndexCompatibilityTest {

  @Test
  public void graphOnlyMethodsRemainOptionalForIndexImplementations() throws Exception {
    assertTrue(CagraIndex.class.getMethod("serializeGraph", OutputStream.class).isDefault());
    assertTrue(
        CagraIndex.class.getMethod("serializeGraph", OutputStream.class, int.class).isDefault());
    assertTrue(
        CagraIndex.class.getMethod("serializeGraph", OutputStream.class, Path.class).isDefault());
    assertTrue(
        CagraIndex.class
            .getMethod("serializeGraph", OutputStream.class, Path.class, int.class)
            .isDefault());

    CagraIndex index = defaultsOnlyProxy(CagraIndex.class);
    UnsupportedOperationException exception =
        assertThrows(
            UnsupportedOperationException.class,
            () -> index.serializeGraph(OutputStream.nullOutputStream()));
    assertEquals(
        "Graph-only serialization is not supported by this CagraIndex implementation",
        exception.getMessage());
  }

  @Test
  public void graphOnlyDeserializationRemainsOptionalForBuilderImplementations() throws Exception {
    assertTrue(
        CagraIndex.Builder.class.getMethod("fromGraph", java.io.InputStream.class).isDefault());

    CagraIndex.Builder builder = defaultsOnlyProxy(CagraIndex.Builder.class);
    UnsupportedOperationException exception =
        assertThrows(
            UnsupportedOperationException.class,
            () -> builder.fromGraph(new ByteArrayInputStream(new byte[0])));
    assertEquals(
        "Graph-only deserialization is not supported by this CagraIndex.Builder implementation",
        exception.getMessage());
  }

  private static <T> T defaultsOnlyProxy(Class<T> interfaceClass) {
    Object proxy =
        Proxy.newProxyInstance(
            interfaceClass.getClassLoader(),
            new Class<?>[] {interfaceClass},
            (instance, method, arguments) -> {
              if (method.isDefault()) {
                return InvocationHandler.invokeDefault(instance, method, arguments);
              }
              throw new AssertionError("Unexpected abstract method invocation: " + method);
            });
    return interfaceClass.cast(proxy);
  }
}
