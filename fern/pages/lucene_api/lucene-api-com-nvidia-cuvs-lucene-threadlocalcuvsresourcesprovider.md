---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-threadlocalcuvsresourcesprovider
---

# ThreadLocalCuVSResourcesProvider

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class ThreadLocalCuVSResourcesProvider
```

Provides a mechanism to create ThreadLocal based CuVSResource instances.

## Public Members

### getCuVSResourcesInstance

```java
public static CuVSResources getCuVSResourcesInstance()
```

Gets the caller-owned resources used by the accessing thread for index construction and
serialization, merge work, and query execution.

Retained native indexes loaded by `CuVS2510GPUVectorsReader` use independently owned
resources from the `CuVSReaderResourcesFactory` configured on the vectors format or
codec. Readers neither retain nor close this thread-local instance.

**Returns**

an instance of CuVSResources

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:35`_

### createIndependentCuVSResourcesInstance

```java
static CuVSResources createIndependentCuVSResourcesInstance()
```

Creates an independently owned resources instance for long-lived index allocations.

The caller owns the returned instance and must close it. This is intended for native
allocations whose lifetime is tied to a long-lived object rather than to the current thread.
It intentionally does not reserve the per-query workspace pool configured by \{@link
#WORKSPACE_POOL_SIZE_PROPERTY\}.

This method backs the default `CuVSReaderResourcesFactory`. Applications that need a
custom temporary directory, memory tracking, or other reader-specific resource configuration
should supply their own factory to `CuVS2510GPUVectorsFormat` or \{@link
CuVS2510GPUSearchCodec\}. Thread-local resources installed through \{@link
#setCuVSResourcesInstance(CuVSResources)\} remain construction, serialization, query, and merge
resources; readers do not take ownership of them.

**Returns**

a new resources instance, or `null` when cuVS is unavailable

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:56`_

### createRequiredIndependentCuVSResourcesInstance

```java
static CuVSResources createRequiredIndependentCuVSResourcesInstance()
```

Creates independently owned reader resources or fails when cuVS is unavailable.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:61`_

### setCuVSResourcesInstance

```java
public static void setCuVSResourcesInstance(CuVSResources resources)
```

Sets the caller-owned resources used by the current thread for index construction and
serialization, merge work, and query execution.

This does not configure the resources that own retained reader indexes. Supply a \{@link
CuVSReaderResourcesFactory\} to `CuVS2510GPUVectorsFormat` or \{@link
CuVS2510GPUSearchCodec\} when reader-specific resource configuration is required.

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | the instance of CuVSResources to set |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:79`_

### resolveWorkspacePoolBytes

```java
static long resolveWorkspacePoolBytes(String raw)
```

Resolves a raw workspace-pool property value to a 256-byte-aligned size. Zero or an absent
value disables the per-resources pool. Invalid, negative, or unalignable values warn and also
disable it.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:123`_

### closeCuVSResourcesInstance

```java
public static void closeCuVSResourcesInstance()
```

Attempts to close the thread's `CuVSResources` instance.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:165`_

### assertIsSupported

```java
public static void assertIsSupported() throws UnsupportedOperationException
```

Checks if cuVS is supported and throws `UnsupportedOperationException` otherwise.

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` |  |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:181`_

### isSupported

```java
public static boolean isSupported()
```

Checks if cuVS is supported.

**Returns**

true if cuVS is supported else false

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:192`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:16`_
