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

Gets an instance of CuVSResources for the accessing thread.

**Returns**

an instance of CuVSResources

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:31`_

### setCuVSResourcesInstance

```java
public static void setCuVSResourcesInstance(CuVSResources resources)
```

Sets the instance of CuVSResources

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | the instance of CuVSResources to set |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:40`_

### resolveWorkspacePoolBytes

```java
static long resolveWorkspacePoolBytes(String raw)
```

Resolves a raw workspace-pool property value to a 256-byte-aligned size. Zero or an absent
value disables the per-resources pool. Invalid, negative, or unalignable values warn and also
disable it.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:81`_

### closeCuVSResourcesInstance

```java
public static void closeCuVSResourcesInstance()
```

Attempts to close the thread's `CuVSResources` instance.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:123`_

### assertIsSupported

```java
public static void assertIsSupported() throws UnsupportedOperationException
```

Checks if cuVS is supported and throws `UnsupportedOperationException` otherwise.

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` |  |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:136`_

### isSupported

```java
public static boolean isSupported()
```

Checks if cuVS is supported.

**Returns**

true if cuVS is supported else false

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:147`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/ThreadLocalCuVSResourcesProvider.java:16`_
