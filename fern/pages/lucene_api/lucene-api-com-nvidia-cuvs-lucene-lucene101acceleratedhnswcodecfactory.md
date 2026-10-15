---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene101acceleratedhnswcodecfactory
---

# Lucene101AcceleratedHNSWCodecFactory

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class Lucene101AcceleratedHNSWCodecFactory implements Function<Map<String, Object>, Map<String, Object>>
```

Constructs an accelerated HNSW codec from one self-contained parameter request.

The standard `Function` and `Map` types form a narrow bridge for generated Java
bindings that do not wrap parameterized constructors. The request has exactly two entries,
max_conn and beam_width. Both must be `Integer` values in the inclusive range 1 through
512. The response contains the configured `codec` and the verified applied values under the
same parameter keys. Invalid requests throw `IllegalArgumentException`; codec construction
or initialization failures throw `IllegalStateException`.

The factory is stateless. In particular, it does not use JVM system properties, so concurrent
callers cannot observe or overwrite one another's configuration.

## Public Members

### apply

```java
@Override public Map<String, Object> apply(Map<String, Object> request)
```

Constructs one codec from the complete request.

**Parameters**

| Name | Description |
| --- | --- |
| `request` | exactly the `max_conn` and `beam_width` integer entries |

**Returns**

the codec and its verified applied parameter values

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | if the request is null, incomplete, has extra keys, contains non-integer values, or contains values outside the inclusive range 1 through 512 |
| `IllegalStateException` | if codec construction or vector-format initialization fails |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodecFactory.java:42`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodecFactory.java:28`_
