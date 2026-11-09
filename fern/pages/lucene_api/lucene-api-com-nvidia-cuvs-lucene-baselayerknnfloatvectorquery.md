---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-baselayerknnfloatvectorquery
---

# BaseLayerKnnFloatVectorQuery

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class BaseLayerKnnFloatVectorQuery extends KnnFloatVectorQuery
```

A CPU float-vector query that selects a starting point for a single-layer HNSW graph by scoring
random vector ordinals. The best sampled ordinal seeds one Lucene HNSW traversal; this does not
run a separate search for every sample.

Sampling is with replacement and deterministic for the target, field, segment document base,
and vector count. It includes physically stored vectors that may be deleted or rejected by a
filter, just as ordinary HNSW traversal may use those nodes. Lucene still enforces the original
filter and live-document mask on returned results. Each sample consumes one visited-vector unit;
the winning node is scored and counted again by Lucene when traversal starts.

Only unquantized float-vector readers exposing a one-level `HnswGraph` are seeded.
Quantized readers, other graph topologies, non-codec readers, and explicitly seeded strategies
retain Lucene's search behavior. The query does not change an index or require a GPU. Lucene
continues to own exact-search fallback, result collection, and multi-segment search.

## Public Members

### BaseLayerKnnFloatVectorQuery

```java
public BaseLayerKnnFloatVectorQuery(String field, float[] target, int k)
```

Creates an unfiltered query using `DEFAULT_NUM_ENTRY_POINTS` samples.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/BaseLayerKnnFloatVectorQuery.java:54`_

### BaseLayerKnnFloatVectorQuery

```java
public BaseLayerKnnFloatVectorQuery(String field, float[] target, int k, Query filter)
```

Creates a filtered query using `DEFAULT_NUM_ENTRY_POINTS` samples.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/BaseLayerKnnFloatVectorQuery.java:59`_

### BaseLayerKnnFloatVectorQuery

```java
public BaseLayerKnnFloatVectorQuery( String field, float[] target, int k, Query filter, int numEntryPoints)
```

Creates a query with the requested number of random samples per eligible segment search.

**Parameters**

| Name | Description |
| --- | --- |
| `field` | vector field |
| `target` | query vector |
| `k` | number of results, at least one |
| `filter` | optional document filter |
| `numEntryPoints` | number of samples, nonnegative; zero disables random entry selection |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/BaseLayerKnnFloatVectorQuery.java:72`_

### getNumEntryPoints

```java
public int getNumEntryPoints()
```

Returns the number of candidate entry-point samples, not the number of graph traversals.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/BaseLayerKnnFloatVectorQuery.java:83`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/BaseLayerKnnFloatVectorQuery.java:49`_
