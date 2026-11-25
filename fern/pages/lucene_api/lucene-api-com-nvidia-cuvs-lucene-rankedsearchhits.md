---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-rankedsearchhits
---

# RankedSearchHits

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class RankedSearchHits implements BiFunction<IndexReader, TopDocs, byte[]>
```

Reads external IDs from the same reader that produced the hits and returns them in score order.

## Public Members

### apply

```java
@Override public byte[] apply(IndexReader reader, TopDocs results)
```

Packs one little-endian (signed long ID, float score) per hit. A null result requests the
caller's existing per-hit path, which also owns the diagnostic for a missing external ID.
An empty result is a zero-length array.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/RankedSearchHits.java:27`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/RankedSearchHits.java:18`_
