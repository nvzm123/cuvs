---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvsbenchfbinindexingbridge
---

# CuvsBenchFbinIndexingBridge

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class CuvsBenchFbinIndexingBridge implements Function<Map<String, Object>, Map<String, Object>>
```

Streams a validated FBIN prefix through ordinary Lucene `IndexWriter#addDocument` calls.

This class is a narrow interoperability bridge for generated bindings that cannot directly
wrap cuVS-specific classes. It intentionally implements only standard `Function` and
`Map` types. It is not a bulk, mapped, external-vector, or out-of-core index writer: every
vector is added to a normal Lucene writer, and the resulting index is self-contained.

The historical `num_indexing_threads` request key denotes a number of equal contiguous
partitions. Those partitions are built sequentially with one actual indexing thread, using
`CREATE` mode for the first and `APPEND` mode thereafter. Automatic and final
merges are disabled.

The target must be an existing, empty, non-symbolic-link directory. This bridge never removes,
replaces, or publishes that directory. Its caller remains responsible for validating and
atomically installing the staged index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuvsBenchFbinIndexingBridge.java:60`_
