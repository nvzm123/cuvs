---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswbinaryquantizedcodec
---

# LuceneAcceleratedHNSWBinaryQuantizedCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWBinaryQuantizedCodec extends CuVSFilterCodec
```

cuVS based codec for GPU graph construction and CPU HNSW search. Binary quantization is used
only while building the graph; persisted vectors and CPU scoring remain full precision.

This codec, named `Lucene101AcceleratedHNSWBinaryQuantizedCodec`, wraps `Lucene101Codec`, the default codec of Lucene 10.2. `CuVSCodecs#acceleratedHNSWBinaryQuantized` returns the codec of
this family for the Lucene release in use.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedCodec.java:19`_
