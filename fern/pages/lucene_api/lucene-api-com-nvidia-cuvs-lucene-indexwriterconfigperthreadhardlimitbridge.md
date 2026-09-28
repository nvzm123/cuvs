---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-indexwriterconfigperthreadhardlimitbridge
---

# IndexWriterConfigPerThreadHardLimitBridge

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class IndexWriterConfigPerThreadHardLimitBridge implements Function<Map<String, Object>, Map<String, Object>>
```

Sets Lucene's per-thread indexing-memory limit without the public setter's 2048 MiB cap.

The standard `Function` and `Map` types form a narrow bridge for generated Java
bindings that do not wrap this class directly. The request must contain
`config` (a `LiveIndexWriterConfig`) and `per_thread_hard_limit_mb` (a positive
`Integer`). The response contains the same config and the verified applied limit under
those keys.

This bridge deliberately depends on Lucene's non-public field name. It fails if the field
cannot be found, made accessible, written, or verified so that a caller never silently continues
with Lucene's lower default limit.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/IndexWriterConfigPerThreadHardLimitBridge.java:26`_
