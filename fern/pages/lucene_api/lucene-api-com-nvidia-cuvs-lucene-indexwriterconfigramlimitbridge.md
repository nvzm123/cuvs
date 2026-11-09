---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-indexwriterconfigramlimitbridge
---

# IndexWriterConfigRAMLimitBridge

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class IndexWriterConfigRAMLimitBridge implements Function<Map<String, Object>, Map<String, Object>>
```

Applies and verifies Lucene's per-thread indexing-memory limit.

Lucene 10.2 accepts limits below 2048 MiB through its public setter. Larger limits require an
explicit opt-in and are applied to Lucene's non-public `perThreadHardLimitMB` field. That
unsupported override is intended only for controlled cuVS Bench vector-only builds that disable
automatic merges and validate their final segment topology. It must not be treated as a general
replacement for Lucene's safety limit.

The standard `Function` and `Map` types provide a narrow bridge for generated Java
bindings that do not expose reflection. The request must contain `config` (an
`IndexWriterConfig`), `per_thread_hard_limit_mb` (a positive `Integer`), and
`allow_unsupported_lucene_ram_limit` (a `Boolean`). The response returns the same
config, the verified limit, and `application_mode`, which is either `public_setter`
or `unsupported_field_override`.

The non-public path deliberately depends on Lucene's field name and type. It fails if the
field cannot be found, made accessible, written, or read back, so callers never silently continue
with a different limit.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/IndexWriterConfigRAMLimitBridge.java:35`_
