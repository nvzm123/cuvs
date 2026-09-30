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
explicit opt-in and are applied reflectively to Lucene's protected `perThreadHardLimitMB`
field. That unsupported override is intended only for controlled cuVS Bench vector-only builds
that disable automatic merges and validate their final segment topology. It must not be treated
as a general replacement for Lucene's safety limit.

The standard `Function` and `Map` types provide a narrow bridge for generated Java
bindings that do not expose reflection.

An `IndexWriterConfig` must be supplied under `config`. The request also requires a
positive `Integer` under `per_thread_hard_limit_mb` and a `Boolean` under
`allow_unsupported_lucene_ram_limit`. The response returns the same config and verified
limit. The `application_mode` entry reports the application strategy. Its value is either
`public_setter` or `unsupported_field_override`.

The reflective path deliberately depends on Lucene's field name and type. It rejects the
request if the field cannot be found, made accessible, written, or verified through Lucene's
public getter, so callers never silently continue with a different limit.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/IndexWriterConfigRAMLimitBridge.java:37`_
