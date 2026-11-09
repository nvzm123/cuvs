---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-gpusearchcodecfactory
---

# GPUSearchCodecFactory

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class GPUSearchCodecFactory implements Function<Map<String, Object>, Map<String, Object>>
```

Stateless, binding-compatible factory for codecs that build and search CAGRA indexes on the GPU.

An empty request preserves `GPUSearchParams.Builder` defaults, including heuristic mode
and 32 writer threads. The response contains a single `codec` entry. This factory configures
index construction, not the parameters of individual queries.

Optional controls are `build_strategy` (`heuristic` or `custom`),
`graph_degree` (1..512), `intermediate_graph_degree` (2..512),
`graph_build_algo` (`nn_descent`), `nn_descent_n_iters`
(1..100, NN-descent only), `writer_threads` (1..512), and `build_quality`
(0..Integer.MAX_VALUE, default 7, heuristic only). Counts must be `Integer`; enum names are
lowercase strings.
Explicit geometry, algorithm or iteration options infer custom mode. Explicit heuristic
mode accepts graph_degree but rejects other custom controls it would ignore. Custom graph degree
defaults to 64; intermediate degree defaults to `max(128, graph_degree)` and must not be
below graph_degree.

Build quality is a construction-effort setting, not a percentage or a 0..10 score. Native cuVS
describes values below 20 as most practical. Below one million vectors, its current dataset
heuristic uses `5 + build_quality` maximum NN-descent iterations. The integer type limit
is not a recommended tuning range or a guarantee that native construction will succeed.

## Public Members

### apply

```java
@Override public Map<String, Object> apply(Map<String, Object> request)
```

Constructs one GPU CAGRA codec.

**Parameters**

| Name | Description |
| --- | --- |
| `request` | optional build controls; an empty map preserves defaults |

**Returns**

the configured codec under `codec`

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | for null, unknown, invalid or incompatible options |
| `IllegalStateException` | if codec construction or vector-format initialization fails |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchCodecFactory.java:47`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchCodecFactory.java:36`_
