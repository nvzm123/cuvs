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
`graph_build_algo` (`nn_descent` or `ivf_pq`), `nn_descent_n_iters`
(1..100, NN-descent only), `writer_threads` (1..512), and `build_quality`
(0..Integer.MAX_VALUE, default 7, heuristic only). Counts must be `Integer`; enum names are
lowercase strings.
Explicit geometry, algorithm, iteration or IVF-PQ options infer custom mode. Explicit heuristic
mode accepts graph_degree but rejects other custom controls it would ignore. Custom graph degree
defaults to 64; intermediate degree defaults to `max(128, graph_degree)` and must not be
below graph_degree.

Build quality is a construction-effort setting, not a percentage or a 0..10 score. Native cuVS
describes values below 20 as most practical. Its current dataset heuristic uses
`5 + build_quality` maximum NN-descent iterations below one million vectors, otherwise
`round(2 + sqrt(n_lists) / 20 + build_quality)` IVF-PQ probes. The integer type limit is not
a recommended tuning range or a guarantee that native construction will succeed.

IVF-PQ construction controls require explicit `graph_build_algo=ivf_pq`: positive Integer
`n_lists`, `n_probes`, `kmeans_n_iters`, `max_train_points_per_pq_code`;
nonnegative Integer `pq_dim` (0 selects the native heuristic); Integer `pq_bits`
(4..8); Float or Double `kmeans_trainset_fraction` (finite, in (0,1]) and
`refinement_rate` (finite and greater than 1 after conversion to float); Boolean
`force_random_rotation`; `codebook_kind` (`per_subspace` or `per_cluster`);
`lut_dtype` (`float32`, `float16`, `uint8`); and
`internal_distance_dtype` (`float32` or `float16`).
A float16 internal distance requires a float16 or uint8 LUT; an omitted LUT defaults to float32.
The product `pq_dim * pq_bits` must be a multiple of 8, and explicit n_probes must not exceed n_lists.
Omitted n_lists is the floor of the square root of the field's vector count in the segment being
built; omitted n_probes is min(32, n_lists). Omitted pq_dim is one quarter of the vector dimension,
rounded upward to a multiple of 32. Other custom defaults include pq_bits=8, kmeans_n_iters=10,
and kmeans_trainset_fraction=0.25. Explicit pq_dim=0 still selects cuVS's own PQ heuristic.
These binding-factory defaults do not change the native HEURISTIC strategy or existing Java
parameter-builder defaults. Shape-dependent values resolve when the segment is built.
These IVF-PQ search controls construct the CAGRA graph; they do not configure CAGRA queries.

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchCodecFactory.java:66`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchCodecFactory.java:55`_
