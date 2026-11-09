---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-acceleratedhnswcodecfactory
---

# AcceleratedHNSWCodecFactory

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class AcceleratedHNSWCodecFactory implements Function<Map<String, Object>, Map<String, Object>>
```

Constructs an accelerated HNSW codec from one self-contained parameter request.

The standard `Function` and `Map` types form a narrow bridge for generated Java
bindings that do not wrap parameterized constructors. The required entries, max_conn and
beam_width, must be `Integer` values in the inclusive range 1 through 512.
The response contains the configured `codec` and the applied values under the
same parameter keys. Invalid requests throw `IllegalArgumentException`; codec construction
or initialization failures throw `IllegalStateException`.

The factory is stateless. In particular, it does not use JVM system properties, so concurrent
callers cannot observe or overwrite one another's configuration.

Optional build controls are `build_strategy` (`heuristic` or `custom`),
`graph_degree` (1..512), `intermediate_graph_degree` (2..512),
`graph_build_algo` (`nn_descent`), `nn_descent_n_iters`
(1..100, NN-descent only), `hnsw_layers` (1..99), and `writer_threads` (1..512).
Counts must be Integer values; enum names are lowercase strings.

Supplying geometry, algorithm or iteration controls selects custom mode unless an
explicit strategy is supplied. Heuristic mode rejects controls it would ignore. Custom mode
defaults to graph degree `min(64, 2 * max_conn)` and intermediate degree
`max(128, graph_degree)`; the graph degree must not exceed `2 * max_conn` or the
intermediate degree. Heuristic mode retains the SAME_GRAPH_FOOTPRINT default: cuVS derives
`graph_degree = 2 * max_conn` and `intermediate_graph_degree = 3 * max_conn`.
This factory does not expose a heuristic selector.

## Public Members

### apply

```java
@Override public Map<String, Object> apply(Map<String, Object> request)
```

Constructs one codec from the complete request.

**Parameters**

| Name | Description |
| --- | --- |
| `request` | the required `max_conn` and `beam_width` entries and optional controls |

**Returns**

the codec and its verified applied parameter values

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | if the request is null, incomplete, or contains unknown, incorrectly typed, out-of-range or incompatible options |
| `IllegalStateException` | if codec construction or vector-format initialization fails |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWCodecFactory.java:55`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWCodecFactory.java:41`_
