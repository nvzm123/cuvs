---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-gpuknnfloatvectorqueryfactory
---

# GPUKnnFloatVectorQueryFactory

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class GPUKnnFloatVectorQueryFactory implements Function<Map<String, Object>, Query>
```

Constructs a GPU vector query through types exposed by standard generated Java bindings.

The request requires `field` (`String`), `target` (`float[]`), and
`k` (a positive `Integer`). Optional integer entries are `itopk` (defaults
to `k`), `search_width` (1), `max_iterations` (0, automatic), and
`thread_block_size` (0, automatic). The optional `algo` string is one of `auto`
(default), `single_cta`, or `multi_cta`. No other entries are accepted.

An explicit `itopk` must be at least `k`; this bridge rejects a request that the
query would otherwise silently raise to `k`. Other search-parameter limits are enforced by
`GPUKnnFloatVectorQuery`. In particular, SINGLE_CTA supports at most 512 intermediate
results. Requests above 1024 results require the optimized multi-partition CAGRA path and fail
if the index would require per-segment fallback (for example, unequal graph degrees). Accepted
integer ranges are only representational, not a promise that native CAGRA can execute every
accepted combination. Native limits also depend on the index and other search parameters.

The factory is stateless and returns the base Lucene `Query` type, so callers can use
`Function#apply(Object)` without generating wrappers for this class or its constructor.
Query construction does not allocate native resources or execute a GPU search.

## Public Members

### apply

```java
@Override public Query apply(Map<String, Object> request)
```

Creates one unfiltered GPU query from the request.

**Parameters**

| Name | Description |
| --- | --- |
| `request` | required query arguments and optional CAGRA search parameters |

**Returns**

the configured query, exposed as a base Lucene query

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | if the request has unknown keys, missing required values, incorrect types, invalid search parameters, or non-finite vector values |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUKnnFloatVectorQueryFactory.java:65`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUKnnFloatVectorQueryFactory.java:37`_
