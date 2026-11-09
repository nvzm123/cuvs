---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-baselayerknnfloatvectorqueryfactory
---

# BaseLayerKnnFloatVectorQueryFactory

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class BaseLayerKnnFloatVectorQueryFactory implements Function<Map<String, Object>, Query>
```

Binding-compatible factory for `BaseLayerKnnFloatVectorQuery`. The request requires `field` (`String`), `target` (`float[]`), and `k` (`Integer`). Optional
`num_entry_points` is a nonnegative `Integer` controlling the sample count; zero
disables random entry selection. When omitted, it uses `BaseLayerKnnFloatVectorQuery#DEFAULT_NUM_ENTRY_POINTS`. No other entries are accepted. The factory
creates unfiltered queries, returns the base Lucene `Query` type, and does not load a native
library.

## Public Members

### apply

```java
@Override public Query apply(Map<String, Object> request)
```

Creates one unfiltered query from the required arguments and optional sample count.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/BaseLayerKnnFloatVectorQueryFactory.java:28`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/BaseLayerKnnFloatVectorQueryFactory.java:23`_
