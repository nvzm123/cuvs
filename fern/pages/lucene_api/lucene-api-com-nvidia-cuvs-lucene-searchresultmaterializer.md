---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-searchresultmaterializer
---

# SearchResultMaterializer

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class SearchResultMaterializer implements Function<Map<String, Object>, Map<String, Object>>
```

Copies search hits into primitive dataset-ID and score arrays in their original result order.

The request contains `reader` (`IndexReader`), `top_docs` (`TopDocs`)
from that reader, and `id_field` (a `String` naming a numeric doc-values field).
The response contains `document_ids` (`long[]`) and `scores` (`float[]`).
These new arrays remain valid after the reader closes; this class neither closes the reader nor
retains it. Missing IDs are errors, not a reason to substitute Lucene's internal document IDs.

`Function` and `Map` let generated bindings call this helper without wrapping a new
Java type. Retrieval stays separate from `IndexSearcherTimingBridge`'s search-only timer.
An I/O failure is exposed as an `UncheckedIOException`.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/SearchResultMaterializer.java:31`_
