---
slug: api-reference/java-api-com-nvidia-cuvs-cagraindex
---

# CagraIndex

_Java package: `com.nvidia.cuvs`_

```java
public interface CagraIndex extends AutoCloseable
```

`CagraIndex` encapsulates a CAGRA index, along with methods to interact
with it.

CAGRA is a graph-based nearest neighbors algorithm that was built from the
ground up for GPU acceleration. CAGRA demonstrates state-of-the art index
build and query performance for both small and large-batch sized search. Know
more about this algorithm
here

## Public Members

### setDelegate

```java
public final void setDelegate(AutoCloseable delegate, long handleAddress)
```

Internal wiring hook used by the Java wrapper implementation.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:35`_

### isPresent

```java
public final boolean isPresent()
```

Returns true when this view has a native handle.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:43`_

### nativeHandleAddress

```java
public final long nativeHandleAddress()
```

Internal accessor for native handle address.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:50`_

### setDelegate

```java
public final void setDelegate(AutoCloseable delegate)
```

Internal wiring hook used by the Java wrapper implementation.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:85`_

### setDelegate

```java
public final void setDelegate(AutoCloseable delegate, long handleAddress)
```

Internal wiring hook used by the Java wrapper implementation.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:92`_

### isPresent

```java
public final boolean isPresent()
```

Returns true when this handle owns native dataset storage.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:100`_

### nativeHandleAddress

```java
public final long nativeHandleAddress()
```

Internal accessor for native handle address.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:107`_

### close

```java
@Override void close() throws Exception
```

Invokes the native destroy_cagra_index to de-allocate the CAGRA index. Also attempts to close
any dataset whose ownership transferred to this index during construction.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:139`_

### search

```java
SearchResults search(CagraQuery query) throws Throwable
```

Invokes the native search_cagra_index via the Panama API for searching a
CAGRA index.

**Parameters**

| Name | Description |
| --- | --- |
| `query` | an instance of `CagraQuery` holding the query vectors and other parameters |

**Returns**

an instance of `SearchResults` containing the results

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:149`_

### makePaddedDataset

```java
PaddedDataset makePaddedDataset(CuVSMatrix dataset) throws Throwable
```

Create an owning padded dataset by allocating padded storage and copying
`dataset`. Prefer this when the source matrix is not already padded to CAGRA's
required row stride (e.g. unaligned dimensions).

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:156`_

### makePaddedDatasetView

```java
PaddedDatasetView makePaddedDatasetView(CuVSMatrix dataset) throws Throwable
```

Create a caller-owned padded dataset view handle from a matrix that is already
padded to CAGRA's required row stride. For unpadded matrices use
`#makePaddedDataset(CuVSMatrix)`.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:163`_

### makeStandardDatasetView

```java
StandardDatasetView makeStandardDatasetView(CuVSMatrix dataset) throws Throwable
```

Create a caller-owned standard dataset view handle from a matrix.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:166`_

### updateDataset

```java
void updateDataset(PaddedDatasetView datasetView) throws Throwable
```

Update this index with a caller-provided padded device dataset view and leave it
search-ready in padded-device layout. The caller retains ownership of the underlying
padded storage and must keep it alive while this index uses it.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:173`_

### updateDataset

```java
void updateDataset(PaddedDataset dataset) throws Throwable
```

Update this index with a caller-owned padded device dataset. The dataset must remain alive
while this index uses it.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:179`_

### getGraph

```java
CuVSDeviceMatrix getGraph()
```

Returns the CAGRA graph

**Returns**

a `CuVSDeviceMatrix` encapsulating the native int (uint32_t) array used to represent the cagra graph

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:186`_

### getGraphDegree

```java
long getGraphDegree()
```

Returns the degree of the built CAGRA graph (its number of edges per node), which may be
smaller than the requested `graph_degree` when the dataset is small enough that the
build truncated it.

**Returns**

the built graph degree (`graph().extent(1)`)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:195`_

### size

```java
long size()
```

Returns the number of vectors in this index.

**Returns**

the number of rows of the indexed dataset

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:202`_

### serialize

```java
void serialize(OutputStream outputStream) throws Throwable
```

A method to persist a CAGRA index using an instance of `OutputStream`
for writing index bytes.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes into |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:211`_

### serialize

```java
void serialize(OutputStream outputStream, int bufferLength) throws Throwable
```

A method to persist a CAGRA index using an instance of `OutputStream`
for writing index bytes.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes into |
| `bufferLength` | the length of buffer to use for writing bytes. Default value is 1024 |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:222`_

### serialize

```java
default void serialize(OutputStream outputStream, Path tempFile) throws Throwable
```

A method to persist a CAGRA index using an instance of `OutputStream`
for writing index bytes.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes into |
| `tempFile` | an intermediate `Path` where CAGRA index is written temporarily |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:233`_

### serialize

```java
void serialize(OutputStream outputStream, Path tempFile, int bufferLength) throws Throwable
```

A method to persist a CAGRA index using an instance of `OutputStream`
and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |
| `tempFile` | an intermediate `Path` where CAGRA index is written temporarily |
| `bufferLength` | the length of buffer to use for writing bytes. Default value is 1024 |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:248`_

### serializeToHNSW

```java
void serializeToHNSW(OutputStream outputStream) throws Throwable
```

A method to create and persist HNSW index from CAGRA index using an instance
of `OutputStream` and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:257`_

### serializeToHNSW

```java
void serializeToHNSW(OutputStream outputStream, int bufferLength) throws Throwable
```

A method to create and persist HNSW index from CAGRA index using an instance
of `OutputStream` and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |
| `bufferLength` | the length of buffer to use for writing bytes. Default value is 1024 |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:268`_

### serializeToHNSW

```java
default void serializeToHNSW(OutputStream outputStream, Path tempFile) throws Throwable
```

A method to create and persist HNSW index from CAGRA index using an instance
of `OutputStream` and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |
| `tempFile` | an intermediate `Path` where CAGRA index is written temporarily |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:279`_

### serializeToHNSW

```java
void serializeToHNSW(OutputStream outputStream, Path tempFile, int bufferLength) throws Throwable
```

A method to create and persist HNSW index from CAGRA index using an instance
of `OutputStream` and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |
| `tempFile` | an intermediate `Path` where CAGRA index is written temporarily |
| `bufferLength` | the length of buffer to use for writing bytes. Default value is 1024 |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:294`_

### getCuVSResources

```java
CuVSResources getCuVSResources()
```

Gets an instance of `CuVSResources`

**Returns**

an instance of `CuVSResources`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:301`_

### newBuilder

```java
static Builder newBuilder(CuVSResources cuvsResources)
```

Creates a new Builder with an instance of `CuVSResources`.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsResources` | an instance of `CuVSResources` |

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` | if the provider does not cuvs |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:309`_

### merge

```java
static CagraIndex merge(CagraIndex[] indexes) throws Throwable
```

Merges multiple CAGRA indexes into a single index using default merge parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:321`_

### merge

```java
static CagraIndex merge(CagraIndex[] indexes, CagraIndexParams mergeParams) throws Throwable
```

Merges multiple CAGRA indexes into a single index with the specified merge parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergeParams` | Parameters to control the merge operation, or null to use defaults |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:333`_

### merge

```java
static CagraIndex merge(CagraIndex[] indexes, CagraIndexParams mergeParams, BitSet rowFilter) throws Throwable
```

Merges multiple CAGRA indexes into a single index, keeping only the rows selected by
`rowFilter`.

The merge concatenates the input datasets in the order the indexes are given, so bit
`i` of the filter refers to row `i` of that concatenation: bits `0` to
`indexes[0].size() - 1` address the first index, the bits that follow address the second,
and so on. A set bit keeps the row; a clear bit drops it. The rows that survive keep
their relative order and are packed together, so the merged index has one row per set bit.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergeParams` | Parameters to control the merge operation, or null to use defaults |
| `rowFilter` | The rows to keep, or null to keep all of them. A BitSet shorter than the total row count is valid: the rows beyond its logical length are treated as clear (dropped). A bit set at a position at or beyond the total row count throws `IllegalArgumentException`. |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | if `rowFilter` has a bit set beyond the last row, or if it is non-null but keeps no rows at all |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:357`_

### isPaddedDataset

```java
static boolean isPaddedDataset(CuVSMatrix dataset)
```

Reports whether the rows of `dataset` already sit at the row stride CAGRA requires, which
is the row length in bytes rounded up to a 16 byte boundary.

Use it to pick between the two padded dataset factories: a matrix that is already padded has
to go through `#makePaddedDatasetView(CuVSMatrix)`, because cuVS rejects a request to
copy it into padded storage it already occupies, and one that is not has to go through
`#makePaddedDataset(CuVSMatrix)`.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | the matrix to inspect |

**Returns**

true when the rows are already padded the way CAGRA requires

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:385`_

### from

```java
Builder from(InputStream inputStream)
```

Sets an instance of InputStream typically used when index deserialization is
needed.

**Parameters**

| Name | Description |
| --- | --- |
| `inputStream` | an instance of `InputStream` |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:402`_

### from

```java
Builder from(InputStream inputStream, DeserializeDataset outDataset)
```

Sets an input stream and an empty caller-owned output handle for explicit dataset
deserialization. The concrete output type must match the dataset layout stored in the
serialized index. Keep `outDataset` alive while the built index is in use.

**Parameters**

| Name | Description |
| --- | --- |
| `inputStream` | an instance of `InputStream` |
| `outDataset` | an empty `PaddedDataset` or `StandardDataset` |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:413`_

### from

```java
Builder from(CuVSMatrix graph)
```

Sets a CAGRA graph instance to re-create an index from a
previously built graph.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:419`_

### withDataset

```java
Builder withDataset(float[][] vectors)
```

Sets the dataset vectors for building the `CagraIndex`.

**Parameters**

| Name | Description |
| --- | --- |
| `vectors` | a two-dimensional float array |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:427`_

### withDataset

```java
Builder withDataset(CuVSMatrix dataset)
```

Sets the dataset for building the `CagraIndex`.

The caller retains ownership until a build that uses this dataset returns successfully.
The returned index then owns the dataset, and the caller must leave it open until the index
is closed. If the build fails or uses another configured input source, ownership remains
with the caller.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | a `CuVSMatrix` object containing the vectors |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:440`_

### withBbqDataset

```java
Builder withBbqDataset(BbqQuantizer... quantizers)
```

Builds the graph from one or two encoded BBQ representations. An optional dense dataset
supplied with `#withDataset(CuVSMatrix)` is attached before search; otherwise call
`CagraIndex#updateDataset(PaddedDatasetView)` or
`CagraIndex#updateDataset(PaddedDataset)` before searching.

The index stores views over the quantizer tensors rather than copying them, so they must
stay open for as long as the index is in use. A dense dataset passed to
`#withDataset(CuVSMatrix)` is owned by the index, as it is for a non-BBQ build, and is
closed with it.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:453`_

### withIndexParams

```java
Builder withIndexParams(CagraIndexParams cagraIndexParameters)
```

Registers an instance of configured `CagraIndexParams` with this
Builder.

**Parameters**

| Name | Description |
| --- | --- |
| `cagraIndexParameters` | An instance of CagraIndexParams. |

**Returns**

An instance of this Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:462`_

### build

```java
CagraIndex build() throws Throwable
```

Builds and returns an instance of CagraIndex.

**Returns**

an instance of CagraIndex

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:469`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:26`_
