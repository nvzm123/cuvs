---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-gpusearchparams
---

# GPUSearchParams

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class GPUSearchParams
```

## Public Members

### GRAPH_AND_DATASET

```java
GRAPH_AND_DATASET(0), /** Persist only the graph and reconstruct its dataset from Lucene's flat vectors on open. */ GRAPH_ONLY(1)
```

Persist both the graph and its dataset, matching the original on-disk format.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:19`_

### GRAPH_ONLY

```java
GRAPH_ONLY(1)
```

Persist only the graph and reconstruct its dataset from Lucene's flat vectors on open.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:21`_

### getWriterThreads

```java
public int getWriterThreads()
```

Get the cuVS writer threads parameter

**Returns**

cuVS writer threads parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:155`_

### getIntermediateGraphDegree

```java
public int getIntermediateGraphDegree()
```

Get the intermediate graph degree

**Returns**

the graph degree parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:164`_

### getGraphdegree

```java
public int getGraphdegree()
```

Get the graph degree

**Returns**

the graph degree parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:173`_

### getCagraGraphBuildAlgo

```java
public CagraGraphBuildAlgo getCagraGraphBuildAlgo()
```

Get the CAGRA build algorithm parameter value

**Returns**

the CAGRA build algorithm parameter value

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:182`_

### getIndexType

```java
public IndexType getIndexType()
```

Get the index type parameter

**Returns**

the index type parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:191`_

### getCuVSIvfPqParams

```java
public CuVSIvfPqParams getCuVSIvfPqParams()
```

Get the instance of CuVSIvfPqParams

**Returns**

an instance of CuVSIvfPqParams

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:200`_

### getStrategy

```java
public Strategy getStrategy()
```

Get the chosen strategy:

When HEURISTIC [Default] is chosen, the CAGRA build algorithm and its indexing parameters are automatically chosen based on the size of the data set
When CUSTOM is chosen, the build algorithm and its parameters (either defaults or overridden values with the use of With* methods) is used internally

**Returns**

get the chosen `Strategy`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:213`_

### getCuvsDistanceType

```java
public CuvsDistanceType getCuvsDistanceType()
```

Get the cuvs distance type

**Returns**

the distance type

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:222`_

### getnNDescentNumIterations

```java
public int getnNDescentNumIterations()
```

get the number of Iterations to run if building with NN_DESCENT

**Returns**

the number of iterations for NN_DESCENT

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:231`_

### getBuildQuality

```java
public int getBuildQuality()
```

Get the build quality handed to cuVS' build heuristic. Only consulted under the \{@link
Strategy#HEURISTIC\} strategy.

**Returns**

the build quality

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:241`_

### getCagraPersistenceMode

```java
public CagraPersistenceMode getCagraPersistenceMode()
```

Returns how CAGRA indexes are persisted.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:246`_

### getCagraSerializationBufferSize

```java
public int getCagraSerializationBufferSize()
```

Returns the copy buffer used while moving native serialized bytes into Lucene outputs.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:251`_

### withWriterThreads

```java
public Builder withWriterThreads(int writerThreads)
```

Set the number of cuVS writer threads while building the index
Valid range - Minimum: \{@value MIN_WRITER_THREADS\}, Maximum: \{@value MAX_WRITER_THREADS\}
Default value - \{@value DEFAULT_WRITER_THREADS\}

**Parameters**

| Name | Description |
| --- | --- |
| `writerThreads` | the number of cuVS writer threads |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:310`_

### withIntermediateGraphDegree

```java
public Builder withIntermediateGraphDegree(int intermediateGraphDegree)
```

Set the intermediate graph degree to use while building CAGRA index
Valid range - Minimum: \{@value MIN_INT_GRAPH_DEG\}, Maximum: \{@value MAX_INT_GRAPH_DEG\}
Default value - \{@value DEFAULT_INT_GRAPH_DEGREE\}

**Parameters**

| Name | Description |
| --- | --- |
| `intermediateGraphDegree` | the intermediate graph degree parameter |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:323`_

### withGraphDegree

```java
public Builder withGraphDegree(int graphDegree)
```

Set the graph degree to use while building CAGRA index
Valid range - Minimum: \{@value MIN_GRAPH_DEG\}, Maximum: \{@value MAX_GRAPH_DEG\}
Default value - \{@value DEFAULT_GRAPH_DEGREE\}

**Parameters**

| Name | Description |
| --- | --- |
| `graphDegree` | the graph degree parameter |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:336`_

### withCagraGraphBuildAlgo

```java
public Builder withCagraGraphBuildAlgo(CagraGraphBuildAlgo cagraGraphBuildAlgo)
```

Set the CAGRA build algorithm.
Cannot be null, defaults to NN_DESCENT

**Parameters**

| Name | Description |
| --- | --- |
| `cagraGraphBuildAlgo` | the CAGRA build algorithm to use |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:348`_

### withIndexType

```java
public Builder withIndexType(IndexType indexType)
```

Set the type of index to build - CAGRA, BRUTEFORCE, or both.
Cannot be null, defaults to CAGRA

**Parameters**

| Name | Description |
| --- | --- |
| `indexType` | the type of index to build |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:360`_

### withCuVSIvfPqParams

```java
public Builder withCuVSIvfPqParams(CuVSIvfPqParams cuVSIvfPqParams)
```

Set the instance of `CuVSIvfPqParams`

**Parameters**

| Name | Description |
| --- | --- |
| `cuVSIvfPqParams` |  |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:371`_

### withStrategy

```java
public Builder withStrategy(Strategy strategy)
```

Set the chosen strategy:

When HEURISTIC [Default] is chosen, the CAGRA build algorithm and its indexing parameters are automatically chosen based on the size of the data set
When CUSTOM is chosen, the build algorithm and its parameters (either defaults or overridden values with the use of With* methods) is used internally

Valid options - HEURISTIC, CUSTOM
Default value - HEURISTIC

**Parameters**

| Name | Description |
| --- | --- |
| `strategy` | , the strategy to choose |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:388`_

### withCuvsDistanceType

```java
public Builder withCuvsDistanceType(CuvsDistanceType cuvsDistanceType)
```

Set the CuvsDistanceType

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsDistanceType` | the CuvsDistanceType to set |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:399`_

### withNNDescentNumIterations

```java
public Builder withNNDescentNumIterations(int nnDescentNumIterations)
```

Set the number of Iterations to run if building with NN_DESCENT

Valid range - Minimum: \{@value MIN_NN_DESCENT_NUM_ITERATIONS\}, Maximum: \{@value MAX_NN_DESCENT_NUM_ITERATIONS\}
Default value - \{@value DEFAULT_NN_DESCENT_NUM_ITERATIONS\}

**Parameters**

| Name | Description |
| --- | --- |
| `nnDescentNumIterations` | number of merge workers to set |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:413`_

### withBuildQuality

```java
public Builder withBuildQuality(int buildQuality)
```

Set the build quality cuVS applies when deriving the build algorithm's parameters. Higher
values trade build cost for graph quality.

Only consulted under the `Strategy#HEURISTIC` strategy.

Valid range - Minimum: \{@value MIN_BUILD_QUALITY\}, unbounded above. cuVS documents any value
as valid, with values below 20 being the most practical.
Default value - \{@value DEFAULT_BUILD_QUALITY\}

**Parameters**

| Name | Description |
| --- | --- |
| `buildQuality` | the build quality to set |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:431`_

### withCagraPersistenceMode

```java
public Builder withCagraPersistenceMode(CagraPersistenceMode cagraPersistenceMode)
```

Sets which parts of a CAGRA index are persisted. Graph-only persistence reconstructs the
dataset from Lucene's flat vectors when the segment is opened.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:440`_

### withCagraSerializationBufferSize

```java
public Builder withCagraSerializationBufferSize(int cagraSerializationBufferSize)
```

Sets the copy buffer used while moving a native serialized CAGRA file into Lucene's index
output. The value must be positive and no larger than \{@value
MAX_CAGRA_SERIALIZATION_BUFFER_SIZE\} bytes.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:450`_

### build

```java
public GPUSearchParams build()
```

Creates and returns an instance of `GPUSearchParams`

**Returns**

instance of `GPUSearchParams`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:528`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUSearchParams.java:15`_
