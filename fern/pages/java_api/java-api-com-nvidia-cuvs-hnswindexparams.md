---
slug: api-reference/java-api-com-nvidia-cuvs-hnswindexparams
---

# HnswIndexParams

_Java package: `com.nvidia.cuvs`_

```java
public class HnswIndexParams
```

Supplemental parameters to build HNSW index.

## Public Members

### NONE

```java
NONE(0), /** * Full hierarchy is built using the CPU */ CPU(1), /** * Full hierarchy is built using the GPU */ GPU(2)
```

Flat hierarchy, search is base-layer only

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:39`_

### CPU

```java
CPU(1), /** * Full hierarchy is built using the GPU */ GPU(2)
```

Full hierarchy is built using the CPU

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:44`_

### GPU

```java
GPU(2)
```

Full hierarchy is built using the GPU

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:49`_

### getHierarchy

```java
public CuvsHnswHierarchy getHierarchy()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:99`_

### getEfConstruction

```java
public int getEfConstruction()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:107`_

### getNumThreads

```java
public int getNumThreads()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:115`_

### getVectorDimension

```java
public int getVectorDimension()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:123`_

### getM

```java
public long getM()
```

Gets the HNSW M parameter: number of bi-directional links per node
used to derive the internal graph build parameters for GPU construction.

**Returns**

the M parameter

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:133`_

### getMetric

```java
public CuvsDistanceType getMetric()
```

Gets the distance metric type.

**Returns**

the metric type

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:142`_

### getAceParams

```java
public HnswAceParams getAceParams()
```

Gets the optional ACE parameters for explicit out-of-core graph construction. When not set, the
graph build algorithm is selected automatically.

**Returns**

the ACE parameters, or null if not set

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:152`_

### Builder

```java
public Builder()
```

Constructs this Builder with an instance of Arena.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:191`_

### withHierarchy

```java
public Builder withHierarchy(CuvsHnswHierarchy hierarchy)
```

Sets the hierarchy for HNSW index when converting from CAGRA index.

NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only
index.

**Parameters**

| Name | Description |
| --- | --- |
| `hierarchy` | the hierarchy for HNSW index when converting from CAGRA index |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:203`_

### withEfConstruction

```java
public Builder withEfConstruction(int efConstruction)
```

Sets the maximum candidate list size used during index construction.

**Parameters**

| Name | Description |
| --- | --- |
| `efConstruction` | the maximum candidate list size used during construction |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:214`_

### withNumThreads

```java
public Builder withNumThreads(int numThreads)
```

Sets the number of host threads to use to construct hierarchy when hierarchy
is `CPU`.

**Parameters**

| Name | Description |
| --- | --- |
| `numThreads` | the number of threads |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:226`_

### withVectorDimension

```java
public Builder withVectorDimension(int vectorDimension)
```

Sets the vector dimension

**Parameters**

| Name | Description |
| --- | --- |
| `vectorDimension` | the vector dimension |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:237`_

### withM

```java
public Builder withM(long m)
```

Sets the HNSW M parameter: number of bi-directional links per node used to derive the internal
graph build parameters for GPU construction.

**Parameters**

| Name | Description |
| --- | --- |
| `m` | the M parameter |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:249`_

### withMetric

```java
public Builder withMetric(CuvsDistanceType metric)
```

Sets the distance metric type.

**Parameters**

| Name | Description |
| --- | --- |
| `metric` | the metric type |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:260`_

### withAceParams

```java
public Builder withAceParams(HnswAceParams aceParams)
```

Sets optional ACE parameters for explicit out-of-core graph construction. When not set, the
graph build algorithm is selected automatically.

**Parameters**

| Name | Description |
| --- | --- |
| `aceParams` | the ACE parameters |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:272`_

### build

```java
public HnswIndexParams build()
```

Builds an instance of `HnswIndexParams`.

**Returns**

an instance of `HnswIndexParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:282`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:12`_
