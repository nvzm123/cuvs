---
slug: api-reference/java-api-com-nvidia-cuvs-bruteforcequery
---

# BruteForceQuery

_Java package: `com.nvidia.cuvs`_

```java
public class BruteForceQuery
```

BruteForceQuery holds the query vectors to be used while invoking search.

Thread Safety: Each BruteForceQuery instance should use its own
CuVSResources object that is not shared with other threads. Sharing CuVSResources
between threads can lead to memory allocation errors or JVM crashes.

## Public Members

### BruteForceQuery

```java
public BruteForceQuery( float[][] queryVectors, LongToIntFunction mapping, int topK, BitSet[] prefilters, int numDocs, CuVSResources resources)
```

Constructs an instance of `BruteForceQuery` using queryVectors,
mapping, and topK.

**Parameters**

| Name | Description |
| --- | --- |
| `queryVectors` | 2D float query vector array |
| `mapping` | a function mapping ordinals (neighbor IDs) to custom user IDs |
| `topK` | the top k results to return |
| `prefilters` | the prefilters data to use while searching the BRUTEFORCE index |
| `numDocs` | Maximum of bits in each prefilter, representing number of documents in this index. Used only when prefilter(s) is/are passed. |
| `resources` | CuVSResources instance to use for this query |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:43`_

### getQueryVectors

```java
public float[][] getQueryVectors()
```

Gets the query vector 2D float array.

**Returns**

2D float array

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:63`_

### getMapping

```java
public LongToIntFunction getMapping()
```

Gets the function mapping ordinals (neighbor IDs) to custom user IDs

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:70`_

### getTopK

```java
public int getTopK()
```

Gets the topK value.

**Returns**

an integer

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:79`_

### getPrefilters

```java
public BitSet[] getPrefilters()
```

Gets the prefilter long array

**Returns**

an array of bitsets

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:88`_

### getNumDocs

```java
public int getNumDocs()
```

Gets the number of documents supposed to be in this index, as used for prefilters

**Returns**

number of documents as an integer

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:97`_

### getResources

```java
public CuVSResources getResources()
```

Gets the CuVSResources instance for this query.

**Returns**

the CuVSResources instance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:106`_

### Builder

```java
public Builder(CuVSResources resources)
```

Constructor that requires CuVSResources.

Important: The provided CuVSResources instance should not be
shared with other threads. Each thread performing searches should create its own
CuVSResources instance to avoid memory allocation conflicts and potential JVM crashes.

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | the CuVSResources instance to use for this query (must not be shared between threads) |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:144`_

### withQueryVectors

```java
public Builder withQueryVectors(float[][] queryVectors)
```

Registers the query vectors to be passed in the search call.

**Parameters**

| Name | Description |
| --- | --- |
| `queryVectors` | 2D float query vector array |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:154`_

### withMapping

```java
public Builder withMapping(LongToIntFunction mapping)
```

Sets the function used to map ordinals (neighbor IDs) to custom user IDs

**Parameters**

| Name | Description |
| --- | --- |
| `mapping` | a function mapping ordinals (neighbor IDs) to custom user IDs |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:165`_

### withTopK

```java
public Builder withTopK(int topK)
```

Registers the topK value.

**Parameters**

| Name | Description |
| --- | --- |
| `topK` | the topK value used to retrieve the topK results |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:176`_

### withPrefilters

```java
public Builder withPrefilters(BitSet[] prefilters, int numDocs)
```

Sets the prefilters data for building the `BruteForceQuery`.

**Parameters**

| Name | Description |
| --- | --- |
| `prefilters` | array of bitsets, as many as queries, each containing as many bits as there are vectors in the index |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:188`_

### build

```java
public BruteForceQuery build()
```

Builds an instance of `BruteForceQuery`

**Returns**

an instance of `BruteForceQuery`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:199`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceQuery.java:21`_
