---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvsreaderresourcesfactory
---

# CuVSReaderResourcesFactory

_Java package: `com.nvidia.cuvs.lucene`_

```java
public interface CuVSReaderResourcesFactory
```

Creates independently owned cuVS resources for long-lived vector readers.

Each successful invocation must return a new, non-null `CuVSResources` instance that is
not shared with another reader or with a query thread. Ownership transfers to the reader, which
closes the resources after closing every native index and matrix allocated from them. If creation
fails before an instance is returned, the factory remains responsible for cleaning up any
partially created resources.

The factory is invoked during construction of each non-merge vector reader. Readers opened for
Lucene merges do not retain GPU indexes and therefore do not invoke it. Implementations must
permit concurrent invocations because Lucene may open readers concurrently.

## Public Members

### create

```java
CuVSResources create() throws Throwable
```

Creates resources whose ownership will transfer to one vector reader.

**Returns**

a new, non-null resources instance

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if the resources cannot be created |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSReaderResourcesFactory.java:32`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSReaderResourcesFactory.java:25`_
