# PR 2476: CAGRA GPU-search placement experiment

## Scope

This branch is based on PR 2476 commit
[`49d69d5453b2848a4ce762bdbee23bd6471792fa`](https://github.com/NVIDIA/cuvs/commit/49d69d5453b2848a4ce762bdbee23bd6471792fa).
It isolates one question: how should `CuVS2510GPUVectorsWriter` preserve PR 2476's
host-staged CAGRA build while still producing a CAGRA index that can be searched on a GPU?

This branch does **not** include the deleted-document fix, accelerated-HNSW merge fixes, comment
restoration, or other review changes. Mixing those changes into this branch would make the memory
and recall comparison ambiguous.

## Conclusions that motivated the change

### Host-staged build and out-of-core search are different

Current cuVS supports building CAGRA from a host-backed dataset. That can be useful for
`CuVS2510GPUSearchCodec` even though the completed searchable index must be GPU-resident. There is
a useful capacity interval where:

```text
final padded dataset + graph + search state fits on the GPU

but

input dataset + graph + CAGRA build workspace does not fit at the same time
```

Moving the input dataset to host memory during construction can make the second case buildable.
Uploading the final dataset after `CagraIndex.Builder.build()` returns does not inherently erase
that benefit, because the upload no longer has to overlap the logical lifetime of CAGRA's build
workspace. An allocator pool may retain freed blocks, so this must still be verified from
allocation high-water data rather than inferred from a final `nvidia-smi` reading.

This is a GPU-out-of-core or host-staged **build**. It is not out-of-core **search**.

### Current single-GPU CAGRA search is not out of core

The current native search path requires the graph and searchable dataset to be device-resident.
The current cuVS guide says that both must already reside in GPU memory, and the CAGRA search
implementation rejects a disk-backed index with guidance to convert it to HNSW and load that for
search:

- [CAGRA memory requirements](https://github.com/NVIDIA/cuvs/blob/ef29c4cfd53082d31c1fc05ec35251c835120efa/fern/pages/neighbors/cagra.md#L1410-L1423)
- [host build and `update_dataset` contract](https://github.com/NVIDIA/cuvs/blob/ef29c4cfd53082d31c1fc05ec35251c835120efa/cpp/include/cuvs/neighbors/cagra.hpp#L194-L224)
- [disk-backed CAGRA search rejection](https://github.com/NVIDIA/cuvs/blob/ef29c4cfd53082d31c1fc05ec35251c835120efa/cpp/src/neighbors/detail/cagra/cagra_search.cuh#L190-L230)

`MultiPartitionCagraSearch` is not an out-of-core mechanism. Every partition supplied to one call
must already have a device-padded dataset, and the native multi-partition implementation explicitly
rejects disk-backed partitions:

- [C API residency validation](https://github.com/NVIDIA/cuvs/blob/ef29c4cfd53082d31c1fc05ec35251c835120efa/c/src/neighbors/cagra.cpp#L834-L843)
- [native multi-partition requirements](https://github.com/NVIDIA/cuvs/blob/ef29c4cfd53082d31c1fc05ec35251c835120efa/cpp/src/neighbors/detail/cagra/cagra_search.cuh#L343-L385)

The Lucene reader also eagerly deserializes its CAGRA indexes before search, and a query gathers the
already-resident indexes for all relevant leaves. Supporting streamed partitions would therefore
require reader/cache/orchestration work as well as a different residency policy:

- [eager reader loading](https://github.com/NVIDIA/cuvs/blob/ef29c4cfd53082d31c1fc05ec35251c835120efa/java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java#L300-L347)
- [query collection and multi-partition search](https://github.com/NVIDIA/cuvs/blob/ef29c4cfd53082d31c1fc05ec35251c835120efa/java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUKnnFloatVectorQuery.java#L139-L216)

There is a narrow technical nuance: expert native callers can supply CUDA-managed or mapped host
memory that is device-addressable, and Java exposes managed RMM allocation modes. That can be worth
a small oversubscription experiment, but it is not a first-class paged CAGRA search subsystem:
cuVS supplies no CAGRA page cache or traversal prefetcher, disk-backed search is rejected, and
random graph/vector access can cause expensive migration or PCIe traffic. Treat it as an
experiment, not as a supported capacity promise.

### What to do when the final index does not fit

This branch does not pretend that such an index is searchable by single-GPU CAGRA. Applications
should select a supported alternative: CAGRA-built HNSW for CPU search, a suitable compressed or
partitioned index, a multi-GPU sharded design, or an SSD-oriented algorithm such as DiskANN when
that is the actual requirement. The present Java/Lucene multi-partition path still requires all of
its partitions on one GPU, and CAGRA-Q is not currently wired into this Lucene codec, so neither is
an automatic fallback here. The current cuVS Vamana implementation builds and serializes an index
for DiskANN CPU/SSD search; it is not presently a drop-in Java codec replacement:

- [cuVS Vamana and DiskANN scope](https://github.com/NVIDIA/cuvs/blob/ef29c4cfd53082d31c1fc05ec35251c835120efa/fern/pages/neighbors/vamana.md#L1-L13)
- [Vamana serialization and search guidance](https://github.com/NVIDIA/cuvs/blob/ef29c4cfd53082d31c1fc05ec35251c835120efa/fern/pages/neighbors/vamana.md#L190-L216)

A simple application-level prototype could sequentially load a serialized shard, search a retained
query batch, unload it, and merge top-k results. That is most plausible for large offline batches;
full-fanout transfer on every online query would usually have poor latency. Rough planning ranges,
not upstream commitments, are:

- 3–5 engineer-days for a C++ managed/mapped-memory feasibility benchmark across modest VRAM
  oversubscription ratios and both single-query and large-batch workloads;
- 1–3 engineer-weeks for a constrained sequential prototype;
- 6–12 engineer-weeks for a production lazy reader, bounded/ref-counted cache, asynchronous
  double-buffering, filtering, concurrency, cancellation, and observability;
- 4–9+ engineer-months with research risk for transparent native paged CAGRA traversal over graph
  and vector pages.

The last option changes the algorithm's memory-access model and should be compared against CPU
HNSW, IVF-PQ, compression, and multi-GPU sharding before implementation.

## Candidate change

PR 2476 currently follows this sequence in `CuVS2510GPUVectorsWriter`:

```text
host matrix -> host-backed CAGRA build -> ordinary device matrix
            -> device-padded copy when dimensions are unaligned -> attach -> serialize
```

For aligned float dimensions such as 128, the ordinary device matrix already has CAGRA's required
16-byte row alignment and is attached through a view. For unaligned dimensions such as 127, the
padding step temporarily retains both the ordinary device matrix and the final padded device
dataset.

This branch exposes the target memory kind already accepted by the native
`cuvsDatasetMakePadded` API through `CagraIndex.makePaddedDataset(...)`. The writer then requests a
device-padded owning dataset directly from its host matrix:

```text
host matrix -> host-backed CAGRA build -> final device-padded dataset -> attach -> serialize
```

Expected allocation consequences:

- The full input remains off-device during the build, preserving PR 2476's intended build-phase
  benefit.
- The final padded dataset is still uploaded, because current CAGRA GPU search requires it.
- Aligned dimensions retain one full device dataset allocation.
- Unaligned dimensions no longer need an intermediate unpadded device allocation in addition to
  the padded one.
- The final graph plus padded dataset and required search workspace still have to fit on the
  target GPU.

The existing one-argument `makePaddedDataset(dataset)` continues to derive the target from the
source matrix's memory kind; using the public `CuVSHostMatrix` interface also handles host delegates
correctly. The new overload is additive and selects `HOST` or `DEVICE` explicitly.

## Correctness risks to verify

Do not accept good recall alone as proof. `CuVS2510GPUVectorsWriter` can fall back to brute force
after a CAGRA failure, and brute force can still return excellent results. A valid run must prove
that a CAGRA payload was written, reloaded, and searched.

The most important risks are:

1. Host-to-device padding must work for both aligned and unaligned dimensions.
2. The owning padded dataset must remain alive through `updateDataset` and serialization.
3. Serialization and a fresh reader must preserve the graph and dataset.
4. Recall and returned document ordinals must match the accepted baseline.
5. The candidate must reduce or preserve the GPU high-water mark rather than merely move an
   allocation to a different phase.
6. RMM pooling must not make `nvidia-smi` retention look like live-object overlap.

## Build and focused tests

Use JDK 22 and a native cuVS build from the same checkout. From the repository root:

```sh
./build.sh libcuvs java lucene --run-java-tests
```

At minimum, run the new Java API test and the Lucene aligned/unaligned regression test:

```sh
cd java/cuvs-java
mvn clean integration-test \
  -Dit.test=com.nvidia.cuvs.CagraBuildAndSearchIT#testHostMatrixCanCreateDevicePaddedDatasetDirectly

cd ../cuvs-lucene
mvn -Dtest=TestCagraIndexAtAlignedDimensions test
```

Also run these existing GPU-search suites before accepting the change:

```sh
cd java/cuvs-lucene
mvn -Dtest=TestCuVSVectorsFormat,TestCuVSRandomizedVectorSearch,TestMerge test
```

Record randomized-test seeds for every failure.

## Required benchmark comparison

Compare three exact revisions with the same native library version, JDK, JVM settings, allocator,
CAGRA parameters, input order, random seed, and machine:

1. PR 2476 merge base `97d585796b66444c2a80e8ab254e896bed8d5b30` (device input baseline).
2. PR 2476 head `49d69d5453b2848a4ce762bdbee23bd6471792fa` (host build followed by
   `toDevice()` and conditional padding).
3. This branch (host build followed by direct host-to-device-padded creation).

Run each case in a fresh JVM on an otherwise idle GPU. Record the candidate's exact SHA with
`git rev-parse HEAD`; do not benchmark a moving branch name without recording it.

Use both deterministic synthetic data and multiple real datasets already used by the project.
The same vectors and queries must be used for every revision. Include at least:

| Purpose | Dimensions | Dataset size |
| --- | ---: | ---: |
| aligned smoke | 128 | 10K vectors |
| unaligned smoke | 127 | 10K vectors |
| aligned realistic | actual aligned embedding dimension | existing real dataset |
| unaligned realistic | actual unaligned dimension, or a controlled 127/769 case | real or deterministic synthetic dataset |
| capacity boundary | aligned and unaligned | increase rows until variants separate or approach OOM |

Include 768/769 if those dimensions are representative and the GPU has sufficient capacity. Do
not invent a reduced real-data dimension by truncating vectors unless that transformation is
explicitly reported.

For every tuple, perform at least three independent fresh-JVM runs. Report all runs and summarize
the median time and maximum observed memory; do not discard an OOM or fallback run.

## Phase and memory instrumentation

The harness should emit timestamped phase markers around:

1. host-matrix construction;
2. `CagraIndex.Builder.build()`;
3. final device-dataset creation;
4. `updateDataset()`;
5. serialization;
6. resource closure;
7. fresh-reader deserialization and search.

Sample GPU memory at 50 ms or faster, for example:

```sh
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits \
  -lms 50
```

Run the indexing process under `/usr/bin/time -v` to capture maximum host RSS. Record Java heap
settings and native host allocation separately where instrumentation permits. If available, prefer
RMM allocation high-water tracking or an Nsight Systems memory trace in addition to
`nvidia-smi`; a pool can retain logically freed blocks.

Report these peaks separately:

- idle CUDA-context baseline;
- matrix-construction phase;
- CAGRA build phase;
- final dataset creation/padding phase;
- serialization phase;
- whole process;
- fresh-reader load and search.

Also report logical input bytes (`rows * dimensions * 4`) and padded bytes
(`rows * ceil(dimensions / 4) * 16`) for float vectors.

The existing `CagraIndexingBenchmarks` JMH case is only a smoke test for this question: it uses
1,000 vectors, dimension 128, and frequent commits. It is too small and too segmented to establish
the memory-capacity result without modification or a separate one-shot harness.

## Prove that CAGRA—not fallback—was tested

For every `IndexType.CAGRA` run, verify all of the following:

- no `InfoStream` message contains `CAGRA build failed`;
- the field metadata has `cagraIndexLength > 0`;
- the field metadata has `bruteForceIndexLength == 0`;
- the writer and directory are closed, then a fresh reader opens the serialized CAGRA index;
- the fresh reader executes GPU CAGRA search successfully.

The package-private `CuVS2510GPUVectorsReader.FieldEntry` exposes the serialized lengths to a
same-package test harness. This is stronger than inferring the selected index from recall.

## Recall and result validation

For a fixed query set and fixed build/search parameters:

1. Compute exact top-k ground truth with a trusted brute-force implementation.
2. Search the freshly reopened candidate index.
3. Report recall@10 and recall@100 where the dataset size permits.
4. Report mean recall and the worst per-query recall, not only an aggregate.
5. Verify every returned document/ordinal is in range and result counts are correct.
6. Compare exact serialized index type and fallback logs before comparing recall.

Predeclare acceptable timing and recall variation before examining the candidate results. Do not
retrofit a tolerance to make one revision pass.

## Acceptance criteria

The candidate should be accepted only if:

- both aligned and unaligned cases build, serialize, reopen, and search as CAGRA without fallback;
- recall stays within the predeclared tolerance of the accepted baseline;
- the host-staged CAGRA build retains PR 2476's build-phase memory advantage over the device-input
  baseline;
- the candidate does not regress the aligned high-water mark beyond measurement noise;
- the unaligned final-dataset phase no longer exhibits simultaneous ordinary and padded full-size
  device allocations;
- the largest workload that succeeds on PR 2476 still succeeds;
- no GPU OOM, Java OOM, native error, invalid ordinal, resource leak, or incomplete index occurs;
- performance stays within the predeclared indexing-time tolerance.

Keep the raw phase log, GPU-memory CSV, `/usr/bin/time -v` output, exact command line, environment
record, parameters, commit SHA, fallback evidence, and recall report for every run.

## Non-goals

- Deleted-document handling during Lucene merges.
- Accelerated-HNSW graph/flat-vector ordinal correctness.
- Binary or scalar-quantized writer changes.
- Native CAGRA merge behavior.
- Search-parameter or recall tuning.
- A production out-of-core CAGRA search implementation.
- Multi-GPU implementation or benchmarking.
- CAGRA-Q integration into cuVS-Lucene.
- Changing the on-disk CAGRA serialization format.
