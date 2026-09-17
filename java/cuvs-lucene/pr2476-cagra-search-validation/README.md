# PR 2476: EC2 validation for CAGRA placement and accelerated-HNSW merge fixes

## Scope

This branch is based on PR 2476 commit
[`49d69d5453b2848a4ce762bdbee23bd6471792fa`](https://github.com/NVIDIA/cuvs/commit/49d69d5453b2848a4ce762bdbee23bd6471792fa).
It validates two independent change sets that must be reported separately:

1. **GPU CAGRA placement:** `CuVS2510GPUVectorsWriter` preserves PR 2476's host-staged CAGRA build,
   creates the final device-padded dataset directly, serializes it, and reopens it for native GPU
   CAGRA search.
2. **Accelerated HNSW merge correctness:** the `Lucene101AcceleratedHNSWCodec` path uses CAGRA to
   construct a Lucene HNSW graph while force-merging. The review fixes must size the native host
   matrix from live vectors, preserve the same live-vector ordinal order as Lucene's flat-vector
   merge, handle zero and one live vector before allocating the full input matrix, close the
   single-node temporary allocation, and keep one authoritative trivial-case implementation.

These are different codecs and different search paths. Track 1 ends in a GPU-resident CAGRA index
searched by `GPUKnnFloatVectorQuery`. Track 2 ends in a Lucene HNSW graph and Lucene flat-vector
files; CAGRA is a build-time implementation detail and search is the ordinary Lucene HNSW path.
Do not use a passing test or benchmark from one track as evidence for the other.

Restored semantic comments are part of the review cleanup but have no runtime behavior to validate.
Review them for accuracy in the source diff; do not invent a benchmark result for comment-only
changes.

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

## Track 1 candidate: native GPU CAGRA placement

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

## Track 2 candidate: accelerated-HNSW merge correctness

`Lucene99AcceleratedHNSWVectorsWriter.vectorBasedMerge` obtains a
`MergedVectorValues` view after Lucene has merged the authoritative flat vectors. Its
`FloatVectorValues.size()` is the sum of the source segments' on-disk vector counts; when live-doc
filters remove deleted documents, that value can be larger than the number of rows yielded by its
iterator. Preallocating `CuVSMatrix.hostBuilder(size, ...)` from that raw value leaves unpopulated
rows at the end of the matrix. CAGRA can still build a graph over those rows, so approximate-search
recall can look good while the graph's cardinality or neighbor ordinals exceed the merged flat
vector file.

The review fix must establish these invariants:

- the matrix row count is the exact number of vectors yielded by the live merged iterator;
- rows are appended in the iterator's ordinal order, with no deleted row and no gap;
- the graph's level-0 node count equals the authoritative merged flat-vector count;
- every graph node and neighbor ordinal is in `[0, liveVectorCount)`;
- negative or out-of-domain native adjacency sentinels are discarded before HNSW serialization;
- zero live vectors write the empty-field representation and one live vector writes the supported
  single-node representation;
- zero- and one-vector inputs are dispatched before a full input matrix is allocated, and the
  temporary single-node adjacency matrix is closed after `GPUBuiltHnswGraph` copies it;
- for inputs of two or more vectors, a successfully built `CagraIndex` owns and closes the input
  matrix, while parameter/build failures close it before ownership transfers; and
- list-backed flush/sort input and streamed merge input share one zero/one dispatch helper, while
  both nontrivial paths continue through the single matrix-based CAGRA implementation.

Counting live vectors may require an additional iterator pass when any input segment has deletes.
That is a correctness cost to measure, not a reason to use the raw `size()`. When no segment has a
live-doc mask, retain the fast path if the implementation can prove that `size()` then equals the
iterator cardinality.

## Correctness risks to verify

Do not accept good recall alone as proof. In Track 1, `CuVS2510GPUVectorsWriter` can fall back to
brute force after a CAGRA failure, and brute force can still return excellent results. A valid run
must prove that a CAGRA payload was written, reloaded, and searched. In Track 2, an out-of-range
graph edge may be reached only by particular queries, so a search-only test can pass despite a
structurally invalid graph. Walk the complete serialized graph.

The most important risks are:

1. Host-to-device padding must work for both aligned and unaligned dimensions.
2. The owning padded dataset must remain alive through `updateDataset` and serialization.
3. Serialization and a fresh reader must preserve the graph and dataset.
4. Recall and returned document ordinals must match the accepted baseline.
5. The candidate must reduce or preserve the GPU high-water mark rather than merely move an
   allocation to a different phase.
6. RMM pooling must not make `nvidia-smi` retention look like live-object overlap.
7. Deleted-vector merges must produce identical flat-vector and graph cardinalities and valid
   ordinals, including non-contiguous deletions in every input segment.
8. Zero- and one-live-vector merges must write reopenable output without allocating the full native
   host input matrix, and the temporary single-node adjacency allocation must still be released.

## EC2 environment and reproducibility

Use one dedicated Linux GPU instance for the entire comparison. Choose a GPU whose VRAM is large
enough for the final CAGRA graph, padded dataset, and search workspace; increase the row count until
the placement variants separate rather than switching instance types mid-comparison. Provision
enough host RAM for the host matrix plus the JVM and native build, and enough local or EBS storage
for three builds, indexes, datasets, and raw traces. Do not run another GPU workload concurrently.

Install a cuVS-supported NVIDIA driver/CUDA toolchain, the repository's normal C++ build
dependencies, JDK 22, and Maven 3.9.6 or newer. Use the same AMI, driver, toolkit, compiler, JDK,
Maven, JVM flags, power/clock policy, and physical GPU for every revision. Before building, capture:

```sh
mkdir -p validation-results
git rev-parse HEAD | tee validation-results/commit.txt
git status --short --branch | tee validation-results/git-status.txt
uname -a | tee validation-results/uname.txt
nvidia-smi -q | tee validation-results/nvidia-smi-q.txt
nvcc --version | tee validation-results/nvcc.txt
java -version 2>&1 | tee validation-results/java.txt
mvn -version | tee validation-results/maven.txt
```

Run `nvidia-smi` once more immediately before every measured fork. Record ECC/Xid events, thermal
or power throttling, and other processes using the GPU. Stop and repeat a run contaminated by
another workload; retain the rejected run and the reason instead of silently deleting it.

Clone or copy the repository onto the instance, check out an immutable SHA, initialize required
submodules, and build the native library and Java artifacts from that same checkout. Never combine
Java from one revision with `libcuvs.so` from another. The repository build scripts set the local
library path; if invoking Maven directly in a new shell, make the matching `cpp/build` visible:

```sh
export LD_LIBRARY_PATH="$PWD/cpp/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./build.sh libcuvs java lucene
```

Use a fresh build directory or a separate worktree for each comparison SHA. Before testing, verify
that `nvidia-smi`, `java -version`, and `mvn -version` succeed and that the tests do not skip with
`cuVS not supported`.

## Build and focused tests

From the repository root, run the complete Java build at least once on the final candidate:

```sh
./build.sh libcuvs java lucene --run-java-tests
```

For Track 1, run the Java API residency test and the Lucene aligned/unaligned regression test:

```sh
mvn -f java/cuvs-java/pom.xml clean integration-test \
  -Dit.test=com.nvidia.cuvs.CagraBuildAndSearchIT#testHostMatrixCanCreateDevicePaddedDatasetDirectly
mvn -f java/cuvs-lucene/pom.xml -Dtest=TestCagraIndexAtAlignedDimensions test
mvn -f java/cuvs-lucene/pom.xml \
  -Dtest=TestCuVSVectorsFormat,TestCuVSRandomizedVectorSearch,TestMerge test
```

For Track 2, the review-fix patch must include a deterministic graph-structure regression test.
Use `TestMergedGraphOrdinalBounds` as the test class (or update this command if the final patch uses
a different explicit name); absence of an equivalent test means the fix is not ready to validate:

```sh
mvn -f java/cuvs-lucene/pom.xml -Dtest=TestMergedGraphOrdinalBounds test
mvn -f java/cuvs-lucene/pom.xml \
  -Dtest=TestAcceleratedHNSWDeletedDocuments,TestCuVSAcceleratedHNSWDeletedDocuments,TestLucene99AcceleratedHNSWVectorsFormat,TestCagraToHnswSerializationAndSearch,TestCuVSRandomizedHNSWVectorSearch \
  test
```

The checked-in deterministic suite creates source segments with fixed vectors and seeds, disables
incidental background merging while constructing them, applies scattered deletions to every source
segment, re-enables a merge policy, and `forceMerge(1)`. It then reopens the index and asserts the
following without relying on a random query reaching a bad edge:

1. the merged flat-vector count equals the number of known live vector documents;
2. the HNSW level-0 node count equals that flat-vector count;
3. every node on every level and every neighbor returned by the graph is in
   `[0, liveVectorCount)`; and
4. the zero- and one-live-vector forced-merge boundaries reopen and search with the expected empty
   or single-node representation.

For the EC2 validation harness, add two stronger end-to-end checks beyond the checked-in regression:

- iterate the merged flat vectors and compare the exact expected live document IDs and vector
  values in ordinal order, with no deleted ID, duplicate, or gap; and
- query the reopened index, reject any `EOFException` or unknown/deleted ID, and recompute each
  returned score from the stored source vector to detect an in-range but incorrectly mapped
  ordinal.

The suite includes two non-random edge cases with non-vector sentinel documents kept live so Lucene
must retain and merge the segment while the vector field has exactly zero or one live vector:

- **zero live vectors:** delete every vector-bearing document; assert that the field has no vector
  values, the empty graph metadata is readable, no CAGRA build is attempted, and close succeeds;
- **one live vector:** delete every vector-bearing document except a preselected survivor; assert
  one flat vector, one level-0 node with ordinal `0`, no out-of-range edge, and an exact top-1 hit on
  the survivor.

The new cases specifically exercise the forced-merge entry. Also run the existing single-vector
flush coverage in `TestCagraToHnswSerializationAndSearch`, plus an empty-field flush case, to compare
the list-backed and merge-backed empty/single-node semantics. Repeat zero/one cases in one
bounded-heap JVM and monitor process RSS/native allocation; a monotonic per-iteration increase or a
failure on later iterations is a resource-ownership signal, not benchmark noise. Code review must
confirm that trivial dispatch happens before the full input-matrix allocation and that the
single-node adjacency matrix is closed, because final RSS alone cannot prove object ownership.

The current `CuVSMatrix.Builder` API is not `AutoCloseable` and exposes its backing matrix only from
`build()`. Cleanup if `addVector(...)` itself throws before `build()` is therefore a pre-existing API
hardening problem and is not claimed as part of this patch.

Record randomized-test seeds for every failure. For the required deterministic tests, also run at
least once with a fixed `-Dtests.seed` and retain that seed in the results bundle.

## Track 1 required benchmark comparison

Compare three exact revisions with the same native build configuration, JDK, JVM settings,
allocator, CAGRA parameters, input order, random seed, and machine. Build each revision's Java and
native artifacts from that same revision; if the native sources are identical, record the matching
`libcuvs.so` hashes:

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

## Track 1 phase and memory instrumentation

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

## Track 1: prove that CAGRA—not fallback—was tested

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

For Track 2, also compare every returned hit's score with a score recomputed from the source vector
belonging to that stored document ID. Recall alone does not detect a graph ordinal that is valid but
points to the wrong flat vector.

## Track 2 benchmark comparison

Benchmark at least two immutable revisions on the same EC2 instance:

1. PR 2476 head `49d69d5453b2848a4ce762bdbee23bd6471792fa`, before the review fixes.
2. The final combined candidate SHA containing the live-cardinality, ownership, trivial-case, and
   comment-restoration changes.

Use `Lucene101AcceleratedHNSWCodec`, not `CuVS2510GPUSearchCodec`. Build identical deterministic
source segments, commit each segment explicitly, apply the chosen deletions, then time only the
forced merge in one measurement and end-to-end indexing plus merge in another. Reopen the output in
a fresh reader before recording it as a successful fork.

Include these workloads:

| Purpose | Source segments | Vectors per segment | Deleted vector fraction |
| --- | ---: | ---: | ---: |
| no-delete fast path | 4 | 10K or larger | 0% |
| sparse deletions | 4 | 10K or larger | 1% scattered |
| ordinary deletions | 4 | 10K or larger | 25% scattered |
| heavy deletions | 4 | 10K or larger | 90% scattered |
| cardinality boundary | 2 | at least 64 before deletion | leave exactly 0, 1, and 2 live |
| realistic merge | production-like segment sizes | production-like | measured production distribution |

Use a fixed deletion mask and vector input for both revisions. Scattered deletions are required;
prefix-only or suffix-only deletes are not sufficient to exercise ordinal compaction. Report:

- source vector count, deleted vector count, and expected/live output count;
- force-merge wall time and vectors processed per second, with the zero-delete and delete-present
  cases separated so the cost of the extra count pass is visible;
- maximum Java heap, maximum process RSS, native host-memory high-water, and GPU high-water;
- serialized flat-vector and graph sizes;
- graph level-0 cardinality and the maximum node/neighbor ordinal observed by the full structural
  walk;
- CPU-HNSW reopen/search p50, p95, and p99 latency and throughput at fixed concurrency;
- recall@10/100 against exact live-vector ground truth, plus worst-query recall; and
- any exception, timeout, fallback, OOM, or invalid index. A corrupt baseline is a failed baseline,
  not a performance win.

The existing `AcceleratedHnswIndexingBenchmarks` JMH case is only a smoke test: it uses 1,000
vectors, dimension 128, commits every 100 documents, and does not deterministically force a merge
with deletions. Extend it or use a one-shot harness that exposes segment count and deletion masks.
Use at least three fresh-JVM forks per non-boundary workload, publish every fork, and compare medians
only after all structural assertions pass. The zero/one cases are correctness/resource tests, not
throughput claims.

## Acceptance criteria

The candidate should be accepted only if:

- the environment record names the exact candidate SHA and matching native library, and no required
  test was skipped because cuVS or the GPU was unavailable;
- both aligned and unaligned cases build, serialize, reopen, and search as CAGRA without fallback;
- recall stays within the predeclared tolerance of the accepted baseline;
- the host-staged CAGRA build retains PR 2476's build-phase memory advantage over the device-input
  baseline;
- the candidate does not regress the aligned high-water mark beyond measurement noise;
- the unaligned final-dataset phase no longer exhibits simultaneous ordinary and padded full-size
  device allocations;
- the largest workload that succeeds on PR 2476 still succeeds;
- every accelerated-HNSW deletion case satisfies
  `graph level-0 count == flat vector count == expected live count`, and a complete graph walk finds
  no invalid node or neighbor ordinal;
- the accelerated-HNSW ordinal-to-document/vector checks pass, not merely aggregate recall;
- zero- and one-live-vector merges reopen with the specified empty/single-node representation for
  both list-backed flush and forced-merge entry paths;
- repeated zero/one merge runs and code inspection confirm that no full input matrix is allocated
  on the trivial paths and that the temporary single-node adjacency allocation is released;
- the no-delete accelerated-HNSW path stays within its predeclared force-merge throughput tolerance,
  and the measured deletion-path overhead is consistent with the intentional live-count pass;
- no GPU OOM, Java OOM, native error, invalid ordinal, resource leak, or incomplete index occurs;
- performance stays within the predeclared indexing-time tolerance.

Keep the raw phase log, GPU-memory CSV, `/usr/bin/time -v` output, exact command line, environment
record, parameters, commit SHA, deletion mask/seed, structural-test output, fallback evidence, and
recall report for every run.

## Non-goals

- Binary or scalar-quantized writer changes.
- Redesigning `CuVSMatrix.Builder` to make partially populated builders closeable.
- Native CAGRA merge behavior.
- Search-parameter or recall tuning.
- A production out-of-core CAGRA search implementation.
- Multi-GPU implementation or benchmarking.
- CAGRA-Q integration into cuVS-Lucene.
- Changing the on-disk CAGRA serialization format.
