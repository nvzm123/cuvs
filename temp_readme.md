# Temporary validation handoff: cuVS Lucene indexing work

> This is temporary coordination material for cross-machine validation. Remove it before any
> upstream NVIDIA pull request. Do not copy benchmark numbers into a performance claim without
> repeating the matched controls on the target machine.

## Branch and scope

Fetch branch `cuvs-issue-2592-validation-handoff` from the `nvzm123/cuvs` fork. Do not push this
branch, its commits, or its results to the NVIDIA remote.

This branch is stacked on PR 2476 commit `49d69d545`. It is not based directly on `main`, because
PR 2476 was still open when this branch was assembled. Its implementation commits, in order, are:

| Commit | Purpose | Provenance |
|---|---|---|
| `b10b7bdbf` | Bounded parallel CAGRA-adjacency materialization | Safe post-ingest work recovered from PR 2481 |
| `9cce52973` | Bounded parallel level-zero HNSW serialization | Safe post-ingest work recovered from PR 2481 |
| `bae51728f` | Correct accelerated-HNSW merges with deleted vectors | Correctness found while validating the recovered path |
| `a7fdae341` | Harden graph ordinal validation, ownership, and cleanup | Correctness found while validating the recovered path |
| `59ca30473` | Restore direct-device input for GPU-search CAGRA | Targeted follow-up to PR 2476 |

No custom CAGRA, IVF-PQ, `pq_dim`, list, or probe heuristic is included.

PR 2594's implementation is **not** in this branch. In particular, this branch does not contain
the external-FBIN API, mapped/self-contained bulk storage, pre-`addDocument` buffering,
hard-linking, registry/lifecycle machinery, or merge-free multi-segment publication from that PR.
PR 2594 supplied useful measurements and edge-case questions, but no PR-2594-specific code was
carried into this branch. The two parallel graph steps were recovered directly from PR 2481.

## Why these changes exist

PR 2476 fixed large-index GPU and Java-heap failures by making shared matrix construction
host-backed. Accelerated-HNSW merges now stream vectors into native host memory instead of first
building another complete `List<float[]>`. That is still the intended behavior for
`CAGRA_HNSW`, including the ordinary, scalar-quantized, and binary-quantized writers.

After PR 2481 was reverted, profiling showed useful, contract-safe work that occurs only after
Lucene has ingested the documents:

1. Converting native CAGRA adjacency rows into Lucene `NeighborArray` objects.
2. Delta/VInt encoding the level-zero Lucene HNSW graph.

These steps can use the existing writer-thread setting without assuming vector count, ordering,
file ownership, codec choice, or any other state before Lucene calls `addDocument`.

PR 2476 also moved `CuVS2510GPUVectorsWriter` onto the shared host-backed helper. That is useful
for accelerated HNSW, but it is a placement regression for GPU-search CAGRA: the searchable index
ultimately needs a device-resident dataset, so the writer paid for a full host matrix followed by
a full host-to-device copy. The final commit restores a private device builder only in that CAGRA
path. It does not undo PR 2476 for accelerated HNSW.

## Change map

### PR 2476 behavior retained

- Shared float and byte matrix helpers remain host-backed.
- Accelerated-HNSW merge vectors continue to stream directly into a native host matrix.
- Higher HNSW layers can read only the sampled rows they need from that native matrix.
- Float, scalar-quantized, and binary-quantized accelerated-HNSW writers retain this behavior.

### Post-ingest optimization recovered from PR 2481

- For graphs with at least 65,536 nodes, materialize CAGRA adjacency concurrently when
  `writerThreads > 1`.
- Bulk-copy device adjacency to native host memory before concurrent row reads; the device row
  reader is stateful and cannot safely be shared by worker threads.
- Cap that temporary host copy at 4 GiB. Larger device graphs deliberately use the serial path
  instead of creating an unbounded extra allocation.
- Encode level-zero neighbor deltas/VInts concurrently in ordered buffers.
- Limit each serialization wave to a worst-case estimate of 64 MiB, then concatenate buffers in
  node order. This preserves the serial on-disk byte layout.
- Keep the much smaller upper levels serial.
- Pass the existing writer-thread setting through ordinary, scalar-quantized, and
  binary-quantized accelerated-HNSW writers.

### Correctness and lifecycle hardening around the recovered path

- Count vectors actually yielded by the live-vector iterator during deleted-document merges,
  rather than trusting the merged view's raw size.
- Handle zero- and one-live-vector merges without allocating a full native CAGRA graph.
- Transfer dataset ownership only after successful CAGRA construction and close it on failures
  before ownership transfer.
- Filter native negative adjacency sentinels, but fail fast on positive out-of-domain ordinals.
- Validate upper-layer ordinals against the complete graph domain rather than that layer's row
  count.
- Close internally built upper-layer adjacency matrices after copying them.

### Targeted PR 2476 follow-up for GPU search

- Build GPU-search CAGRA input directly in a `CuVSDeviceMatrix` using a writer-private helper.
- Leave the shared host-backed helpers, and therefore `CAGRA_HNSW`, unchanged.
- For aligned dimensions, serialize through a non-owning padded dataset view. For unaligned
  dimensions, create and own a padded device copy.
- Make built-in matrix builders closeable. Closing an unbuilt builder releases its preallocated
  matrix; successful `build()` transfers ownership; later builder use fails.
- Close the input matrix if CAGRA construction fails before ownership transfers. Preserve the
  primary failure and attach cleanup failures as suppressed exceptions.
- Permit brute-force fallback only for runtime/native CAGRA failures. Propagate I/O failures,
  JVM errors, and unexpected checked failures.
- Record a nonzero CAGRA payload length only after serialization succeeds, so metadata does not
  advertise a partial payload.
- Test aligned and unaligned indexes by reopening the GPU reader, asserting that CAGRA rather
  than brute force was persisted, and issuing a search.

The brute-force writer still uses the shared host-backed helper. Do not describe this branch as a
brute-force placement optimization.

## Validation already completed

All values below are observed results. They are not projections. Benchmark times are the
harness-reported end-to-end indexing times. The benchmark harness revision used for the final
runs was `f3672ee4df948d4ad8c98c8fdacd4f9b3efba4bf`.

### Automated tests

On the exact source patch now committed as `59ca30473`, an earlier full GPU-capable run reported:

- `cuvs-java`: 5 unit tests and 109 integration tests; 0 failures, 0 errors, 1 skipped integration
  test.
- `cuvs-lucene`: 343 tests; 0 failures, 0 errors, 30 skipped.

During this handoff, the same source patch was revalidated:

- `cuvs-java`: the same 5 unit and 109 integration tests passed, with 1 integration test skipped.
- Focused Lucene changed-path tests: 18 discovered, 0 failures, 0 errors, 3 GPU-dependent merge
  tests skipped. The 15 executed tests covered fallback/cleanup, bounded graph materialization,
  invalid ordinals, and byte-identical multi-wave serialization.
- Spotless passed for both Java modules, and `git diff --check` passed.

The fresh full Lucene rerun compiled all sources, but it was not a passing validation run: this
machine's GPU driver became unavailable (`nvidia-smi` could not communicate with it), and two GPU
concurrency tests errored because `cudaMallocAsync` was unsupported by the active driver/runtime.
Treat that as an environment failure, not as a green run and not as evidence of a code regression.
The requested second-machine run should repeat the complete GPU suite.

### Recovered CAGRA-HNSW performance

The following cold-source runs used Deep1B, Euclidean distance, default heuristics, graph degree
32, intermediate graph degree 48, one CAGRA-HNSW layer, `writerThreads=16`, `efSearch=1500`, and
`topK=1500`.

| Case | PR-2476 control | Parallel post-ingest candidate | Recall comparison | Notes |
|---|---:|---:|---:|---|
| Deep1B 10M, one segment | 34.955 s | 26.674 s | 97.7669% vs 97.7665% | Single matched runs |
| Deep1B 100M, one segment | 456.642 s | 393.827 s and 397.696 s | 94.9715% vs 94.9587% and 94.9763% | One control, two candidate runs |

The final hardened Deep1B 10M one-segment run measured 26.026 s indexing, 8.60074 ms mean search
latency, and 97.7721% recall. These limited repetitions establish direction, not a confidence
interval.

### Correctness-hardening no-regression checks

These four-segment comparisons isolate the later deletion/ordinal hardening from the performance
commits. They are individual runs, so small differences should be treated as noise.

| Dataset | Before hardening (`9cce52973`) | Final hardening (`a7fdae341`) | Recall before/after | Latency before/after |
|---|---:|---:|---:|---:|
| Jasper 10M | 251.878 s | 252.527 s | 96.3075% / 96.3107% | 38.0921 / 38.3709 ms |
| Deep1B 100M | 252.230 s | 249.722 s | 96.7225% / 96.6210% | 26.5216 / 25.7228 ms |

Deleted-vector validation also covered zero, one, and many surviving vectors; exact live counts;
graph ordinals; sampled edge walks; and search smoke tests. No deletion optimization is claimed.

### GPU-search placement comparison

Matched cold Deep1B 10M runs compared the PR-2476 host-backed GPU-search path with the restored
direct-device path. Each list contains three independent indexing times:

| Input placement | Runs | Median |
|---|---|---:|
| PR-2476 host-backed | 46.424 s, 46.143 s, 46.401 s | 46.401 s |
| Candidate direct-device | 43.422 s, 43.495 s, 43.503 s | 43.495 s |
| Corrected pre-2476 reference | 43.455 s, 43.480 s, 44.113 s | 43.480 s |

The candidate median was 6.26% lower than the matched PR-2476 host-backed median and essentially
matched the corrected pre-2476 reference. Recall remained around 91.8%, and mean latency remained
around 1.02-1.04 ms. An older 34.866 s result is intentionally excluded: that run silently fell
back to brute force and is not a valid CAGRA control.

After lifecycle/failure hardening, a final cold Deep1B 10M GPU-search run measured:

| Algorithm | Indexing time | Mean latency | Recall | Index size | Segments |
|---|---:|---:|---:|---:|---:|
| CAGRA GPU search | 43.163 s | 1.01586 ms | 91.8189% | 8.34478 GiB | 1 |
| CAGRA-HNSW | 26.026 s | 8.60074 ms | 97.7721% | 4.47841 GiB | 1 |

Those rows validate their respective code paths; they are not comparable recall/latency operating
points and do not isolate native graph-build time.

## Validation instructions for the second machine

### 1. Fetch and record provenance

```bash
git fetch https://github.com/nvzm123/cuvs.git \
  refs/heads/cuvs-issue-2592-validation-handoff
git switch --detach FETCH_HEAD
git rev-parse HEAD
git status --short
git submodule status --recursive
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
java -version
mvn -version
nvcc --version
```

The checkout must be clean. Use separate worktrees for controls and candidates, and record the
commit, effective classpath, JAR hashes, native-library hashes, GPU state, and harness revision for
every run.

Useful isolation points are:

- `49d69d545`: PR-2476 branch head before any recovered work.
- `9cce52973`: both parallel post-ingest optimizations, before deletion/ordinal hardening.
- `a7fdae341`: complete accelerated-HNSW salvage and hardening, before GPU-search placement repair.
- The fetched branch tip: all work, including the GPU-search placement repair and this document.

### 2. Build and test

From the cuVS repository root:

```bash
./build.sh clean libcuvs java lucene --run-java-tests
git diff --check
mvn -f java/cuvs-java/pom.xml spotless:check
mvn -f java/cuvs-lucene/pom.xml spotless:check
```

Run the full suite on a host where `nvidia-smi` succeeds before the JVM starts. Do not classify a
GPU test skip, a brute-force fallback, or a driver/runtime error as a passing GPU validation.

### 3. Run matched cold benchmarks

Use Deep1B 10M for the required validation. Deep1B 20M is the preferred additional capacity and
scaling check if it is available. Use the same harness revision, dataset files, filesystem,
hardware, JVM, native libraries, and run order for controls and candidates.

Set the machine-specific repository and data locations in environment variables, then invoke the
harness without embedding another host's locations:

```bash
cd "$BENCH_ROOT"
SOURCE_CACHE_MODE=cold ./run_sweep.sh \
  --data-dir "$DATA_ROOT" \
  --datasets "$DATASETS_FILE" \
  --sweeps "$SWEEPS_FILE" \
  --configs-dir "$CONFIGS_DIR" \
  --results-dir "$RESULTS_ROOT" \
  --run-benchmarks
```

Required fixed settings:

```text
dataset: Deep1B 10M, and Deep1B 20M when available
dimensions and distance: 96, EUCLIDEAN
source cache mode: cold; verify zero resident source bytes before JVM start
CAGRA parameter strategy: HEURISTIC; no custom overrides
graph degree: 32
intermediate graph degree: 48
CAGRA-HNSW layers: 1
writer threads: 16
topK: 1500
efSearch: 1500
compound files: disabled
tiered merge: disabled
normal performance runs: forceMerge=0
```

Minimum matrix:

| Case | Algorithm | Indexing threads / expected segments | Comparison purpose |
|---|---|---:|---|
| A | CAGRA GPU search | 1 / 1 | PR-2476 host input versus direct-device repair |
| B | CAGRA-HNSW | 1 / 1 | Recovered post-ingest work and HNSW no-regression |
| C | CAGRA-HNSW | 4 / 4 | Multi-segment post-ingest behavior |

Set the flush threshold above the document count for the one-segment cases. For the four-segment
case, use four indexing threads and verify exactly four physical segments. Do not compare a
one-segment candidate with a multi-segment control.

Run at least three fresh JVM repetitions per variant, alternating control and candidate order.
Before every run, verify that the GPU is idle, clean the prior output through the harness, evict
only the source FBIN from page cache, and verify zero resident source pages. Preserve individual
runs and report medians rather than the best run.

### 4. Acceptance checks

- No CAGRA build failure or brute-force fallback appears in a GPU-search log.
- GPU-search output contains a nonzero CAGRA payload and a zero-length brute-force payload.
- A freshly opened reader executes a GPU query successfully.
- Physical segment count matches the requested count.
- No JVM, native, OOM, ordinal, or cleanup error occurs.
- Recall and latency remain within ordinary run-to-run variation against the matched control.
- Record `total_indexing_time`, available build/ingest/fsync timers, latency, recall, source-cache
  evidence, segment count, per-file index sizes, effective config, and complete provenance.

For deletion validation, use a separate four-segment run, delete a deterministic sparse subset
from every segment, force one merge, and verify:

```text
numDocs == maxDoc == flat-vector count == graph size == expected live count
```

Then walk sampled graph edges and run a fresh-reader search smoke test. This forced-merge
correctness run is separate from the normal `forceMerge=0` performance runs.

## Planned follow-up cuVS issues

These are known limitations or unisolated opportunities. They are not fixed by this branch and no
performance gain is claimed for them yet.

1. **Direct-device CAGRA capacity:** the GPU-search repair restores the pre-2476 placement and may
   give up some of PR 2476's host-staged peak-VRAM headroom. Validate Deep1B 20M and capture GPU
   high-water memory before upstreaming.
2. **Unaligned device padding:** unaligned CAGRA input temporarily holds ordinary and padded device
   datasets. A direct padded-device builder could avoid the extra allocation.
3. **Brute-force placement:** brute-force GPU input remains host-backed and has not received a
   matched correctness/performance study.
4. **Native build failure lifecycle:** `CagraIndexImpl.build` allocates native state and resets OMP
   settings only after a successful native build. Cleanup and thread-setting restoration on native
   failure need a separate audit.
5. **Builder adoption:** the new builder contract is safe at the new call site, but older shared
   helper call sites are not all try-with-resources. An `addVector` failure before `build()` there
   remains separate hardening work.
6. **CAGRA persistence I/O:** GPU search persists vectors in Lucene's flat-vector file and again
   inside the serialized CAGRA payload. Native CAGRA serialization also uses a temporary file that
   Java copies with a small default stream buffer. The Deep1B 10M outputs were 8.34478 GiB for GPU
   search versus 4.47841 GiB for CAGRA-HNSW, but the time attributable to each I/O step has not
   been isolated.
7. **Transactional fallback:** metadata now reports zero CAGRA length after a failed write, but a
   runtime failure after output begins can leave unreachable bytes before the brute-force payload.
   A rollback-capable or temporary-output design should be evaluated separately.
8. **Graphs above the copy budget:** device adjacency above 4 GiB deliberately falls back to serial
   materialization. Tiled copies or a concurrency-safe native reader could recover parallelism
   without allocating the whole adjacency matrix on host.
9. **Stage-level timing:** existing end-to-end timers do not cleanly separate native CAGRA build,
   padding/copy, graph extraction, materialization, serialization, output, and durability.
10. **PR 2594 bulk/external-FBIN design:** none of its two-artifact integrity, relocation, network
    sharing, or lifecycle contract is present here. Reviving that design requires a separate API
    decision; it must not be implied by these results.

## Interpretation limits

- All timings depend on hardware, storage, cache state, and segment layout. Use matched runs on the
  same idle host.
- Several large results above are single runs. They are useful for direction and regression
  detection, not statistical claims.
- Search was warmed before latency measurement even though source ingestion was cold.
- CAGRA GPU search and CAGRA-HNSW persist different structures and operate at different measured
  recall/latency points.
- This branch changes post-ingest processing and GPU input placement. It does not tune index
  heuristics.
