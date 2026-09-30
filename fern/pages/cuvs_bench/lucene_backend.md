---
slug: user-guide/benchmarking-guide/cu-vs-bench-tool/lucene-backend
---

# Lucene Backend

The optional `lucene` backend runs cuVS Bench against a local Lucene index. It
embeds the JVM through PyLucene. The CPU HNSW control uses stock Lucene through
PyLucene. The cuVS-Lucene thin JAR supplies GPU CAGRA search and GPU-accelerated
HNSW construction with CPU HNSW search. PyLucene is an implementation detail;
the public cuVS Bench backend name is `lucene`.

Selecting this backend is explicit. Commands that do not select it with
`--backend lucene` continue to use the `cpp_gbench` backend and its existing
`cuvs_cagra` default.

## Algorithms

| Algorithm | Codec | Build and search path |
| --- | --- | --- |
| `lucene_cuvs_cagra` | `CuVS2510GPUSearchCodec` | cuVS CAGRA on a supported NVIDIA GPU |
| `lucene_accelerated_hnsw` | `Lucene101AcceleratedHNSWCodec` | cuVS CAGRA build with Lucene CPU HNSW search; CPU HNSW build fallback when GPU support is unavailable |
| `lucene_cpu_hnsw` | `Lucene101` | Lucene CPU HNSW control |

`lucene_cuvs_cagra` is selected when `--backend lucene` is used without an
explicit `--algorithms` value. Select `lucene_cpu_hnsw` explicitly when a CPU
control is needed.

The accelerated-HNSW build accepts Lucene's `m` and `beam_width` parameters.
Both must be integers in the range 1 through 512 when either is specified;
omitting both preserves the codec defaults of 32 and 32. cuVS derives the
CAGRA build parameters with the `SAME_GRAPH_FOOTPRINT` heuristic, so `m: 16`
produces `graph_degree=32` and `intermediate_graph_degree=48`. For example, an
algorithm configuration for `m=16` and `beam_width=80` is:

```yaml
name: lucene_accelerated_hnsw
groups:
  m16_bw80:
    build:
      codec: ["Lucene101AcceleratedHNSWCodec"]
      m: [16]
      beam_width: [80]
    search: {}
```

Pass that file with `--configuration`, select
`--algorithms lucene_accelerated_hnsw --groups m16_bw80`, and use `--build`.
The normalized HNSW parameters are recorded in the index manifest. Result
metadata records those requested parameters together with graph degrees derived
from the selected heuristic; the graph-degree fields are not direct native
observations. The machine-readable
`graph_degree_source=requested_hnsw_same_graph_footprint_derivation` field makes
that provenance explicit, including when the writer uses its CPU fallback.

Both cuVS-backed algorithms can control sequential ingestion partitions and
the final segment topology. `premerge_segment_count`,
`force_merge_segment_count`, and `ram_per_thread_hard_limit_mb` must be
specified together. The partition count must be a positive integer, and
`force_merge_segment_count` must be either `0` (retain the partition segments)
or `1` (produce one final segment). CAGRA builds require
`force_merge_segment_count: 0`; they publish the segments built directly
rather than invoking the codec's vector-merge path.

Each equal, contiguous partition uses a separate writer lifecycle. Automatic
merges are disabled during ingestion, and the document-count flush boundary is
derived from the partition size. Values from 1 through 2047 MiB use Lucene's
public per-thread RAM-limit setter. If that limit causes an earlier flush, the
backend rejects the physical topology mismatch instead of silently reporting
the requested segment shape. Use smaller partitions when a partition cannot
fit under the public safety limit.

Larger direct-built segments are an explicit unsupported mode. A
`ram_per_thread_hard_limit_mb` value of 2048 MiB or greater also requires
`allow_unsupported_lucene_ram_limit: true` and
`force_merge_segment_count: 0`. The thin JAR applies the requested value to
Lucene 10.2's protected, non-public
`LiveIndexWriterConfig.perThreadHardLimitMB` field and reads the value back.
The bridge rejects a missing field, a field with the wrong type, an access or
write failure, or a getter-readback mismatch. The override only changes
Lucene's flush threshold: it does not reserve memory, make construction
out-of-core, guarantee that the host/JVM/GPU can hold the segment, or make a
later force merge safe.

When a final count of one is requested from more than one pre-merge segment, a
serial `forceMerge(1)` exercises the codec's vector-merge path; an index already
containing one segment does not invoke a merge and records
`runtime_force_merge_seconds=0` and `final_merge_policy=null`. This control does
not by itself make that merge out-of-core. For example, a four-partition build
that retains all four segments uses:

```yaml
name: lucene_accelerated_hnsw
groups:
  m16_bw80_four_partitions:
    build:
      codec: ["Lucene101AcceleratedHNSWCodec"]
      m: [16]
      beam_width: [80]
      premerge_segment_count: [4]
      force_merge_segment_count: [0]
      ram_per_thread_hard_limit_mb: [1945]
    search: {}
```

This produces and retains four equal segments. Set
`force_merge_segment_count: [1]` to merge them serially into one segment after
ingestion. Results record the requested and observed pre-merge segment counts,
exact segment vector counts, derived `max_buffered_docs`, applied hard limit,
RAM-limit application mode, merge policies, and final segment count. Both RAM
limit paths verify the applied value at runtime. The manifest stores the
canonical request and runtime topology evidence. Reuse validates the evidence
against the request, dataset row count, and final physical segment count, and
reused build/search results surface that same evidence.

For example, this CAGRA configuration requests one directly built segment with
a 6144 MiB flush threshold:

```yaml
name: lucene_cuvs_cagra
groups:
  one_large_segment:
    build:
      codec: ["CuVS2510GPUSearchCodec"]
      premerge_segment_count: [1]
      force_merge_segment_count: [0]
      ram_per_thread_hard_limit_mb: [6144]
      allow_unsupported_lucene_ram_limit: [true]
    search: {}
```

Use this opt-in only after sizing the process and device for the complete
segment. The same topology keys are available to
`lucene_accelerated_hnsw`.

```bash
python -m cuvs_bench.run \
    --backend lucene \
    --dataset test-data \
    --dataset-path /absolute/path/to/datasets \
    --algorithms lucene_cuvs_cagra \
    --groups test \
    --batch-size 10 \
    -k 10 \
    --build --search
```

## Runtime requirements

The Lucene backend is opt-in because its runtime is not provisioned by the
ordinary cuVS Bench installation. Provisioning the required custom PyLucene
build is currently external to cuVS Bench. Every algorithm requires PyLucene
10.2.0 and JDK 22. Both cuVS-backed algorithms require matching `cuvs-java`
and thin `cuvs-lucene` JARs. GPU execution additionally requires compatible
native cuVS, CUDA, and a supported NVIDIA GPU. `lucene_cuvs_cagra` fails when
that GPU path is unavailable. `lucene_accelerated_hnsw` instead retains the
codec's intentional CPU-writer fallback.
The backend discovers artifacts built by the current checkout or installed in
the local Maven repository. For nonstandard locations, set both
`CUVS_LUCENE_CUVS_JAVA_JAR` and `CUVS_LUCENE_JAR`; use `JAVA_LIBRARY_PATH` for
native libraries. An explicit native-library path must contain an unversioned
`libcuvs_c.so`.

Do not use the cuVS-Lucene JAR assembled with dependencies: PyLucene already
provides Lucene classes, and loading a second copy can make the embedded JVM
classpath inconsistent.

The initial backend accepts nonempty, finite, `float32` Euclidean/L2 vectors
and supports latency-mode sweeps. The CPU HNSW algorithm accepts at most 1024
dimensions; both cuVS-backed algorithms accept at most 4096. CAGRA uses the
codec's fixed defaults and supports `k <= 1024`. Accelerated HNSW accepts the
build parameters described above. The backend validates the
physical segment codec and every persisted vector field before searching, and
fails if CAGRA construction silently produced a brute-force index. Both HNSW
algorithms support an explicit `num_candidates` value greater than or equal to
`k`; their CPU search path is not subject to CAGRA's `k <= 1024` limit.

### Large-build memory and ingestion

The initial build path materializes the complete training-vector file as a
NumPy array. It then indexes one document at a time through PyLucene: every row
is converted to a Python list and Java `float[]`, a Lucene `Document` is
created, and `IndexWriter.addDocument` crosses the JCC boundary. This path is
not a streaming, bulk-FBIN, or out-of-core ingestion path.

Size the Python process and JVM heap for the dataset and codec being tested.
Additional JVM arguments can be supplied through the Lucene backend
configuration, for example:

```yaml
backend: lucene
jvm_args:
  - -Xms16g
  - -Xmx64g
  - -XX:+ExitOnOutOfMemoryError
```

Pass the file with `--backend-config`. These values are illustrative, not
defaults: choose them from the available memory and expected workload. JVM
arguments are immutable after PyLucene initializes the process-global JVM, so
use a new Python process when changing them.

### Build PyLucene 10.2.0 from source

The ordinary cuVS Bench wheel, conda package, and container do not include the
custom PyLucene 10.2.0 runtime. The helper below is available only in a cuVS
source checkout. Automated CI provisioning for the live integration suite is
tracked in [NVIDIA/cuvs#2635](https://github.com/NVIDIA/cuvs/issues/2635).

This procedure has been validated on Linux x86_64 with CPython 3.14 and JDK 22.
A PyLucene wheel is specific to its operating system, architecture, Python ABI,
and JDK toolchain; do not attach or redistribute this local wheel as a general
binary. No real ARM64 build or execution has been performed.

Start in the normal cuVS source-build environment described in the
[shared source-build prerequisites](/installation#build-from-source). Also
install the [Java build prerequisites](/installation/java#build-from-source),
CPython 3.11-3.14 with development headers and `venv` support, GNU Make, a C/C++
compiler, `awk`, `curl`, `flock`, `gzip`, `patch`, `tar`, GNU coreutils, and GNU
findutils. CUDA and a supported NVIDIA GPU are required for the GPU integration
cases.

To run either cuVS-backed algorithm or the full CPU/GPU integration suite, start
at the repository root and, before activating the isolated PyLucene environment,
build matching native cuVS, base `cuvs-java`, and thin `cuvs-lucene` artifacts.
The artifact-free `lucene_cpu_hnsw` control does not need native cuVS or either
JAR, so CPU-only users can skip this command and the native-library setup below,
but should retain the documented cuVS build environment for the editable install.

```bash
export JAVA_HOME=/absolute/path/to/jdk-22
./build.sh libcuvs java lucene
```

Choose a stable absolute PyLucene build location outside `/tmp`. Do not move
the completed directory or selected JDK, and do not remove the base Python
installation: JCC and the virtual environment retain absolute paths. Allow at
least 3 GB of disk space.

```bash
export PYLUCENE_BUILD_ROOT="$HOME/.local/share/cuvs/pylucene-10.2.0"

python/cuvs_bench/tools/pylucene/build_pylucene_10_2.sh \
    --python python3 \
    --build-root "$PYLUCENE_BUILD_ROOT"

source "$PYLUCENE_BUILD_ROOT/activate.sh"
python -m pip install -e ./python/cuvs_bench
python -m pip check
```

The helper verifies checksum-pinned Apache PyLucene 10.0.0 scaffolding, Lucene
10.2.0 sources, and the Gradle distribution. It applies the tracked
compatibility patch, builds JCC 3.15 and a Python-ABI-specific PyLucene wheel in
an isolated virtual environment, runs the upstream PyLucene tests, and performs
a JVM class-loading smoke test. The Python packages requested by the helper are
version-pinned but not hash-locked, and Gradle dependencies remain
network-resolved, so this is not a hermetic or bit-for-bit-reproducible build.

Verify that the activated interpreter uses the expected runtime:

```bash
python - <<'PY'
import os
from pathlib import Path

import lucene

build_root = Path(os.environ["PYLUCENE_BUILD_ROOT"]).resolve()
module_path = Path(lucene.__file__).resolve()
assert lucene.VERSION == "10.2.0", lucene.VERSION
assert module_path.is_relative_to(build_root), module_path
print(f"PyLucene {lucene.VERSION} from {module_path}")
PY
```

The cuVS-backed algorithms do not require the `bench-ann` target. Make the
fresh source-build libraries visible to both Java and the ELF loader:

```bash
CUVS_NATIVE_BUILD="$(cd cpp/build && pwd -P)"
export JAVA_LIBRARY_PATH="$CUVS_NATIVE_BUILD/c:$CUVS_NATIVE_BUILD"
export LD_LIBRARY_PATH="$JAVA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Keep CUDA's runtime directory loader-visible through the normal cuVS build
environment. Some conda toolchains encode their active prefix as `DT_RPATH`,
which takes precedence over `LD_LIBRARY_PATH`. The standard `./build.sh`
command installs the freshly built libraries into that active prefix. If you
override `INSTALL_PREFIX`, verify that the C wrapper resolves matching cuVS
libraries and relocations rather than an older installation:

```bash
ldd -r "$CUVS_NATIVE_BUILD/c/libcuvs_c.so" \
    | grep -E 'libcuvs|librmm|librapids_logger|undefined symbol'
```

The backend discovers the matching JARs under the current checkout; do not
select the native-classifier `cuvs-java` JAR or the cuVS-Lucene
`jar-with-dependencies`.

In each new shell, reactivate the normal cuVS source-build environment before
sourcing `activate.sh`, then restore the two native-library path variables
before starting Python. PyLucene's JVM is process-global, so its classpath and
JVM arguments cannot be changed after `lucene.initVM(...)`.

## Timing contract

`build_time_seconds` and `index_build_call_seconds` cover the in-process Lucene
build call: directory open/close, writer setup, document ingestion, an optional
synchronous force merge, writer commit/close, and the post-build reader check.
They exclude dataset loading, index verification, manifest publication,
installation, and size measurement; use
`backend_build_total_seconds` for that complete backend lifecycle.

In particular, `runtime_document_ingest_seconds` includes NumPy-to-Python and
Python-to-Java conversion, Java object creation, JCC dispatch, and Lucene work
performed by `addDocument`. It is not a measurement of cuvs-lucene or GPU graph
construction alone. `runtime_force_merge_seconds` measures an explicitly
requested synchronous `forceMerge` as a nested sub-timer; it is already included
in the enclosing build-call timings and must not be added to them.
`runtime_writer_commit_close_seconds` includes only work that the selected
codec defers until flush, commit, or close. Absolute build times from this
document-at-a-time path are therefore not directly comparable with a
Java-native or bulk-FBIN benchmark harness.

This initial backend invokes one Lucene query at a time. The common cuVS Bench
`--batch-size` value is retained for configuration and result-file
compatibility, but it does not introduce bulk or concurrent execution. Results
therefore record both `requested_batch_size` and
`effective_search_batch_size=1` and use one latency sample per query.

The headline latency is the complete `client_query` boundary: Java vector and
query preparation, the synchronous search call, and hit materialization. QPS
uses the wall time for the entire serial query corpus, so it is not simply the
reciprocal of mean latency. The raw result also reports narrower preparation,
PyLucene search-dispatch, result-materialization, reader lifecycle, and NumPy
conversion timings. `search_dispatch_kind` distinguishes direct PyLucene
dispatch from the thin-JAR timing bridge. Compare timing results only when that
value matches. Selecting the CPU and GPU algorithms together loads the bridge
for both and provides a like-for-like dispatch-boundary comparison; the
artifact-free CPU control remains useful but is not dispatch-identical.

When the backend initializes from a validated cuvs-java and cuvs-lucene
artifact pair, an additional `java_index_searcher_search` measurement uses
`System.nanoTime()` around exactly `IndexSearcher.search(Query, int)` inside
the JVM. This exact Java measurement is diagnostic and is unavailable to the
artifact-free CPU control. The reported timing boundaries are nested rather
than additive; summing parent and child values double-counts work.

`backend_search_invocation_total_ms` measures one complete non-dry-run search
invocation after argument validation and dry-run handling. A parameter sweep
copies that one shared invocation value into every plan row; do not interpret
or sum it as a per-plan duration.

Before each search-parameter plan, the backend sequentially reads every regular
index file to establish the same host page-cache policy for CPU and GPU paths.
It runs no discarded warmup queries: the first query remains part of the
result, and first-query and subsequent-query values are reported separately
for the client-query and PyLucene-dispatch boundaries, plus the exact JVM
search boundary when Java timing is available. That contrast can reveal
first-query effects, but it does not establish steady state or isolate every
JVM, CUDA, or cuVS initialization cost. Build results similarly separate
dataset preparation, runtime setup, writer lifecycle, index validation,
publication, and total backend time.

## Integration tests

Run the live CPU/GPU suite from the repository root after providing the same
runtime prerequisites:

```bash
python -m pytest -q -s \
    python/cuvs_bench/cuvs_bench/tests/test_lucene_integration.py \
    --run-lucene-e2e
```

Without `--run-lucene-e2e`, ordinary cuVS Bench test runs skip these live
cases. Once selected, every case requires PyLucene and Java. GPU-intended cases
also require the cuVS Java artifacts, native libraries, CUDA, and a supported
GPU; missing prerequisites fail rather than skip. Accelerated-HNSW GPU-intended
cases fail if the codec logs its CPU-writer fallback. A separate GPU-hidden
negative control verifies that this warning remains observable and attributable
to the case that produced it.

The direct PyLucene greater-than-2-GiB cases are independently gated because
they generate a 3 GiB FBIN and build one retained segment by issuing every
document through the Python/JCC boundary. Run them alone, in a fresh process,
on a local filesystem with at least 16 GiB free:

```bash
python -m pytest -q -s -x \
    python/cuvs_bench/cuvs_bench/tests/test_lucene_large_segment_python_integration.py \
    --run-lucene-large-segment-e2e \
    --basetemp=/path/to/local-disk/lucene-large-pytest
```

Run this suite serially, without pytest-xdist. The `-x` option stops after the
first failure while pytest unwinds the fixture and removes its 3 GiB source
file.

The test configures a 12 GiB JVM maximum heap and requires at least 16 GiB of
currently available host memory. It maps the generated vectors read-only to
avoid a second 3 GiB Python allocation, but still exercises the production
per-document PyLucene conversion and `IndexWriter.addDocument` loop. Both the
CAGRA-built HNSW and GPU CAGRA cases require one physical segment, verify the
explicit 6144 MiB override, and reject CPU/brute-force fallback and graph
parameter clamp warnings. The large-suite flag does not select the ordinary
live module, and `--run-lucene-e2e` does not select these capacity cases.
