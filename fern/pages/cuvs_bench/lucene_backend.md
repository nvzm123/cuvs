---
slug: user-guide/benchmarking-guide/cu-vs-bench-tool/lucene-backend
---

# Lucene Backend

The optional `lucene` backend runs cuVS Bench against a local Lucene index. It
embeds the JVM through PyLucene. The CPU HNSW control uses stock Lucene through
PyLucene. The cuVS-Lucene thin JAR supplies GPU CAGRA search and GPU-accelerated
HNSW construction with CPU HNSW search. PyLucene is an implementation detail;
the public cuVS Bench backend name is `lucene`.

Selecting this backend is explicit. Commands that do not select it through a
Lucene `--backend-config` file continue to use the `cpp_gbench` backend and its
existing `cuvs_cagra` default. Save this reusable selector as
`lucene-backend.yaml`:

```yaml
backend: lucene
```

## Algorithms

| Algorithm | Codec | Build and search path |
| --- | --- | --- |
| `lucene_cuvs_cagra` | `CuVS2510GPUSearchCodec` | cuVS CAGRA on a supported NVIDIA GPU |
| `lucene_accelerated_hnsw` | `Lucene101AcceleratedHNSWCodec` | cuVS CAGRA build with Lucene CPU HNSW search; CPU HNSW build fallback when GPU support is unavailable |
| `lucene_cpu_hnsw` | `Lucene101` | Lucene CPU HNSW control |

`lucene_cuvs_cagra` is selected when the Lucene backend configuration is used
without an explicit `--algorithms` value. Select `lucene_cpu_hnsw` explicitly
when a CPU control is needed.

```bash
python -m cuvs_bench.run \
    --backend-config lucene-backend.yaml \
    --dataset test-data \
    --dataset-path /absolute/path/to/datasets \
    --algorithms lucene_cuvs_cagra \
    --groups test \
    --batch-size 10 \
    -k 10 \
    --build --search
```

## Optional build controls

An empty `build` configuration keeps the codec's defaults and Lucene's ordinary
flush and merge policies. To configure accelerated HNSW, supply both `m` and
`beam_width` (the HNSW-equivalent construction effort, often called
`ef_construction`). Native cuVS derives its CAGRA build settings through the
default heuristic; these are not search parameters.

Both cuVS-backed algorithms also accept the following optional build controls.
Omitting them retains the corresponding Java codec's defaults.

| Controls | Meaning |
| --- | --- |
| `build_strategy` | `heuristic` or `custom`. Explicit graph geometry, graph-builder selection, NN-descent iterations select `custom` unless a strategy is supplied. |
| `graph_degree`, `intermediate_graph_degree` | Final and intermediate CAGRA graph degrees, in ranges 1–512 and 2–512. Intermediate degree must be at least final degree. Accelerated HNSW additionally requires final degree ≤ `2 * m`. |
| `graph_build_algo` | `nn_descent` (custom-mode default), for constructing the initial CAGRA graph—not selecting the final search algorithm. |
| `nn_descent_n_iters` | NN-descent construction iterations, 1–100; only for the NN-descent graph builder. |
| `writer_threads` | Calling-thread OpenMP count for native graph build/merge work, 1–512, restored after the operation; not Python ingestion threads or a process-wide cap. Defaults remain 1 for accelerated HNSW and 32 for GPU CAGRA. |
| `hnsw_layers` | Accelerated HNSW layer limit, 1–99; the graph may have fewer layers on small datasets. |
| `build_quality` | GPU CAGRA construction-effort setting, not a percentage or a 0–10 score. Default 7; cuVS describes values below 20 as most practical. The binding accepts integers from 0 through 2,147,483,647, but that type limit is not a recommended tuning range or a guarantee of successful construction. Requires heuristic mode. |

Heuristic mode rejects settings it would ignore. The GPU CAGRA dataset heuristic
does consume `graph_degree`, so use `build_strategy: [heuristic]` explicitly to
combine that degree with `build_quality`. Accelerated HNSW keeps the
`SAME_GRAPH_FOOTPRINT` default, with no exposed heuristic selector. It derives
`graph_degree = 2 * m` and `intermediate_graph_degree = 3 * m`, before any
dataset-size clamping during construction; `beam_width` tunes construction
effort. In custom mode, omitted graph
degree is 64 (capped at `2 * m` for HNSW); omitted intermediate degree is the
larger of 128 and the final degree. If neither `m` nor `beam_width` is supplied,
their Java defaults remain 32 and 32.

In the current native dataset heuristic, `build_quality` increases NN-descent's
maximum iterations to `5 + build_quality` for segments with fewer than one
million vectors. Start with the
default 7; larger values request more construction work, not a proportional
recall improvement. Very large values can be impractical or fail native limits.

Choose graph settings for the actual segment sizes. Small segments can cause
native clamping; larger resource requests are not memory reservations or
guarantees of successful allocation.

For reproducible segment layouts, both cuVS-backed algorithms also accept
`premerge_segment_count`, `force_merge_segment_count`, and
`ram_per_thread_hard_limit_mb` together. For example, add this group to an
algorithm YAML passed through the existing `--configuration` option:

```yaml
name: lucene_accelerated_hnsw
groups:
  controlled:
    build:
      m: [16]
      beam_width: [100]
      premerge_segment_count: [1]
      force_merge_segment_count: [0]
      ram_per_thread_hard_limit_mb: [1024]
    search:
      num_candidates: [100]
```

The RAM value above is an example, not a dataset-size budget. Choose a value
appropriate for the segment and leave headroom for JVM, native-host, and device
allocations. The controls have these meanings:

- `premerge_segment_count` partitions the vectors into equal contiguous groups,
  ingested sequentially through separate writer lifecycles. It does not request
  indexing threads. The vector count must be divisible by the partition count.
- Automatic merges and the ordinary RAM-triggered flush are disabled during
  controlled ingestion. The per-thread safety threshold still applies. The
  build fails if the committed segment/vector counts differ from the request.
- `force_merge_segment_count` is `0` to retain the segments or `1` to merge
  accelerated-HNSW segments serially. GPU CAGRA requires `0`.
- Limits from 1 through 2047 MiB use Lucene's public per-thread setter.
  Values outside this range are rejected. The threshold is a Lucene safety
  limit, not a guarantee that arbitrary segment sizes are safe. It reserves
  neither host nor GPU memory.

The codec/query factories belong to the thin cuVS-Lucene JAR. Configured graph
builds require an updated JAR containing `AcceleratedHNSWCodecFactory` or
`GPUSearchCodecFactory`; explicit GPU query controls and CAGRA requests with
`k > 1024` require `GPUKnnFloatVectorQueryFactory`. Missing helpers fail rather
than silently ignoring the request. Default builds and empty search
configurations with `k <= 1024` keep the existing path and do not require these
factories. Java's factory selects the writing codec for its artifact's Lucene
release; the Python runtime still supports only PyLucene 10.2.0.

Configured index names distinguish their build settings; new tuning combinations
use a stable digest to keep filenames bounded. Their schema-3 manifests
record the settings and, when topology is controlled, observed segment counts
and applied RAM-limit mode. Ordinary builds keep their previous names and
schema-2 manifests; those indexes remain reusable. An index built with different
settings must be rebuilt with `--build --force`. Saved codec identities remain
independent of the current writing codec.

## Runtime requirements

The Lucene backend is opt-in because its runtime is not provisioned by the
ordinary cuVS Bench installation. Provisioning the required custom PyLucene
build is currently external to cuVS Bench. Every algorithm requires PyLucene
10.2.0 and JDK 22. Both cuVS-backed algorithms require matching `cuvs-java`
and thin `cuvs-lucene-10.2` JARs. Select the 10.2 artifact for this PyLucene
runtime, not `cuvs-lucene-10.3`. GPU execution additionally requires compatible
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
dimensions; both cuVS-backed algorithms accept at most 4096. The backend
validates the physical segment codec and every persisted vector field before
searching, and fails if CAGRA construction silently produced a brute-force
index. Both HNSW algorithms support an explicit `num_candidates` value greater
than or equal to `k`.

### Search controls

Result count remains the common CLI `-k` option. For HNSW search, `ef_search`
is accepted as an alias for `num_candidates`; it is the candidate count passed
to Lucene's vector query, not an independent hnswlib-style search parameter.
If both names are supplied, they must agree.

GPU CAGRA search optionally accepts `itopk` (internal candidate budget, at least
`k`), `search_width` (entry points per search iteration, not query batch size),
`max_iterations` (`0` = automatic), `algo` (`auto`, `single_cta`, `multi_cta`),
and `thread_block_size` (`0` = automatic, or 64/128/256/512/1024). Single-CTA
search requires an effective `itopk` no larger than 512. Resource requirements
depend on the combination; accepted integer ranges do not guarantee sufficient
device memory. Defaults are `itopk=k`, width 1, automatic iterations/block size
and algorithm. Explicit controls construct `GPUKnnFloatVectorQuery` through the
thin-JAR bridge. Search sweeps reuse one index: these settings do not change its
build identity.

CAGRA requests with `k > 1024` also use this bridge, including with an empty
search configuration. They require the optimized multi-partition path (which
also handles one segment), with a CAGRA index of the same graph degree in every
segment. An incompatible index fails clearly rather than silently using the
per-segment path, whose CAGRA search remains capped at 1024. Use `auto` or
`multi_cta` for larger result counts; `single_cta` retains its 512 limit.

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
CPython 3.12-3.14 with development headers and `venv` support, GNU Make, a C/C++
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

conda/recipes/cuvs-bench/build_pylucene_10_2.sh \
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

The backend discovers the base JAR under `java/cuvs-java/target/` and the
matching thin JAR under `java/cuvs-lucene/lucene-10.2/target/`. It also supports
the corresponding local Maven artifacts, including
`com.nvidia.cuvs.lucene:cuvs-lucene-10.2`. Do not select the native-classifier
`cuvs-java` JAR or the cuVS-Lucene `jar-with-dependencies`.

In each new shell, reactivate the normal cuVS source-build environment before
sourcing `activate.sh`, then restore the two native-library path variables
before starting Python. PyLucene's JVM is process-global, so its classpath and
JVM arguments cannot be changed after `lucene.initVM(...)`.

## Timing contract

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

The above suite also checks configured HNSW builds, retained segment layouts
for both GPU algorithms, and HNSW force merging. Its real warning controls must
run alongside tests that rely on warning absence; a failed control invalidates
that negative evidence until logging or capture is repaired.
