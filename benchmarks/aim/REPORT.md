# AIM performance: benchmarks, bottlenecks, speed-up plan and results

## TL;DR

> **Update 4: further large-data work** (see [section 8](#8-further-large-data-optimizations)).
> 1M rows × 8 columns now takes **91s** on 4 cores (accelerated estimator
> with the new `loss_rtol` stopping rule), against 1,709s before section 7.
> The default configuration takes 471s. 10M rows takes 90s.
>
> **Update 3: large datasets** (see [section 7](#7-large-datasets)).
> At 1M+ rows the bottleneck was algorithmic: mirror descent doesn't
> converge on AIM's large, badly conditioned late-round fits, so every fit
> ran the full 1,000 iterations. A native accelerated optimizer
> (`AIM(estimator="accelerated")`) reaches the same fit quality in about 10×
> fewer iterations, and new kernel work makes each iteration on
> multi-million-cell models about 2× cheaper. 1M rows × 8 columns:
> **28.5 min → 3.0 min** on 4 cores (12.4 min with the default estimator),
> with equal or better fit quality. 10M rows × 8 columns takes 2.6 min.
>
> **Update 2: native (Rust) backend** (see [section 6](#6-native-rust-backend)).
> With the optional Rust library, AIM runs **70–500× faster than the
> original** on every configuration benchmarked, single core. The 8-column run
> goes from 411s to about 1s, and 24–32 columns finish in about 10s. JIT
> compilation is gone from the fitting loop entirely.
>
> **Update 1: Phases 1 and 2 are implemented** (see [section 5](#5-results-after-implementing-phases-1-and-2)).
> AIM is now **5–7× faster end to end** on every configuration that
> finished before. Runs with 12–24 columns that used to crash now complete.
> Generating 10M synthetic rows takes 17s instead of 79s with bit-identical
> output. The full test suite passes: 1,559 existing tests plus 74 new
> equivalence tests.

* **AIM spends 91–98% of its runtime compiling JAX/XLA programs, not doing math.**
  Every AIM round builds a model with a different set of cliques, so JAX
  traces and compiles new programs: the mirror-descent loop, the
  message-passing oracle, and one `variable_elimination` program *per
  candidate clique, per round*.
* The math itself is cheap. Re-running the final 8-column fit with an
  already-compiled program takes 2.4s, while the full AIM run took 415s.
* There is also a **hard scaling wall**. Each compiled program holds about 20
  memory mappings that are never freed. Runs with 12 columns crashed after
  about 15 minutes with `LLVM compilation error: Cannot allocate memory`
  because they hit Linux's per-process map limit (`vm.max_map_count` = 65,530).
  RAM was not the issue: 14 GB was still free.
* **The biggest wins are in Python, not in a new language.** Stopping the
  recompilation (stable shapes, NumPy for small eager work, cached or batched
  kernels) should cut runtime by an estimated 10–30×. Rust only pays off later,
  and only for specific pieces (see below).

## 1. Benchmarks

### Setup

* `run_aim_experiment.py` generates a correlated, mixed-type table with
  continuous (skewed, income-like), integer (age-like), categorical (string,
  Zipf-like) and boolean columns. It discretizes the table (quantile bins and
  category codes), then runs AIM. The AIM logic is copied from
  `mechanisms/aim.py` with a timer around each phase, and JIT compile time is
  read from JAX's monitoring hooks.
* Workload: all 2-way marginals. Settings: ε=1, δ=1e-9, `max_model_size=80`,
  `max_iters=1000`, which are the defaults in `mechanisms/aim.py`.
* `run_ladders.py` runs four ladders, each in its own thread. Each thread
  launches its experiments as subprocesses pinned to one CPU core (`taskset`,
  single-threaded XLA), so concurrent ladders don't distort each other. A
  ladder stops after the first experiment that takes longer than 10 minutes.
* Machine: 4 vCPU, 16 GB RAM, Linux, Python 3.13, JAX 0.12, CPU only.

### Results (single core per experiment)

| run | rows | cols | max domain | status | wall (s) | AIM rounds | JIT compile s (% of AIM) | fit s | select s | generate s | avg TV err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rows: 1k | 1,000 | 8 | 16 | ok | 143 | 9 | 137 (96%) | 55 | 86 | 1.7 | 0.267 |
| rows: 10k | 10,000 | 8 | 16 | ok | 375 | 22 | 357 (95%) | 209 | 163 | 2.6 | 0.076 |
| rows: 100k | 100,000 | 8 | 16 | **>10 min** | 1095 | 38 | 1028 (94%) | 837 | 252 | 4.9 | 0.016 |
| cols: 4 | 10,000 | 4 | 16 | ok | 36 | 11 | 32 (91%) | 28 | 6 | 0.7 | 0.067 |
| cols: 6 | 10,000 | 6 | 16 | ok | 115 | 15 | 107 (93%) | 72 | 41 | 1.7 | 0.088 |
| cols: 8 | 10,000 | 8 | 16 | ok | 411 | 23 | 393 (96%) | 227 | 180 | 3.0 | 0.072 |
| cols: 12 | 10,000 | 12 | 16 | **crashed at 938s** (map limit), 17 rounds in | – | 17+ | 899 (98%) | 229 | 684 | – | – |
| bins: 4 | 10,000 | 8 | 4 | ok | 586 | 27 | 571 (97%) | 363 | 217 | 4.0 | 0.014 |
| bins: 16 | 10,000 | 8 | 16 | ok | 435 | 25 | 416 (96%) | 249 | 182 | 3.0 | 0.073 |
| bins: 32 | 10,000 | 8 | 32 | ok | 353 | 20 | 336 (95%) | 192 | 159 | 2.1 | 0.124 |
| bins: 64 | 10,000 | 8 | 64 | ok | 448 | 15 | 430 (96%) | 258 | 185 | 3.3 | 0.215 |
| bins: 128 | 10,000 | 8 | 128 | ok | 318 | 14 | 299 (94%) | 184 | 128 | 3.2 | 0.313 |
| combined | 5,000 | 5 | 8 | ok | 64 | 12 | 59 (93%) | 44 | 18 | 1.5 | 0.046 |
| combined | 20,000 | 8 | 16 | ok | 562 | 29 | 532 (95%) | 362 | 195 | 3.5 | 0.049 |
| combined | 100,000 | 12 | 16 | **crashed at 880s** (map limit), 14 rounds in | – | 14+ | 830 (98%) | 184 | 662 | – | – |

Raw per-round logs are in `results/*.jsonl`. (The `n_in_model` / `n_out_of_model` fields in the ladder logs came from an earlier version of the harness and are wrong. The harness has since been fixed. The timings are unaffected.) Run `python summarize.py` to
regenerate the table.

How each axis behaves:

* **Columns are the dominant axis.** Runtime grows roughly 3× for every 2
  extra columns (36s → 115s → 411s → crash). The number of candidates grows as
  d², the number of rounds grows with d, and the compiled programs get larger
  as the model grows.
* **Rows matter only indirectly.** Counting marginals is cheap (all 28 two-way
  marginals over 10M rows take 3s). More rows give a stronger signal relative
  to the noise, so AIM halves σ less often and runs more rounds (9 → 22 → 38),
  and each round costs mostly compile time.
* **Domain size (bins) barely matters at this scale** and never crossed 10
  minutes. Larger domains made AIM run *fewer* rounds, which offsets the larger
  tables.
* **Using more cores helps little.** The 6-column run took 115s on one core and
  81s on four (1.4×), because XLA compilation is mostly single-threaded.

### Data-dependent steps at large row counts (`gen_scaling.py`, 4 cores)

| rows | discretize | count all 28 2-way marginals | `synthetic_data` |
|---|---|---|---|
| 100k | 0.05s | 0.13s | 0.45s |
| 1M | 0.48s | 0.25s | 4.8s |
| 5M | 2.7s | 1.4s | 31.7s |
| 10M | 7.2s | 3.0s | 79.1s |

Synthetic data generation is the only data-size-dependent step that gets
expensive. In a profile of 3M rows, 13.2 of 17.2s is spent in `argsort`, called
from `np.unique(..., axis=0)` and the stable sort on group ids.

## 2. Where the time goes

Compile time by function (`profile_breakdown.py`, 10k rows, single core):

| jitted function | 6 cols: compiles / seconds | 8 cols: compiles / seconds | what triggers it |
|---|---|---|---|
| `Estimator._multi_step` (mirror-descent scan) | 16 / 44.8s | 24 / 173.4s | Every round adds a measurement, which changes the loss pytree and clique set, so it retraces and recompiles. The loss loops over measurements and message passing loops over junction-tree edges in Python, so the compiled program grows each round. |
| `variable_elimination` | 92 / 35.1s | 315 / 163.1s | `model.project(cl)` for candidates **outside** the model. It is jitted with `clique` as a static argument, and the potentials' structure changes every round, so there is one compile per out-of-model candidate per round. This scales as O(d² × rounds). |
| `message_passing_hugin` | 17 / 10.5s | 33 / 34.1s | `primal_dual_gap` (convergence check) and `_finalize` both call it on a new structure each round. |
| eager `jnp` ops (`_reduce_sum`, `add`, `moveaxis`, ...) | ~200 / ~5s | ~270 / ~8s | Small eager JAX ops on new shapes compile one by one. |

Per-round view, 100k rows × 8 cols:

* Fit compile time grows from 4.9s (round 1, 9 measurements) to about 28s
  (round 30+, 40+ measurements). The program gets bigger as measurements are
  added.
* Selection is all compile time (8–15s per round) until every candidate is
  covered by the model. After that it drops to about 0s, because
  `variable_elimination` is no longer needed.
* At 12 columns, selection alone takes 25–52s per round, all of it
  `variable_elimination` compiles.

Compute cost once compilation is out of the way:

| measurement | value |
|---|---|
| Final 8-column fit, program already compiled | 2.4s, vs 415s for the whole AIM run |
| One cached `message_passing_hugin` call, 8-column benchmark model (2,176 cells) | 0.67 ms (**310 ns/cell**). Dominated by per-op dispatch overhead. |
| Same call on a 3.5 MB model (14 cliques of 32³) | 2.7 ms (**5.8 ns/cell**). Close to memory and compute limits. |
| First call on the 8-column model (compile + run) | 1.6s, which is about 2,400 cached calls |
| Scoring all in-model candidates: JAX eager vs NumPy | 0.9 ms vs 0.3 ms |

Memory-map growth: 326 compiles left 6,632 mappings and 649 compiles left
21,965, so roughly 20–35 mappings per compile. At 65,530, a process can do
about 2–3k compiles before LLVM fails. The 12-column runs reached that in
about 15 rounds.

## 3. Speed-up plan

The work is ordered by payoff per unit of effort. Phases 1–2 stay in Python
and target the 95% of time that is compile overhead.

### Phase 1: stop paying for compilation (estimated 10–30× end to end)

1. **Score candidates without JIT.**
   * In-model candidates: pull marginals to NumPy once per round
     (`np.asarray(factor.values)`) and use `ndarray.sum(axis=...)`. This avoids
     eager JAX op compiles and was 3× faster even with cached programs.
   * Out-of-model candidates: replace the per-clique `jax.jit(static clique)`
     `variable_elimination` with a NumPy implementation (log-space
     `np.einsum` / `logsumexp` over the junction tree). These tables are tiny,
     so NumPy runs them in microseconds with zero compile. Alternatively, run
     one calibrated message pass and compute each out-of-model marginal from
     the calibrated beliefs. This removes the biggest *growing* cost (O(d² ×
     rounds) compiles) and also **fixes the map-limit crash**.
2. **Make the fitting program shape-stable so it compiles once, or a few
   times.**
   * Precision-weight and merge measurements on the same clique, so the loss
     has one term per clique instead of per measurement.
   * Replace the Python loop over measurements in `_l2_loss` with a
     vectorized form: concatenate all flat marginals and use a gather index +
     a single `vdot`. That turns the program-size dependency on the number of
     measurements into a dependency on array *lengths*.
   * Pad array lengths to power-of-two buckets so consecutive rounds hit the
     same compiled executable.
   * Longer term, express the junction tree as padded arrays (fixed max
     clique size, edge list as data) so message passing is a `lax.scan` over
     edges rather than an unrolled Python loop.
3. **Cut the extra oracle compiles.** Compute the `primal_dual_gap`
   max-marginal pass inside the same jitted `_multi_step`, or return it from
   the step, so `message_passing_hugin` isn't compiled separately per round.
4. **Use the existing `Estimator.precompile`** to compile round t+1's fit on a
   background thread while round t scores candidates. The API already returns
   a `Future`, and AIM knows the next clique set once it has selected a
   clique.
5. Cheap hygiene: keep the persistent compilation cache off for sweeps (as
   `mbi` already warns), and call `jax.clear_caches()` periodically as a
   safety net against the map limit until item 1 lands.

### Phase 2: cheap algorithmic and data-path wins

6. **Fix the `synthetic_data` hot spot (measured 79× on that step).** Encode
   the parent columns as one integer with `np.ravel_multi_index`, count groups
   with `np.bincount`, and use `np.argsort(key, kind="stable")` on a
   `uint16`/`uint32` key (NumPy uses radix sort for small ints) instead of
   `np.unique(axis=0)` and argsort on int64. Also vectorize the per-group
   Python loops (the shuffle and the `searchsorted` loop) by adding the group
   offset to the uniform draw and doing one `searchsorted` on concatenated
   CDFs. Expected: 10M rows in about 5–10s instead of 79s.
7. `filter_candidates` builds a junction tree per candidate per round
   (networkx). It is fast today (≤0.02s per round) but grows with d². If it
   becomes visible, cache by `(model cliques, candidate)` or use an
   incremental triangulation check.
8. Warm starts already help. Consider lowering `max_iters` and loosening `tol`
   in middle rounds (only the final fit needs to be tight) and enabling
   float64 for N > 100k, as `mbi` itself warns.

### Phase 3: targeted native code, only where it still matters

After Phases 1–2, re-benchmark. What remains should be dominated by (a)
mirror-descent iterations on small models, where JAX's per-op dispatch costs
about 310 ns/cell, and (b) large models near `max_model_size=80` MB, which are
genuinely compute-bound at about 6 ns/cell.

## 4. Would Rust (or another language) help?

**Short answer: rewriting in Rust is not the first move, and on its own would
not fix today's problem.** The heavy math already runs as compiled C++/LLVM
code through XLA. The slowness is self-inflicted recompilation, which is a
design issue and fixable in Python.

Where a native implementation *would* help:

| part | today | native (Rust/C++) potential | verdict |
|---|---|---|---|
| JIT compilation (95% of time) | 1–30s per program | none needed: no compile step | Phase 1 removes most of this in Python. A Rust core would avoid it by design, but that alone isn't a reason to rewrite. |
| Message passing / mirror descent, **small models** (≲100k cells) | ~300 ns/cell (dispatch-bound) | ~2–10 ns/cell with tight loops over contiguous arrays: **~30–100× per iteration** | Real gain, but only after Phase 1. Good candidate for a small PyO3 extension (logsumexp-message-passing kernel). |
| Message passing, **large models** (≥1M cells, near AIM's 80 MB limit) | ~6 ns/cell (XLA vectorized) | ~3–6 ns/cell with SIMD + rayon | ≤2×. Not worth it on CPU. JAX on a **GPU** is the better lever here (often 10–50× for large tables) and needs no code changes. |
| Candidate scoring / out-of-model marginals | compile-bound | microseconds | NumPy (Phase 1) is enough. |
| `synthetic_data` at 10M+ rows | 79s (argsort-bound) | ~1–3s | NumPy fix gets most of it (5–10s). Rust/Numba gives a further 3–5× if you generate hundreds of millions of rows. |
| Junction-tree construction (networkx) | ms per call | µs | Only if it shows up after the other fixes. |

Options, in recommended order:

1. **Stay in Python + JAX and apply Phases 1–2.** Expected end-to-end speedup
   is 10–30× on the configurations above (for example, the 8-column/10k run
   from about 410s to roughly 15–40s). It also removes the 12+ column crash.
   This keeps GPU support and the existing API.
2. **Add a narrow Rust extension (PyO3 + maturin, `ndarray` + `rayon`)** for
   the small-model message-passing step and `synthetic_data`, behind the
   existing `MarginalOracle` interface. Worth it if you run AIM many times on
   small or medium domains (wide tables with modest bins), where per-iteration
   overhead dominates once compilation is gone.
3. **A full Rust rewrite** would give the best single-machine CPU performance
   for small and medium models. It would lose JAX's autodiff (the loss
   gradients would need to be hand-written), GPU support, and the existing
   estimator ecosystem. That is a large cost for gains you can mostly get
   from options 1–2.

Numba is a middle ground for steps 6–7 if you want native loops without a
Rust toolchain. Julia would also avoid the problem, but would mean a rewrite
with no clear benefit over options 1–2.

## 5. Results after implementing Phases 1 and 2

### What changed

| Plan item | Implemented as | Effect |
|---|---|---|
| 1. Score candidates without JIT | AIM computes in-model marginals with NumPy sums and out-of-model marginals with the new `marginal_oracles.variable_elimination_numpy` | Selection went from 160–680s per run to <1s (8 cols: 180s → 0.3s). This also removes the memory-map crash. |
| 2. Shape-stable fitting program | `merge_measurements` in AIM combines repeated measurements of a clique (precision-weighted, exactly equivalent), so ~30% of rounds reuse the previous round's compiled programs. New opt-in `Estimator(fuse_steps=False)` compiles one optimization step instead of a 50-step `lax.scan`. | Step compile time is 5–6× lower (8 cols: 13s → 2.1s) |
| 3. Cut the extra oracle compiles | `primal_dual_gap` in `estimate` is now a single jitted program instead of an eager gradient (~0.1s of Python dispatch per call, 20 calls per fit) plus a separately compiled oracle | Removes ~25s of overhead per 8-column run |
| 4. Overlap compilation | New opt-in `Estimator(parallel_compile=True)` compiles the step and convergence-check programs on background threads while the estimator initializes. AIM enables it when more than one CPU is available. | About 20% faster per fit on 4 cores, with no extra compiles (tested) |
| 5. Avoid the map-limit crash | AIM clears JAX caches only when the process uses more than half its `vm.max_map_count` (clearing costs about 10s plus recompiles, so doing it unconditionally was slower) | 12–24 column runs complete |
| (new) Cheaper compiles for small models | New opt-in `Estimator(fast_compile=True)` compiles without XLA backend optimizations. AIM uses it while the junction tree has ≤100k cells. | Compile time ~1.5–2.5× lower. Steps run ~2.8× slower, so it is only used where compile dominates. |
| 6. `synthetic_data` hot spot | Integer-key grouping (`_group_rows`) replaces `np.unique(axis=0)` and the int64 argsort. The random-number call sequence is unchanged. | **Bit-identical output**. 1M rows: 4.8s → 0.7s. 10M rows: 79s → 17s (now on 1 core, against 4 before). |
| 7. `filter_candidates` caching | Junction-tree sizes are memoized per (model structure, candidate) | Small, as expected |
| 8. Fewer iterations in middle rounds | Opt-in `AIM(intermediate_iters=...)`. The default is unchanged. | Not used in the benchmarks, so results stay identical |

Two plan items were dropped after measuring:

* **Vectorizing the loss into one gather.** Compiling the loss and its
  gradient takes only 0.4s of the step's compile. The real cost was XLA
  compiling *any* loop body (scan, fori or while) 3–6× slower than straight-line
  code. Packing the loop state into one array and trying XLA flags did not
  help, which is why the fix became `fuse_steps=False`.
* **Padding to power-of-two buckets.** Recompiles come from the clique
  structure, which is static in the jitted program, not from array lengths.
  Padding cannot remove them; only a data-driven junction-tree representation
  could (see next steps).

All new estimator options default to the old behaviour, so existing library
users see no change unless they opt in. The changes that apply to everyone
are the jitted convergence check, which agrees with the eager version to
float32 precision, and the faster `synthetic_data`, which is bit-identical.

### Before / after (same machine and settings, one core per experiment)

| run (rows / cols / bins) | original | optimized | speedup (total) | speedup (per round) |
|---|---|---|---|---|
| bins: 10,000 / 8 / 4 | 586s (27 rounds) | 91s (22 rounds) | 6.4× | 5.2× |
| bins: 10,000 / 8 / 16 | 435s (25 rounds) | 74s (23 rounds) | 5.8× | 5.4× |
| bins: 10,000 / 8 / 32 | 353s (20 rounds) | 57s (20 rounds) | 6.2× | 6.2× |
| bins: 10,000 / 8 / 64 | 448s (15 rounds) | 63s (14 rounds) | 7.1× | 6.6× |
| bins: 10,000 / 8 / 128 | 318s (14 rounds) | 54s (14 rounds) | 5.9× | 5.9× |
| bins: 10,000 / 8 / 256 | – | 37s (14 rounds) | – | – |
| cols: 10,000 / 4 / 16 | 36s (11 rounds) | 10s (12 rounds) | 3.6× | 4.0× |
| cols: 10,000 / 6 / 16 | 115s (15 rounds) | 22s (16 rounds) | 5.2× | 5.6× |
| cols: 10,000 / 8 / 16 | 411s (23 rounds) | 64s (22 rounds) | 6.4× | 6.1× |
| cols: 10,000 / 12 / 16 | crashed at 938s (17 rounds) | 194s (38 rounds) | runs to completion | 10.8× |
| cols: 10,000 / 16 / 16 | – | 283s (37 rounds) | – | – |
| cols: 10,000 / 20 / 16 | – | 585s (56 rounds) | – | – |
| cols: 10,000 / 24 / 16 | – | 774s (59 rounds) | – | – |
| combined: 5,000 / 5 / 8 | 64s (12 rounds) | 19s (10 rounds) | 3.3× | 2.8× |
| combined: 20,000 / 8 / 16 | 562s (29 rounds) | 90s (28 rounds) | 6.2× | 6.0× |
| combined: 100,000 / 12 / 16 | crashed at 880s (14 rounds) | 622s (55 rounds) | runs to completion | 5.6× |
| rows: 1,000 / 8 / 16 | 143s (9 rounds) | 23s (9 rounds) | 6.1× | 6.1× |
| rows: 10,000 / 8 / 16 | 375s (22 rounds) | 73s (23 rounds) | 5.1× | 5.4× |
| rows: 100,000 / 8 / 16 | 1095s (38 rounds) | 164s (34 rounds) | 6.7× | 6.0× |
| rows: 1,000,000 / 8 / 16 | – | timeout at 1500s (33 rounds) | – | – |

* Per-round time is a fairer comparison than total time. AIM's number of
  rounds varies slightly between processes, because `list(set(...))` of
  clique tuples depends on Python's per-process hash seed.
* With the same number of rounds, the optimized and original versions select
  identical cliques (`test/test_aim.py` checks this end to end).
* **New 10-minute boundaries:** 24 columns (774s), 100k rows × 12 columns
  (622s), and 1M rows × 8 columns, which times out at 25 minutes (see below).
  Before, the boundaries were 8–12 columns or 100k rows.
* `results_optimized/gen_scaling.json`: discretize and counting times are
  unchanged. `synthetic_data` takes 0.14s / 0.67s / 5.1s / 16.7s at
  100k / 1M / 5M / 10M rows (before: 0.45s / 4.8s / 31.7s / 79.1s).

### What is slow now

* **Small and medium models are still compile-bound.** On the 8-column run,
  51 of 61s of fitting is compilation. About 2.5s per *new* model structure
  goes to three programs: the step (~1.1s with `fast_compile`), the
  init/finalize oracle (~0.4s) and the convergence check (~0.4s). Removing
  that needs the junction tree expressed as data (padded clique tables, edge
  list as arrays, `lax.scan` over messages), so one compiled program serves
  every model of similar size. That is the main remaining lever, and a larger
  refactor of the oracles.
* **Large runs become compute-bound.** At 1M rows AIM measures more and
  larger cliques. After about round 22, each round takes ~60s, mostly
  mirror-descent iterations on models with hundreds of thousands of cells.
  The original pays this cost too (with fused steps it is ~15% cheaper per
  step, but it compiles far more). The levers here are a GPU,
  `intermediate_iters`, float64 (the library warns that float32 can stall
  convergence for N > 100k, which stops the gap-based early exit from
  triggering), or the Phase 3 native kernel.

### Tests

* `test/_reference_aim.py` and `test/_reference_synthetic_data.py` are
  verbatim copies of the original implementations, used as oracles.
* `test_synthetic_data_equivalence.py` (29 tests): bit-for-bit identical
  synthetic data for the same seed across 4 graph shapes × 3 row counts × 2
  methods; `_group_rows` matches `np.unique` on both code paths.
* `test_estimation_compile_options.py` (21 tests): every combination of
  `fuse_steps` / `fast_compile` / `parallel_compile` matches the default
  estimate. The tolerance is 1e-4 relative in float32 and **1e-9 in
  float64**. Also covers other estimators, warm start and callbacks; the
  jitted gap equals the eager gap; compile counts (one compile per
  structure, reuse across calls, precompile reuse, and no extra compiles from
  `parallel_compile`).
* `test_variable_elimination_numpy.py` (6 tests): matches the jitted
  `variable_elimination` on 4 model shapes and all query orders, matches
  brute force to 1e-10 in float64, and handles `-inf` potentials.
* `test_aim.py` (18 tests): merged measurements leave the total, Lipschitz
  constant and gradient unchanged; estimates agree to 1e-9 (fixed step) or at
  convergence (line search); NumPy scoring matches `model.project`;
  `filter_candidates` matches the original with and without the cache; the
  memory-map-based cache clearing works. Full AIM runs under 7
  configurations select **the same cliques** as the original and fit the
  same model to within 1e-3 of the total.

## 6. Native (Rust) backend

After Phases 1–2, about 80% of AIM's remaining time was still compilation.
JAX compiles a new XLA program (~2.5s) for every new model structure,
because the junction tree is baked into the program, and AIM produces a new
structure most rounds. No amount of Python-side tuning removes that. So,
following section 4, the fitting loop was rewritten in Rust with the model
structure passed in as **data**.

### What was built

`native/` is a dependency-free Rust `cdylib` with a C interface, loaded by
`src/mbi/native.py` with `ctypes`. It is optional: build it with
`python -m mbi.native build`. When it is missing, everything falls back to
the JAX code paths.

| Piece | What it does |
|---|---|
| `NativeMirrorDescent` | Drop-in `MirrorDescent`. The whole `estimate` loop (HUGIN message passing, L2 loss, Armijo line search, primal-dual-gap early stopping) runs in one native call, in float64. It reuses each line-search trial's marginals as the next step's marginals, so it does half the message passing for the same result. It falls back to JAX for constraints, callbacks, non-identity queries and custom oracles. |
| `out_of_model_marginals` | Candidate scoring. Calibrates the junction tree once, then one sweep per source attribute `a` carries `P(a, ·)` across the tree. Every pair is read off from those sweeps, instead of one variable elimination per pair. |
| `hypothetical_model_sizes` | AIM's size filter, computed for all candidates in one native call (bitset graphs, the same greedy elimination, 128-bit sizes with an exact Python fallback on overflow). |
| `expand`, `junction_tree_beliefs`, `synthetic_data(use_native=True)` | NumPy/native replacements for the remaining eager JAX operations in AIM's loop, plus the compile of message passing for the final model. |

Library-level Python improvements found along the way (exact, and used
with or without the native library):

* `junction_tree.greedy_order` caches elimination costs and recomputes only
  the neighbours of the eliminated attribute: **6.8× faster**, with identical
  output on 1,200 random cases.
* `junction_tree.hypothetical_model_size` reads clique sizes directly off the
  elimination (no networkx graph, clique search or spanning tree): **9×
  faster**, identical on 1,500 random cases.
* `filter_candidates` skips the size computation for candidates the model
  already covers, since those are kept regardless of size.

### Kernel engineering, and what each step bought

The first straightforward version was *slower* than XLA on large models (54
against 12 ns/cell). In order of impact, the fixes were:

1. **Block-structured index maps.** Every projection or broadcast between a
   table and a subset of its attributes has trailing axes that are either all
   kept (a contiguous element-wise loop) or all reduced (a contiguous block
   reduction). Iterating by block removes the serial dependency chains of
   per-element scatters. One message-passing call on a 328k-cell model went
   from 14.6 to 3.4 ms.
2. **Small-block fallback.** A trailing boolean attribute makes blocks of 2,
   where per-block overhead dominates. Maps with blocks under 16 cells use
   plain gather/scatter loops instead. 925k-cell model: 42 → 27 ms.
3. **Size-ordered internal layouts.** Junction-tree tables (internal to the
   native code) store attributes from smallest to largest domain, so trailing
   runs are long. 27 → 21 ms.
4. **Convert to probabilities once.** Each junction-tree table is
   exponentiated once, and every loss-clique marginal is a plain sum over it,
   instead of a log-sum-exp projection per marginal. 56 → 40 ms (with 2–3 not
   yet applied).
5. **Cache-blocked fused passes.** Potential expansion, marginal read-out and
   the loss/gradient process each large table once, in 4,096-cell chunks,
   applying every map that touches it.
6. **Vectorized `exp`** (Cody-Waite plus a degree-13 polynomial, using FMA
   when the build targets it).
7. **Deterministic threads** for tables over 1M cells. The partition is
   fixed, so results don't depend on the thread count. This gave ~1.9× on 4
   cores for 4M-cell models.

Final micro-benchmarks (single core unless noted; "cold" includes all
compilation):

| Model | JAX `MirrorDescent` (original, fused) | Native |
|---|---|---|
| AIM's 8-column model (2k junction-tree cells), full fit | 15.3s cold / 0.27s warm | **0.08s cold / 0.03s warm** |
| 4-way chain, 328k cells, full fit (1,000 iters) | 10.0s cold / 7.3s warm | **5.8s** |
| 12 attributes, 925k cells, one message-passing call | 37.8 ms | **19.4 ms** |
| 5-way chain, 4M cells, 200 iters | 31.4s (1 core) / 12.2s (4 cores) | **30.9s** (1 core) / 16.3s (4 cores) |

Native is faster everywhere on one core and never compiles. The only case
where JAX is still ahead is very large models on several cores (12.2s
against 16.3s at 4M cells), where XLA's fused multi-threaded kernels win.

### End-to-end results (same ladders, one core per experiment)

| run (rows / cols / bins) | original | phases 1-2 (JAX) | native (Rust) | speedup native (Rust) vs original |
|---|---|---|---|---|
| bins: 10,000 / 8 / 4 | 586s (27 rds) | 91s (22 rds) | 1.1s (29 rds) | 516× |
| bins: 10,000 / 8 / 16 | 435s (25 rds) | 74s (23 rds) | 1.1s (25 rds) | 395× |
| bins: 10,000 / 8 / 32 | 353s (20 rds) | 57s (20 rds) | 1.1s (21 rds) | 324× |
| bins: 10,000 / 8 / 64 | 448s (15 rds) | 63s (14 rds) | 1.7s (15 rds) | 268× |
| bins: 10,000 / 8 / 128 | 318s (14 rds) | 54s (14 rds) | 1.6s (14 rds) | 195× |
| bins: 10,000 / 8 / 256 | – | 37s (14 rds) | 1.6s (12 rds) | – |
| cols: 10,000 / 4 / 16 | 36s (11 rds) | 9.9s (12 rds) | 0.5s (12 rds) | 67× |
| cols: 10,000 / 6 / 16 | 115s (15 rds) | 22s (16 rds) | 0.7s (14 rds) | 171× |
| cols: 10,000 / 8 / 16 | 411s (23 rds) | 64s (22 rds) | 1.0s (25 rds) | 418× |
| cols: 10,000 / 12 / 16 | crashed at 938s (17 rds) | 194s (38 rds) | 1.6s (36 rds) | completes |
| cols: 10,000 / 16 / 16 | – | 283s (37 rds) | 2.3s (41 rds) | – |
| cols: 10,000 / 20 / 16 | – | 585s (56 rds) | 3.9s (57 rds) | – |
| cols: 10,000 / 24 / 16 | – | 774s (59 rds) | 6.2s (70 rds) | – |
| cols: 10,000 / 32 / 16 | – | – | 9.9s (80 rds) | – |
| combined: 5,000 / 5 / 8 | 64s (12 rds) | 19s (10 rds) | 0.6s (12 rds) | 110× |
| combined: 20,000 / 8 / 16 | 562s (29 rds) | 90s (28 rds) | 1.4s (28 rds) | 415× |
| combined: 100,000 / 12 / 16 | crashed at 880s (14 rds) | 622s (55 rds) | 49s (66 rds) | completes |
| combined: 500,000 / 16 / 32 | – | – | timeout at 1500s (90 rds) | – |
| rows: 1,000 / 8 / 16 | 143s (9 rds) | 23s (9 rds) | 0.8s (10 rds) | 176× |
| rows: 10,000 / 8 / 16 | 375s (22 rds) | 73s (23 rds) | 1.0s (22 rds) | 366× |
| rows: 100,000 / 8 / 16 | 1095s (38 rds) | 164s (34 rds) | 6.9s (31 rds) | 158× |
| rows: 1,000,000 / 8 / 16 | – | timeout at 1500s (33 rds) | timeout at 1500s (43 rds) | – |

Times exclude Python start-up and imports (about 2s), as in the earlier
tables.

* **Every configuration that ran before is 67–516× faster.** 24 columns now
  takes 6s, and 32 columns 10s.
* **Two configurations still exceed 25 minutes on one core:** 1M rows × 8
  columns and 500k rows × 16 columns. These are compute-bound. As the
  privacy budget is spent, AIM's size limit grows toward 80 MB, and late
  rounds fit multi-million-cell models for up to 1,000 iterations each (about
  150s per round for 1M rows after round 35). With 4 cores, see below.
  Further levers are the opt-in `AIM(intermediate_iters=...)`, a GPU via the
  JAX path, or further kernel work.

**With 4 cores** (no pinning, native threads enabled), 1M rows × 8 columns
**completes in 28.5 minutes** (1,709s, 47 rounds, average TV error 0.0027).
That's 99% fitting compute and 0.1s of compilation. No earlier version
completed this configuration: the original JAX code was at round 33 when it
hit the 25-minute cap. Threads give only about 1.5× on these late-round fits,
because many of their tables are below the 1M-cell parallel threshold. A
persistent thread pool (next steps) would let smaller tables and independent
messages run in parallel. Results: `results_native_4core/`.

### Correctness

* **Native routines match their JAX/Python counterparts in float64**
  (`test/test_native.py`, 37 tests). Message passing agrees to 1e-10
  relative. Fixed-step fits agree to 1e-9. Line-search fits agree to 1e-5:
  the optimization is the same, but rounding can flip a near-tied
  accept/reject or move the gap check across `tol`. Pairwise marginals from
  the sweep match variable elimination to 1e-9. Model sizes match exactly.
* **Synthetic data:** `use_native=True` is opt-in, because float64 marginals
  can round a few counts differently from the default float32 path. In tests
  the outputs were identical or within 1e-3 total variation.
* **The rewritten junction-tree helpers** match verbatim copies of the
  originals on thousands of random graphs
  (`test/test_junction_tree_equivalence.py`).
* **AIM end to end** (`test/test_aim.py`). AIM is chaotic with respect to
  rounding: a tiny score change can flip one draw of the exponential
  mechanism, and everything after diverges. Measuring this directly, the
  *original* AIM run on a mathematically identical problem (same noisy
  measurements, passed to the estimator in reverse order) shifts per-round
  scores by up to 1e-2 of the model total. The end-to-end tests therefore
  *replay* the original: the optimized AIM is forced to make the original's
  choices while still consuming the random stream identically. Per-round
  candidates and scores are checked against that measured band, and the
  final model is checked too. One more effect: occasionally AIM's σ-halving
  statistic lands within rounding of its threshold (one case measured
  291.3/291.9 against a threshold of 291.6), after which runs legitimately
  diverge. The test compares up to that point. These tests pass on 44
  different hash seeds, including 12 on the final kernel build.
* **Precision note:** with the native backend, AIM's fits run in float64,
  while the original uses JAX's default float32. In float32, the reference
  fit stops early on a float32 convergence check and is accurate to roughly
  one count per thousand records, so runs with and without the native
  library can make different random choices from the same seed. Both are
  correct samples of the mechanism; the float64 one is more precise.
* Full suite: 1,677 tests pass (1,559 original plus 118 new).

### Answer to section 4, revisited

The earlier recommendation was "fix compilation in Python first; Rust only
for targeted kernels." Phases 1–2 got 5–7×. Moving the fitting loop to
native code with the structure as data got a further 30–100×, because the
remaining cost was structural to JAX's design for this workload (many
differently-shaped small programs, each run once). For workloads that fit
one large model many times, JAX/XLA, especially on GPU, remains the better
engine, and the JAX paths are unchanged and still the default for
`MirrorDescent`.

### Next steps

* Ship prebuilt wheels (maturin or cibuildwheel), so users don't need a Rust
  toolchain.
* Large-model throughput on multi-core machines: a persistent thread pool
  (to parallelize smaller tables and independent messages), and fewer passes
  per message.
* Port more estimators (e.g. `UniversalAcceleratedMethod`) and Shafer-Shenoy
  (for constraints) to the native path.


## 7. Large datasets

### Diagnosis

At 1M rows the native backend spent 99% of its time in fitting (no
compilation, almost no Python). Reconstructing the late-round fits of the
1M-row run offline showed why:

* **The models are huge.** AIM's size limit grows as the privacy budget is
  spent, and with strong signal it keeps adding cliques. By round 36 the
  junction tree has 4.4M cells (one clique of 4.2M), and by round 46 it has
  8.4M, close to the 80 MB cap.
* **Mirror descent doesn't converge on them.** With counts in the hundreds
  of thousands and noise σ ≈ 5, the problem is badly conditioned. Every
  late fit ran all 1,000 iterations and ended with a normalized gap of about
  0.67, against the 1e-3 tolerance. The loss was still falling steeply
  (cold start: 4.4e5 at 1,000 iterations, 2.7e4 at 3,000). The step size
  isn't the limit: Armijo grows it 5–7× and then plateaus.

So most of the cost was iterations, not per-iteration speed.

### 1. A faster optimizer

All of mbi's estimators, compared on a reconstructed 1M-row fit (57k cells,
float64; loss after N iterations, lower is better):

| Estimator | cold 100 / 300 / 1000 | warm 100 / 300 / 1000 |
|---|---|---|
| MirrorDescent (AIM's default) | 1.2e8 / 7.6e6 / 4.4e5 | 6.8e5 / 2.0e5 / 6.1e4 |
| **UniversalAcceleratedMethod** | 6.6e5 / 1.4e4 / **590** | 6.2e4 / 2.2e3 / **295** |
| LBFGS | 9.9e7 / 2.3e7 / 3.9e6 | 5.4e5 / 3.4e5 / 2.0e5 |
| DualAveraging | 1.8e7 / 4.7e5 / 8.7e3 | 1.8e7 / 4.7e5 / 8.3e3 |
| InteriorGradient | 1.7e7 / 4.5e5 / 8.2e3 | 2.6e5 / 4.1e4 / 1.1e3 |

The accelerated method (Nesterov's universal method) reaches in 100 warm
iterations the loss mirror descent reaches in 1,000. It exists in mbi, but
only in JAX, where it compiles per structure and finishes with an L-BFGS
projection of up to 25,000 iterations. So it was ported:

* **`mbi.native.NativeAcceleratedMethod`** runs the method's loop
  (including the backtracking line search) in Rust. Its averaged iterate
  matches JAX's to 1e-6 in float64 (`test_native.py`), and it runs 10–100×
  faster than the JAX version.
* **The final model differs from JAX's.** The averaged iterate `x` isn't
  itself a graphical model, so instead of an expensive L-BFGS projection the
  native version returns the model given by the dual potentials (marginals =
  the mirror iterate `z`). A short mirror-descent projection onto `x` barely
  helped (same conditioning problem). The `z` model is still far better than
  mirror descent's at equal iterations: 1,465 against 61,330 after 1,000
  iterations on the problem above.
* **AIM gets `estimator="accelerated"`** (opt-in; the default stays mirror
  descent, as in the AIM paper).

AIM end to end with `estimator="accelerated", max_iters=200` against the
default (mirror descent, 1,000 iterations), native backend:

| Data | Mirror descent, 1000 iters | Accelerated, 200 iters |
|---|---|---|
| 100k × 8, 3 seeds | 3.3–4.6s, model TV 0.0142 / 0.0142 / 0.0142 | **1.3–1.7s**, TV 0.0141 / 0.0146 / 0.0136 |
| 100k × 12, 3 seeds | 30–51s, TV 0.0166 / 0.0163 / 0.0172 | **11–13s**, TV 0.0169 / 0.0170 / 0.0170 |
| **1M × 8, 4 cores** | **745s** (was 1,709s before this section's kernel work), TV 0.0019 | **179s** (3.0 min), TV 0.0018 |
| **10M × 8, 4 cores** | – | **156s** (2.6 min), TV 0.0003 |

TV is the mean total-variation error of the model's 2-way marginals against
the data (lower is better). Quality is the same within run-to-run noise;
with only 200 iterations, mirror descent is visibly worse (TV 0.0147 at
100k × 8). Recommendation: use `estimator="accelerated"` with `max_iters`
around 200 for datasets of 100k rows or more.

### 2. Cheaper iterations on huge models

A new opt-in profiler (`MBI_NATIVE_PROFILE=1`) breaks fits down by phase. On
the 4.4M-cell fit, half the time went into moving data between the 26
*small* loss cliques (3,716 cells in total) and the huge junction-tree
tables. Fixes:

* **Intermediate-table hierarchy.** Python plans, per junction-tree table, a
  greedy hierarchy of intermediate tables: drop an attribute whenever enough
  small cliques fit in the smaller table to pay for the extra pass. Marginals
  are read from the smallest table containing them, and potentials are
  summed into intermediates before being broadcast into the big table. For
  the 4.2M-cell table this is two 262k-cell intermediates.
* **One `exp` pass for normalization.** Probabilities are computed as
  `exp(b − max)` while summing (in parallel), and `total/sum` is applied to
  the small outputs. The old serial log-sum-exp pass is gone, along with a
  second `exp` pass.
* **Disjoint parallel scatters.** When a reduction keeps the big table's
  leading attribute(s), splitting along them gives each part its own output
  range. Parts write directly into their own sub-slices, with no per-part
  partial buffers. This applies to marginal read-out and message reduction.

| Phase (100 accelerated iterations, 4.4M cells) | before, 1 / 4 threads | after, 1 / 4 threads |
|---|---|---|
| Expand potentials | 4.1s / 2.1s | **1.0s / 0.8s** |
| Read out loss-clique marginals | 4.2s / 2.2s | **1.0s / 0.8s** |
| Normalization (log-sum-exp + `exp`) | 3.8s / 3.2s | **2.6s / 1.4s** |
| Message reduction | 3.1s / 1.9s | 3.0s / 1.6s |
| **Wall time** | **20.3s / 10.3s** | **8.7s / 5.7s** |

### 3. Data-size-dependent steps

* **Counting.** `native.marginal_counts` builds many contingency tables in
  one multi-threaded pass over compact columns. With 88 cliques at 1M rows
  it takes 0.14s against 0.93s, and the counts are exact. (`Dataset.project`
  returns float32, so it can't represent counts above 2^24 exactly.) AIM also
  reuses its initial counts for each round's measurement instead of
  re-scanning the data.
* **Synthetic data.** With `synthetic_data(use_native=True)`, each column is
  sampled natively: rows are grouped with a counting sort, ranks are
  shuffled per group, and values come from a binary search in the
  conditional CDF. Random streams are derived from a seed drawn from
  `np.random`, per group, so results are reproducible and independent of the
  thread count. **10M rows: 1.7s against 19.7s**, with the same fidelity to
  the model.

### What's next for large data

See section 8 for what was done next.

## 8. Further large-data optimizations

### What was done

| Change | Effect |
|---|---|
| **Opt-in `loss_rtol` stopping rule** (`Estimator.loss_rtol`, JAX and native; `AIM(loss_rtol=...)`): stop when a 50-step block improves the loss by at most this fraction. | Fits stop when the loss plateaus, instead of waiting for a gap criterion that at large scale stays around 1e-2. 100k × 12: mirror descent 34.9s → 15.7s at `1e-3`; accelerated 9.5s (fixed 200 iterations) → 6.7s at `1e-2`, with the same TV error. Native and JAX stop at the same block (tested). |
| **Messages from probabilities.** A calibrated clique's message is `max + ln S[o] − rev[o]`, where `S` is a plain sum of its probability table (computed anyway for the marginals), so there's no `exp` per cell. The HUGIN schedule is rooted at the largest clique so it sends only calibrated messages. If a separator sum underflows, that message falls back to the exact log-space path (tested with a 2,000-log-unit range). | Message time on the 4.4M-cell fit: 3.0s → 0.7s (1 thread) |
| **Persistent thread pool.** Workers park on a condition variable instead of being spawned per parallel section. Concurrent or nested use falls back to serial. A race found under stress (a stale worker taking a newer job's task) was fixed with job generations. | 4.4M-cell fit, 4 threads: 5.7s → **2.8s**. Results are bit-identical with 1 and 4 threads (tested in subprocesses). Lowering the parallel threshold below 1M cells made mid-size runs *slower*, so it stays. |
| **Mixed disjoint/partial read-out.** Read-outs where only some maps keep the big table's leading attribute still split along it: those maps write disjoint sub-slices, and the rest use small partials. | 8.4M-cell fit read-out: 2.1s → 0.7s (4 threads) |

### End-to-end (4 cores, native backend, same fit quality throughout)

| Configuration | Before section 7 | Section 7 | Now |
|---|---|---|---|
| 1M × 8, default (mirror descent, 1,000 iters) | 1,709s | 745s | **471s** |
| 1M × 8, mirror descent + `loss_rtol=1e-3` | – | – | **304s** |
| 1M × 8, accelerated + `loss_rtol=1e-2` | – | 179s (fixed 200 iters) | **91s** |
| 10M × 8, accelerated + `loss_rtol=1e-2` | – | 156s (fixed 200 iters) | **90s** |

Model TV error 0.0018–0.0019 at 1M rows and 0.0003 at 10M, synthetic TV
0.0027 / 0.0004, in every configuration. Raw results:
`results_large/final_4core.txt`.

**Recommended settings for large datasets:**
`AIM(..., estimator="accelerated", loss_rtol=1e-2)`. To stay with the
paper's optimizer, `loss_rtol=1e-3` alone saves about a third.

### Where time goes now, and what was deliberately not done

Per step on the 8.4M-cell fit (4 threads, ~57 ms, ~7 ns per cell), the time
is spread across converting the two big cliques to probabilities (one `exp`
pass each, about 25%), expanding potentials (18%), read-out (18%), and the
one log-space message the non-root big clique must send before it is
calibrated (21%). That is close to memory bandwidth.

* **float32 kernels: not done.** They might give ~1.5× by halving memory
  traffic, but would mean making the whole kernel generic over the float
  type, and would give up the float64 precision that lets these fits
  converge properly (the effect seen against JAX's float32 path).
* **Updating probabilities instead of recomputing them** (multiplying by
  incoming messages): not done. Cells that underflow before a large
  incoming message would be lost, and handling that correctly costs about
  what it saves.
* The remaining levers are hardware: more cores (the pool now scales) or the
  JAX path on a GPU.

## Reproducing

```bash
pip install -e . matplotlib
python -m mbi.native build     # optional: Rust backend (needs cargo)
cd benchmarks/aim
python run_ladders.py results                       # original AIM, ~1 hour on 4 cores
python run_ladders.py results_optimized optimized   # optimized AIM
python summarize.py results
python compare.py                                   # before/after table
python profile_breakdown.py --cols 8 --out results/profile_c8.json
python run_ladders.py results_native optimized      # with the native library built
python gen_scaling.py results_optimized/gen_scaling.json
python -m pytest test/test_aim.py test/test_synthetic_data_equivalence.py \
  test/test_estimation_compile_options.py test/test_variable_elimination_numpy.py
```
