# AIM performance: benchmarks, bottlenecks, speed-up plan and results

## TL;DR

> **Update: Phases 1 and 2 are implemented** (see [section 5](#5-results-after-implementing-phases-1-and-2)).
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

## Reproducing

```bash
pip install -e . matplotlib
cd benchmarks/aim
python run_ladders.py results                       # original AIM, ~1 hour on 4 cores
python run_ladders.py results_optimized optimized   # optimized AIM
python summarize.py results
python compare.py                                   # before/after table
python profile_breakdown.py --cols 8 --out results/profile_c8.json
python gen_scaling.py results_optimized/gen_scaling.json
python -m pytest test/test_aim.py test/test_synthetic_data_equivalence.py \
  test/test_estimation_compile_options.py test/test_variable_elimination_numpy.py
```
