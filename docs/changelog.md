# Changelog

This page documents the history of changes for each version of `mbi`.

## Unreleased

* Faster AIM and estimation when compile time dominates (see `benchmarks/aim/REPORT.md`). None of these change results beyond floating-point rounding:
    * New opt-in `Estimator` options: `fuse_steps=False` compiles a single optimization step instead of a fused `lax.scan` (~5x faster to compile on CPU), `fast_compile=True` disables XLA backend optimizations for small models (step, init/finalize oracle and convergence check), and `parallel_compile=True` compiles the step and convergence-check programs on background threads.
    * The convergence check (`primal_dual_gap`) in `Estimator.estimate` is now one jitted program instead of an op-by-op gradient plus a separately compiled oracle call.
    * New `marginal_oracles.variable_elimination_numpy`, a NumPy (no-JIT) equivalent of `variable_elimination` for querying many out-of-model marginals.
    * `MarkovRandomField.synthetic_data` groups parent configurations with integer keys instead of `np.unique(..., axis=0)`, which was the bottleneck for large row counts. Output is bit-for-bit identical for the same seed.
    * `mechanisms/aim.py` scores candidates with NumPy, merges repeated measurements of the same clique, uses the options above, caches junction-tree sizes, and clears JAX caches when the process nears its memory-map limit (long runs previously crashed with "LLVM compilation error: Cannot allocate memory"). It also gains an opt-in `intermediate_iters` setting.

* Optional native (Rust) backend, `mbi.native` (build with `python -m mbi.native build`; see `native/README.md`). It removes JIT compilation from the estimation loop entirely: the model structure is passed to the Rust kernels as data. When the library is unavailable everything falls back to JAX.
    * `NativeMirrorDescent`: drop-in `MirrorDescent` running the whole optimization in float64 native code. It falls back to JAX for constraints, callbacks, non-identity queries and custom oracles.
    * `out_of_model_marginals`: all pairwise marginals from one calibrated junction tree (one sweep per source attribute) instead of a variable elimination per pair.
    * `hypothetical_model_sizes`, `expand`, `junction_tree_beliefs`, and `MarkovRandomField.synthetic_data(use_native=True)`.
    * AIM uses all of these automatically when the library is available.
* Large-data performance: `NativeAcceleratedMethod` and `AIM(estimator="accelerated")`; opt-in `Estimator.loss_rtol` / `AIM(loss_rtol=...)` stopping rule; native multi-threaded marginal counting (`native.marginal_counts`) and synthetic-data sampling (`synthetic_data(use_native=True)`); native kernels with an intermediate-table hierarchy, messages computed from probability tables, and a persistent thread pool. 1M-row AIM: 28.5 min → 1.5 min on 4 cores with the recommended settings.
* `junction_tree.greedy_order` and `junction_tree.hypothetical_model_size` are 7–9x faster with identical results (cached elimination costs; clique sizes computed directly from the elimination instead of building a networkx junction tree).

## Version 2.0.0 - 09/2026

* New unified Estimator API & optimizers. MBI introduced an Estimator class replacing legacy ad-hoc estimation functions. Optimizers including MirrorDescent, DualAveraging, InteriorGradient, and LBFGS now share a common estimation lifecycle and benefit from shared components and a unified API. Key features include:
    * Warm-starting (warm_start=model): Initialize optimization from a previously estimated model, automatically expanding potentials to cover newly observed cliques.
    * Tolerance-based early stopping (tol, patience): Terminate estimation early when relative loss improvement plateaus.
    * Analytic Lipschitz computation: Automatically computes exact L2 Lipschitz constants directly from measurement metadata to tune step sizes without manual parameter sweeps.
    * New estimator models: Added MixtureOfProductsEstimator (i.e., [Relaxed Projection](https://arxiv.org/html/2103.06641v2)) and ReweightedDatasetEstimator (from [PMW^{pub}](https://proceedings.mlr.press/v139/liu21w/liu21w.pdf)) in mbi.extensions.
* Structural constraints (possible/impossible values & functional dependencies). A new first-class Constraint dataclass allows declaring exact domain validity rules in three flexible ways: valid / invalid possible combinations or functional dependencies. Constraints are surfaced throughout the API and automatically converted into log-space potentials (0.0 for valid, -inf for invalid) where needed.
* Library-wide ahead-of-time precompilation. Ahead-of-time compilation has been systematically introduced across the core library (estimation, inference, data generation, and marginal computation) to mitigate JIT tracing latency. Some mechanisms (e.g., AIM) built on top of MBI require compiling a large number of different programs, and compilation time can be a significant overhead.
* Serialization (save / load). Added native numpy-based serialization for MarkovRandomField, LinearMeasurement, and CliqueVector, ensuring models and intermediates can be checkpointed and restored reliably.
* Domain compression: Added LinearMeasurement.compress() and Dataset.compress() for domain reduction.
* Modernization & code quality.
    * Migrated internal structures from attrs to standard Python dataclasses.
    * Deprecated Domain.attrs in favor of Domain.attributes.
    * Migrated from pytype to pyrefly for static analysis checks, and improved type safety across the library.

## Version 1.0 - 11/2024

This library recently underwent major refactorings.  Below are a list of notable changes.

* Library has been modernized to use more recent python features.  Therefore, we require Python>=3.9.
* JAX is now used as the numerical backend rather than numpy.  This means the code now natively supports running on GPUs, although the scalability advantages have not yet been tested.
* To more naturally support JAX with JIT compilation, the code has been reorganized into a more functional design.  This design follows closely from how we describe the approach mathematically in our papers.
* Classes now have more narrowly defined scope, and have been removed where they did not provide significant utility.  Dataclasses are used liberally throughout the code.
* A new belief propagation algorithm is used which is more space efficient than the prior one in settings where the maximal cliques in the Junction Tree are larger than the measured cliques.  This new algorithm is also significantly faster when running on GPUs in some cases.
* Expanded test coverage for marginal inference, we correctly handle a number of tricky edge cases.
* Added type information to most functions for better documentation.  Also added example usages to some functions in the form of doctests.  More will be added in the future.
* Setup continuous integration tests on GitHub.

Currently, not all functionality that was previously supported been integrated into the new design.  However, the core features that are used by the majority of the use cases have been.  These left out functionalities are discussed in the end of this document. 

## Version 0.1.0 (Initial Release)

* Initial release of `mbi`.
