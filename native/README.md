# mbi native kernels (Rust)

A small, dependency-free Rust library that `mbi.native` loads with `ctypes`.
It implements, with no JIT compilation:

* `mbi_mirror_descent`: the whole `MirrorDescent.estimate` loop (HUGIN
  message passing, L2 loss over identity measurements, Armijo line search,
  primal-dual-gap early stopping);
* `mbi_oracle` / `mbi_beliefs`: marginals and calibrated junction-tree
  beliefs;
* `mbi_model_sizes`: junction-tree sizes for AIM's candidate filtering.

The model structure is passed in as flat index arrays built by
`mbi/native.py`, so new model structures cost nothing to "compile". This is
what makes AIM fast: the JAX implementation compiles new XLA programs for
every model structure it sees.

## Build

```bash
python -m mbi.native build
```

This runs `cargo build --release --offline` with `RUSTFLAGS="-C
target-cpu=native"` (set `RUSTFLAGS` yourself to target other CPUs) and
copies the library into `src/mbi/_native/`. Alternatively point
`MBI_NATIVE_LIB` at a prebuilt library. Without it, `mbi` uses the JAX code
paths.

`MBI_NATIVE_THREADS` caps the worker threads used for large tables (default:
the CPUs the process may run on). Results do not depend on the thread count.
