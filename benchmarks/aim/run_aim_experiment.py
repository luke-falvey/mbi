"""Run one instrumented AIM experiment and write timings as JSON lines.

Generates a synthetic mixed-type table (continuous, integer, categorical
strings, booleans), discretizes it, runs AIM from mechanisms/aim.py with
per-phase timers, and generates synthetic data.

Usage:
  python run_aim_experiment.py --rows 10000 --cols 8 --bins 16 --out out.jsonl
"""

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict

import jax

jax.config.update("jax_enable_compilation_cache", False)

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "mechanisms"))

import mbi  # noqa: E402
from mbi import Dataset, Domain, LinearMeasurement, estimation, junction_tree  # noqa: E402
import aim as aim_mod  # noqa: E402

# ---------------------------------------------------------------------------
# JIT compile-time accounting via JAX's monitoring hooks.
# ---------------------------------------------------------------------------
COMPILE = defaultdict(float)
COMPILE_COUNT = defaultdict(int)


def _listener(event, duration, **kwargs):
  COMPILE[event] += duration
  COMPILE_COUNT[event] += 1


jax.monitoring.register_event_duration_secs_listener(_listener)


def compile_seconds():
  # Backend compile + tracing/lowering is the full JIT overhead.
  keys = [
      "/jax/compilation_cache/compile_time_saved_sec",  # ignored if absent
  ]
  total = 0.0
  for k, v in COMPILE.items():
    if k in (
        "/jax/core/compile/backend_compile_duration",
        "/jax/core/compile/jaxpr_trace_duration",
        "/jax/core/compile/jaxpr_to_mlir_module_duration",
    ):
      total += v
  return total


def compile_count():
  return COMPILE_COUNT.get("/jax/core/compile/backend_compile_duration", 0)


# ---------------------------------------------------------------------------
# Data generation: correlated mixed-type table.
# ---------------------------------------------------------------------------
KINDS = ["continuous", "integer", "categorical", "boolean"]


def make_raw_table(rows, cols, bins, seed):
  """Mixed-type table with a latent-factor + chain dependency structure."""
  rng = np.random.default_rng(seed)
  z = rng.normal(size=rows)  # shared latent factor
  prev = rng.normal(size=rows)
  out = {}
  for j in range(cols):
    kind = KINDS[j % len(KINDS)]
    signal = 0.6 * z + 0.5 * prev + rng.normal(size=rows)
    name = f"{kind[:4]}_{j}"
    if kind == "continuous":
      out[name] = np.exp(0.5 * signal) * 1000.0  # skewed, like income
    elif kind == "integer":
      out[name] = np.clip(np.round(40 + 12 * signal), 0, 120).astype(np.int64)
    elif kind == "categorical":
      k = bins
      # zipf-ish category probabilities shifted by the signal
      logits = -np.log1p(np.arange(k))[None, :] + np.outer(
          signal, np.linspace(-1, 1, k)
      )
      g = rng.gumbel(size=(rows, k))
      codes = np.argmax(logits + g, axis=1)
      out[name] = np.array([f"cat_{c}" for c in range(k)], dtype=object)[codes]
    else:
      out[name] = signal > 0.3
    prev = signal
  return pd.DataFrame(out)


def discretize(df, bins):
  """Map every column to integer codes 0..k-1; return (data dict, domain)."""
  data, shape = {}, {}
  for c in df.columns:
    s = df[c]
    if s.dtype == bool:
      codes = s.astype(np.int64).to_numpy()
      k = 2
    elif not pd.api.types.is_numeric_dtype(s):
      cat = pd.Categorical(s)
      codes = cat.codes.astype(np.int64)
      k = len(cat.categories)
    else:
      # quantile bins (fixed public bin edges would be needed for strict DP)
      q = pd.qcut(s, q=bins, labels=False, duplicates="drop")
      codes = q.to_numpy().astype(np.int64)
      k = int(codes.max()) + 1
    data[c] = codes
    shape[c] = k
  return data, Domain(list(shape), list(shape.values()))


# ---------------------------------------------------------------------------
# Instrumented AIM (logic mirrors mechanisms/aim.py AIM.run).
# ---------------------------------------------------------------------------
class Timer:

  def __init__(self):
    self.t = defaultdict(float)
    self.n = defaultdict(int)

  def add(self, key, dt):
    self.t[key] += dt
    self.n[key] += 1


class InstrumentedAIM(aim_mod.AIM):

  def __init__(self, *a, log=None, **k):
    super().__init__(*a, **k)
    self.timer = Timer()
    self.log = log or (lambda rec: None)

  def worst_approximated(self, candidates, answers, model, eps, sigma):
    errors, sensitivity = {}, {}
    n_in, n_out = 0, 0
    t_in = t_out = 0.0
    for cl in candidates:
      wgt = candidates[cl]
      x = answers[cl]
      bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
      t0 = time.perf_counter()
      supported = model.marginals.supports(cl)
      xest = np.asarray(model.project(cl).datavector())
      dt = time.perf_counter() - t0
      if supported:
        n_in += 1
        t_in += dt
      else:
        n_out += 1
        t_out += dt
      errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
      sensitivity[cl] = abs(wgt)
    self.timer.add("select/project_in_model", t_in)
    self.timer.add("select/project_out_of_model(VE)", t_out)
    self._last_counts = (n_in, n_out)
    return self.exponential_mechanism(errors, eps, max(sensitivity.values()))

  def _estimate(self, key, domain, measurements, **kw):
    c0 = compile_seconds()
    t0 = time.perf_counter()
    model = estimation.MirrorDescent().estimate(
        domain, measurements, iters=self.max_iters, **kw
    )
    jax.block_until_ready(model.marginals)
    dt = time.perf_counter() - t0
    dc = compile_seconds() - c0
    self.timer.add(key, dt)
    self.timer.add(key + "/jit_compile", dc)
    return model, dt, dc

  def run(self, data, workload, num_synth_rows=None):
    T = self.timer
    rounds = self.rounds or 16 * len(data.domain)

    t0 = time.perf_counter()
    candidates = aim_mod.compile_workload(workload)
    answers = {cl: np.asarray(data.project(cl).datavector()) for cl in candidates}
    T.add("setup/answer_all_candidates", time.perf_counter() - t0)

    oneway = [cl for cl in candidates if len(cl) == 1]
    sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
    epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)
    measurements = []
    rho_used = len(oneway) * 0.5 / sigma**2
    for cl in oneway:
      x = data.project(cl).datavector()
      y = x + self.gaussian_noise(sigma, x.size)
      measurements.append(LinearMeasurement(y, cl, stddev=sigma))

    model, _, _ = self._estimate("fit/initial", data.domain, measurements)

    t, terminate = 0, False
    potentials = None
    while not terminate:
      t += 1
      r0 = time.perf_counter()
      c_round0 = compile_seconds()
      if self.rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
        remaining = self.rho - rho_used
        sigma = np.sqrt(1 / (2 * 0.9 * remaining))
        epsilon = np.sqrt(8 * 0.1 * remaining)
        terminate = True
      rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
      size_limit = self.max_model_size * rho_used / self.rho

      a = time.perf_counter()
      small = aim_mod.filter_candidates(candidates, model, size_limit)
      t_filter = time.perf_counter() - a
      T.add("select/filter_candidates(jtree size)", t_filter)

      a = time.perf_counter()
      c_sel0 = compile_seconds()
      cl = self.worst_approximated(small, answers, model, epsilon, sigma)
      t_select = time.perf_counter() - a
      c_sel = compile_seconds() - c_sel0
      T.add("select/worst_approximated", t_select)
      T.add("select/jit_compile", c_sel)

      a = time.perf_counter()
      n = data.domain.size(cl)
      x = data.project(cl).datavector()
      y = x + self.gaussian_noise(sigma, n)
      measurements.append(LinearMeasurement(y, cl, stddev=sigma))
      z = np.asarray(model.project(cl).datavector())
      T.add("measure", time.perf_counter() - a)

      pcliques = list(set(M.clique for M in measurements))
      potentials = model.potentials.expand(pcliques)
      model, t_fit, c_fit = self._estimate(
          "fit/round", data.domain, measurements, warm_start=potentials
      )
      w = np.asarray(model.project(cl).datavector())
      if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
        sigma /= 2
        epsilon *= 2

      jt, _ = junction_tree.make_junction_tree(data.domain, model.cliques)
      mx = junction_tree.maximal_cliques(jt)
      self.log({
          "event": "round",
          "round": t,
          "selected": list(cl),
          "n_candidates": len(small),
          "n_in_model": self._last_counts[0],
          "n_out_of_model": self._last_counts[1],
          "n_measurements": len(measurements),
          "model_size_mb": sum(data.domain.size(c) for c in mx) * 8 / 2**20,
          "max_clique_cells": max(data.domain.size(c) for c in mx),
          "t_round": time.perf_counter() - r0,
          "t_filter": t_filter,
          "t_select": t_select,
          "t_select_compile": c_sel,
          "t_fit": t_fit,
          "t_fit_compile": c_fit,
          "t_compile_round": compile_seconds() - c_round0,
      })
    self.rounds_run = t

    model, _, _ = self._estimate(
        "fit/final", data.domain, measurements, warm_start=potentials
    )
    a = time.perf_counter()
    synth = model.synthetic_data(rows=num_synth_rows)
    T.add("generate/synthetic_data", time.perf_counter() - a)
    return model, synth


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--rows", type=int, required=True)
  p.add_argument("--cols", type=int, required=True)
  p.add_argument("--bins", type=int, default=16)
  p.add_argument("--epsilon", type=float, default=1.0)
  p.add_argument("--max_model_size", type=float, default=80)
  p.add_argument("--max_iters", type=int, default=1000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--name", default="")
  p.add_argument("--out", required=True)
  args = p.parse_args()

  fh = open(args.out, "a")

  def log(rec):
    rec = {"name": args.name, **rec}
    fh.write(json.dumps(rec) + "\n")
    fh.flush()

  log({"event": "start", **vars(args), "jax": jax.__version__,
       "threads_env": os.environ.get("XLA_FLAGS", "")})
  wall0 = time.perf_counter()

  t0 = time.perf_counter()
  df = make_raw_table(args.rows, args.cols, args.bins, args.seed)
  t_gen = time.perf_counter() - t0
  t0 = time.perf_counter()
  data_dict, domain = discretize(df, args.bins)
  data = Dataset(data_dict, domain)
  t_disc = time.perf_counter() - t0
  del df

  workload = [(cl, 1.0) for cl in itertools.combinations(domain, 2)]
  np.random.seed(args.seed)
  mech = InstrumentedAIM(
      args.epsilon, 1e-9, max_model_size=args.max_model_size,
      max_iters=args.max_iters, log=log,
      prng=np.random.RandomState(args.seed),
  )
  t0 = time.perf_counter()
  model, synth = mech.run(data, workload, num_synth_rows=args.rows)
  t_aim = time.perf_counter() - t0

  # quality: avg total-variation error over 2-way workload
  errs = []
  for cl, _ in workload:
    X = np.asarray(data.project(cl).datavector())
    Y = np.asarray(synth.project(cl).datavector())
    errs.append(0.5 * np.abs(X / X.sum() - Y / Y.sum()).sum())

  log({
      "event": "done",
      "domain": dict(zip(domain.attributes, domain.shape)),
      "rounds": mech.rounds_run,
      "t_wall": time.perf_counter() - wall0,
      "t_make_raw": t_gen,
      "t_discretize": t_disc,
      "t_aim_total": t_aim,
      "t_compile_total": compile_seconds(),
      "n_compiles": compile_count(),
      "phases": dict(mech.timer.t),
      "phase_calls": dict(mech.timer.n),
      "avg_tv_error": float(np.mean(errs)),
  })


if __name__ == "__main__":
  main()
