"""Time data-dependent steps (discretize, marginal counting, synthetic
generation) at large row counts, using a fixed fitted model."""
import itertools, json, sys, time, os
import jax
jax.config.update("jax_enable_compilation_cache", False)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_aim_experiment as R
from mbi import Dataset, LinearMeasurement, estimation

out = []
cols, bins = 8, 16
for rows in (100_000, 1_000_000, 5_000_000, 10_000_000):
  t = time.perf_counter(); df = R.make_raw_table(rows, cols, bins, 0); t_make = time.perf_counter() - t
  t = time.perf_counter(); dd, dom = R.discretize(df, bins); data = Dataset(dd, dom); t_disc = time.perf_counter() - t
  del df
  pairs = list(itertools.combinations(dom, 2))
  t = time.perf_counter()
  answers = {cl: np.asarray(data.project(cl).datavector()) for cl in pairs}
  t_ans = time.perf_counter() - t
  # chain-structured model: one-way + adjacent pairs (+ a triangle-ish extra)
  cl_fit = [(a,) for a in dom] + [(dom.attributes[i], dom.attributes[i+1]) for i in range(cols-1)]
  meas = [LinearMeasurement(np.asarray(data.project(c).datavector()), c, stddev=1.0) for c in cl_fit]
  model = estimation.MirrorDescent().estimate(dom, meas, iters=200)
  model.synthetic_data(rows=1000)  # warm up
  t = time.perf_counter(); s = model.synthetic_data(rows=rows); t_gen = time.perf_counter() - t
  rec = dict(rows=rows, t_make_raw=t_make, t_discretize=t_disc,
             t_count_all_2way=t_ans, n_2way=len(pairs), t_synthetic_data=t_gen)
  print(json.dumps(rec), flush=True); out.append(rec)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/gen_scaling.json", "w"), indent=1)
