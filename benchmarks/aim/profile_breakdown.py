"""Deep-dive profile of one AIM configuration.

1. Runs instrumented AIM with jax_log_compiles to attribute JIT compile time
   to individual jitted functions / eager ops.
2. Takes the final model's measurements and times:
     - first MirrorDescent fit (includes compile)
     - identical second fit (compile cached -> pure compute)
     - candidate scoring with JAX eager ops vs plain NumPy
"""
import argparse, io, itertools, logging, re, sys, time, json, os
from collections import defaultdict

import jax
jax.config.update("jax_enable_compilation_cache", False)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_aim_experiment as R
from mbi import Dataset, estimation

p = argparse.ArgumentParser()
p.add_argument("--rows", type=int, default=10000)
p.add_argument("--cols", type=int, default=8)
p.add_argument("--bins", type=int, default=16)
p.add_argument("--out", required=True)
a = p.parse_args()

# Capture compile logs
buf = io.StringIO()
h = logging.StreamHandler(buf); h.setLevel(logging.WARNING)
for name in ("jax._src.dispatch", "jax._src.interpreters.pxla", "jax._src.compiler"):
  logging.getLogger(name).addHandler(h)
jax.config.update("jax_log_compiles", True)

df = R.make_raw_table(a.rows, a.cols, a.bins, 0)
dd, dom = R.discretize(df, a.bins)
data = Dataset(dd, dom)
workload = [(cl, 1.0) for cl in itertools.combinations(dom, 2)]

captured = {}
class Cap(R.InstrumentedAIM):
  def _estimate(self, key, domain, measurements, **kw):
    captured["meas"] = list(measurements); captured["kw"] = kw
    return super()._estimate(key, domain, measurements, **kw)

mech = Cap(1.0, 1e-9, prng=np.random.RandomState(0))
t0 = time.perf_counter()
model, synth = mech.run(data, workload, num_synth_rows=a.rows)
t_total = time.perf_counter() - t0
jax.config.update("jax_log_compiles", False)

# Aggregate compile log lines: "Finished XLA compilation of jit(name) in X sec"
by_fn = defaultdict(lambda: [0, 0.0])
pat = re.compile(r"Finished (?:XLA compilation|tracing \+ transforming|jaxpr to MLIR module conversion) (?:of|for) (\S+?)(?: for| with|\s).*?in ([0-9.e-]+) ?s")
for line in buf.getvalue().splitlines():
  m = pat.search(line)
  if m:
    fn = m.group(1); dt = float(m.group(2))
    by_fn[fn][0] += 1; by_fn[fn][1] += dt
top = sorted(by_fn.items(), key=lambda kv: -kv[1][1])[:20]

# Steady-state fit cost
meas = captured["meas"]
def fit():
  t = time.perf_counter()
  m = estimation.MirrorDescent().estimate(dom, meas, iters=1000, **captured["kw"])
  jax.block_until_ready(m.marginals); return time.perf_counter() - t, m
c0 = R.compile_seconds(); t_first, m1 = fit(); c_first = R.compile_seconds() - c0
c0 = R.compile_seconds(); t_second, _ = fit(); c_second = R.compile_seconds() - c0

# Candidate scoring: JAX eager vs numpy
cands = R.aim_mod.compile_workload(workload)
supported = [cl for cl in cands if m1.marginals.supports(cl)]
t = time.perf_counter()
for cl in supported: np.asarray(m1.project(cl).datavector())
t_jax_proj = time.perf_counter() - t
t = time.perf_counter()
for cl in supported: np.asarray(m1.project(cl).datavector())
t_jax_proj2 = time.perf_counter() - t
np_marg = {c: np.asarray(f.values) for c, f in m1.marginals.tables.items()}
t = time.perf_counter()
for cl in supported:
  parent = m1.marginals.parent(cl)
  axes = tuple(i for i, x in enumerate(parent) if x not in cl)
  np_marg[parent].sum(axis=axes)
t_np_proj = time.perf_counter() - t

n_maps = sum(1 for _ in open('/proc/self/maps'))
res = dict(config=vars(a), process_memory_maps=n_maps, t_aim_total=t_total, rounds=mech.rounds_run,
           compile_total=R.compile_seconds(), n_compiles=R.compile_count(),
           phases=dict(mech.timer.t),
           top_compiles=[(k, v[0], round(v[1], 2)) for k, v in top],
           final_fit_first_s=t_first, final_fit_first_compile_s=c_first,
           final_fit_cached_s=t_second, final_fit_cached_compile_s=c_second,
           n_measurements=len(meas), n_supported_candidates=len(supported),
           project_jax_first_s=t_jax_proj, project_jax_cached_s=t_jax_proj2,
           project_numpy_s=t_np_proj)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
