"""AIM wall time and fit quality (mean 2-way TV error) for one config.

Usage: python run_aim_quality.py ROWS COLS {mirror_descent,accelerated} [MAX_ITERS] [SEED]
"""
import sys, time, itertools, os
import numpy as np
sys.path.insert(0, "/home/user/mbi/mechanisms"); sys.path.insert(0,"/home/user/mbi/benchmarks/aim")
import run_aim_experiment as R
from aim import AIM
from mbi import Dataset
rows, cols = int(sys.argv[1]), int(sys.argv[2]); est=sys.argv[3]; iters=int(sys.argv[4]) if len(sys.argv)>4 else 1000; seed=int(sys.argv[5]) if len(sys.argv)>5 else 0
df=R.make_raw_table(rows, cols, 16, seed); dd, dom=R.discretize(df, 16); data=Dataset(dd, dom)
wl=[(cl,1.0) for cl in itertools.combinations(dom,2)]
np.random.seed(seed)
t=time.perf_counter()
model, synth = AIM(1.0,1e-9,prng=np.random.RandomState(seed), estimator=est, max_iters=iters).run(data, wl)
dt=time.perf_counter()-t
def tv(src):
  return np.mean([0.5*np.abs(np.asarray(data.project(c).datavector())/rows - np.asarray(src.project(c).datavector())/float(np.asarray(src.project(c).datavector()).sum())).sum() for c,_ in wl])
print(f"seed={seed} rows={rows} cols={cols} est={est} iters={iters}: {dt:.1f}s  model TV={tv(model):.4f}  synth TV={tv(synth):.4f}")
