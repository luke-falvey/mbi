"""Run AIM benchmark ladders in parallel threads.

Each ladder runs in its own thread and is pinned to its own CPU core (via
taskset, with XLA limited to one thread) so concurrent ladders don't distort
each other's timings. Within a ladder, experiments run from small to large
and the ladder stops after the first experiment that exceeds STOP_AFTER
seconds (or hits the hard TIMEOUT).
"""

import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
IMPL = sys.argv[2] if len(sys.argv) > 2 else "reference"  # or "optimized"
os.makedirs(OUT, exist_ok=True)

STOP_AFTER = 600  # 10 minutes
TIMEOUT = 1500  # hard cap per experiment (25 minutes)

LADDERS = {
    # vary rows; 8 mixed-type columns, 16 bins
    "rows": [dict(rows=r, cols=8, bins=16)
             for r in (1_000, 10_000, 100_000, 1_000_000, 5_000_000,
                       20_000_000)],
    # vary columns; 10k rows, 16 bins
    "cols": [dict(rows=10_000, cols=c, bins=16)
             for c in (4, 6, 8, 12, 16, 20, 24, 32)],
    # vary per-column domain size; 10k rows, 8 columns
    "bins": [dict(rows=10_000, cols=8, bins=b)
             for b in (4, 16, 32, 64, 128, 256)],
    # grow everything together
    "combined": [dict(rows=r, cols=c, bins=b) for r, c, b in (
        (5_000, 5, 8), (20_000, 8, 16), (100_000, 12, 16),
        (500_000, 16, 32), (2_000_000, 20, 32), (5_000_000, 24, 32))],
}

lock = threading.Lock()


def run_ladder(core, ladder, configs):
  out = os.path.join(OUT, f"{ladder}.jsonl")
  env = dict(os.environ)
  env["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                      "intra_op_parallelism_threads=1")
  env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "1"
  env["PYTHONWARNINGS"] = "ignore"
  for cfg in configs:
    name = f"{ladder}:r{cfg['rows']}_c{cfg['cols']}_b{cfg['bins']}"
    cmd = ["taskset", "-c", str(core), sys.executable,
           os.path.join(HERE, "run_aim_experiment.py"),
           "--rows", str(cfg["rows"]), "--cols", str(cfg["cols"]),
           "--bins", str(cfg["bins"]), "--name", name, "--out", out,
           "--impl", IMPL]
    t0 = time.time()
    with lock:
      print(f"[core {core}] START {name}", flush=True)
    try:
      proc = subprocess.run(cmd, env=env, timeout=TIMEOUT,
                            capture_output=True, text=True)
      status = "ok" if proc.returncode == 0 else f"error {proc.returncode}"
      if proc.returncode != 0:
        with open(os.path.join(OUT, f"{ladder}.stderr"), "a") as f:
          f.write(f"=== {name}\n{proc.stderr[-5000:]}\n")
    except subprocess.TimeoutExpired:
      status = "timeout"
    dt = time.time() - t0
    with open(out, "a") as f:
      f.write(json.dumps({"name": name, "event": "exit", "status": status,
                          "elapsed": dt, **cfg}) + "\n")
    with lock:
      print(f"[core {core}] END   {name} {status} {dt:.0f}s", flush=True)
    if dt > STOP_AFTER or status != "ok":
      break


threads = [threading.Thread(target=run_ladder, args=(i, k, v))
           for i, (k, v) in enumerate(LADDERS.items())]
for t in threads:
  t.start()
for t in threads:
  t.join()
print("ALL DONE", flush=True)
