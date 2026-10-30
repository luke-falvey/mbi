"""Re-run specific ladder experiments (optimized impl) and replace their
records in results_optimized/*.jsonl. Each worker is pinned to one core."""
import json, os, queue, subprocess, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_optimized")
names = sys.argv[1:]
# drop old records
for f in os.listdir(OUT):
  if f.endswith(".jsonl"):
    p = os.path.join(OUT, f)
    keep = [l for l in open(p) if json.loads(l)["name"] not in names]
    open(p, "w").writelines(keep)
q = queue.Queue()
for n in sorted(names, key=lambda n: -int(n.split("_c")[1].split("_")[0]) * int(n.split(":r")[1].split("_")[0]) ** 0.3):
  q.put(n)
env = dict(os.environ, PYTHONWARNINGS="ignore",
           XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
def worker(core):
  while True:
    try: name = q.get_nowait()
    except queue.Empty: return
    ladder, cfg = name.split(":")
    rows, cols, bins = (int(x[1:]) for x in cfg.split("_"))
    out = os.path.join(OUT, f"{ladder}.jsonl")
    t0 = time.time()
    try:
      p = subprocess.run(["taskset", "-c", str(core), sys.executable, os.path.join(HERE, "run_aim_experiment.py"),
                          "--rows", str(rows), "--cols", str(cols), "--bins", str(bins), "--name", name,
                          "--out", out, "--impl", "optimized"], env=env, timeout=1500, capture_output=True)
      status = "ok" if p.returncode == 0 else f"error {p.returncode}"
    except subprocess.TimeoutExpired:
      status = "timeout"
    with open(out, "a") as f:
      f.write(json.dumps({"name": name, "event": "exit", "status": status, "elapsed": time.time() - t0,
                          "rows": rows, "cols": cols, "bins": bins}) + "\n")
    print(f"END {name} {status} {time.time()-t0:.0f}s", flush=True)
ts = [threading.Thread(target=worker, args=(c,)) for c in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
print("ALL DONE", flush=True)
