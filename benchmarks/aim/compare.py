"""Before/after tables across benchmark result folders.

Usage: python compare.py [results results_optimized results_native]
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
dirs = sys.argv[1:] or ["results", "results_optimized", "results_native"]
labels = {"results": "original", "results_optimized": "phases 1-2 (JAX)",
          "results_native": "native (Rust)"}

def load(d):
  runs = {}
  d = os.path.join(HERE, d)
  if not os.path.isdir(d):
    return runs
  for f in sorted(os.listdir(d)):
    if not f.endswith(".jsonl"): continue
    for l in open(os.path.join(d, f)):
      r = json.loads(l); run = runs.setdefault(r["name"], {"rounds": 0})
      if r["event"] == "round": run["rounds"] = max(run["rounds"], r["round"])
      elif r["event"] == "done": run["done"] = r
      elif r["event"] == "exit": run["exit"] = r
  return runs

def cell(run):
  if run is None: return "–", None
  ex, done = run.get("exit", {}), run.get("done")
  if done:
    t = done["t_wall"]
    return (f"{t:.1f}s" if t < 10 else f"{t:.0f}s") + f" ({done['rounds']} rds)", t
  st = ex.get("status", "?")
  label = {"timeout": "timeout", "error 1": "crashed"}.get(st, st)
  return f"{label} at {ex.get('elapsed', 0):.0f}s ({run['rounds']} rds)", None

data = [load(d) for d in dirs]
names = set().union(*data)
order = lambda n: (n.split(":")[0], [int(x[1:]) for x in n.split(":")[1].split("_")])
hdr = ["run (rows / cols / bins)"] + [labels.get(d, d) for d in dirs]
if len(dirs) > 1: hdr.append(f"speedup {labels.get(dirs[-1], dirs[-1])} vs {labels.get(dirs[0], dirs[0])}")
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for name in sorted(names, key=order):
  if name.startswith("multicore"): continue
  ladder, cfg = name.split(":"); r, c, b = (x[1:] for x in cfg.split("_"))
  cells = [cell(d.get(name)) for d in data]
  row = [f"{ladder}: {int(r):,} / {c} / {b}"] + [x[0] for x in cells]
  if len(dirs) > 1:
    a, z = cells[0][1], cells[-1][1]
    row.append(f"{a / z:.0f}×" if a and z else ("completes" if z and name in data[0] else "–"))
  print("| " + " | ".join(row) + " |")
