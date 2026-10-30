"""Before/after table: results/ (original AIM) vs results_optimized/."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

def load(d):
  runs = {}
  for f in sorted(os.listdir(d)):
    if not f.endswith(".jsonl"): continue
    for l in open(os.path.join(d, f)):
      r = json.loads(l); run = runs.setdefault(r["name"], {"rounds": 0})
      if r["event"] == "round": run["rounds"] = max(run["rounds"], r["round"])
      elif r["event"] == "done": run["done"] = r
      elif r["event"] == "exit": run["exit"] = r
  return runs

def cell(run):
  if run is None: return "–", None, None
  ex, done = run.get("exit", {}), run.get("done")
  if done:
    t = done["t_wall"]; n = done["rounds"]
    return f"{t:.0f}s ({n} rounds)", t, t / max(n, 1)
  st = ex.get("status", "?")
  label = {"timeout": "timeout", "error 1": "crashed"}.get(st, st)
  el = ex.get("elapsed", 0)
  per = el / run["rounds"] if run["rounds"] and el else None
  return f"{label} at {el:.0f}s ({run['rounds']} rounds)", None, per

before, after = load(os.path.join(HERE, "results")), load(os.path.join(HERE, "results_optimized"))
order = lambda n: (n.split(":")[0], [int(x[1:]) for x in n.split(":")[1].split("_")])
print("| run (rows / cols / bins) | original | optimized | speedup (total) | speedup (per round) |")
print("|---|---|---|---|---|")
for name in sorted(set(before) | set(after), key=order):
  if name.startswith("multicore"): continue
  b, tb, pb = cell(before.get(name)); a, ta, pa = cell(after.get(name))
  sp = f"{tb/ta:.1f}×" if tb and ta else ("runs to completion" if ta and not tb and name in before else "–")
  spr = f"{pb/pa:.1f}×" if pb and pa else "–"
  ladder, cfg = name.split(":")
  r, c, bi = (x[1:] for x in cfg.split("_"))
  print(f"| {ladder}: {int(r):,} / {c} / {bi} | {b} | {a} | {sp} | {spr} |")
