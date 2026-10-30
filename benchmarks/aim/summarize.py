"""Summarize AIM ladder results into a markdown table."""
import glob, json, os, sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
runs = defaultdict(lambda: {"rounds": []})
for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
  for line in open(f):
    r = json.loads(line)
    run = runs[r["name"]]
    if r["event"] == "round":
      run["rounds"].append(r)
    else:
      run[r["event"]] = r

print("| run | rows | cols | cells(max dom) | status | wall s | rounds | compile s (%) | fit s | select s | gen s | TV err |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, run in runs.items():
  ex = run.get("exit", {}); done = run.get("done"); st = run.get("start", {})
  rs = run["rounds"]
  if done:
    ph = done["phases"]
    fit = ph.get("fit/initial",0)+ph.get("fit/round",0)+ph.get("fit/final",0)
    sel = ph.get("select/worst_approximated",0)+ph.get("select/filter_candidates(jtree size)",0)
    comp = done["t_compile_total"]
    print(f"| {name} | {st.get('rows')} | {st.get('cols')} | {max(done['domain'].values())} | {ex.get('status','?')} | "
          f"{done['t_wall']:.0f} | {done['rounds']} | {comp:.0f} ({100*comp/done['t_aim_total']:.0f}%) | "
          f"{fit:.0f} | {sel:.0f} | {ph.get('generate/synthetic_data',0):.1f} | {done['avg_tv_error']:.3f} |")
  else:
    comp = sum(r["t_compile_round"] for r in rs); tot = sum(r["t_round"] for r in rs)
    fit = sum(r["t_fit"] for r in rs); sel = sum(r["t_select"]+r["t_filter"] for r in rs)
    pct = f"{100*comp/tot:.0f}%" if tot else "-"
    print(f"| {name} | {st.get('rows')} | {st.get('cols')} | - | {ex.get('status','running')} | "
          f"{ex.get('elapsed',0):.0f} | {len(rs)}+ | {comp:.0f} ({pct}) | {fit:.0f} | {sel:.0f} | - | - |")
