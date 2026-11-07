"""Writes results/SUMMARY.md: every headline number in the proposal, from the result files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = {k: json.loads((ROOT / f"results/{k}.json").read_text()) for k in ("baselines", "qae", "mps", "resources")}
B, Q, MP, RS = R["baselines"], R["qae"], R["mps"], R["resources"]
T = B["targets"]
L = []
w = L.append

w("# TuringIRL Phase 1 results summary\n")
w("## Queries to 1% relative RMSE (base scenario)\n")
w("| Target | a | " + " | ".join(B["samplers"]) + " | MLAE | best classical / MLAE |")
w("|" + "---|" * (len(B["samplers"]) + 4))
for i, t in enumerate(T):
    v = {n: s["queries_to_1pct"][i] for n, s in B["samplers"].items()}
    q = Q["mlae_ratio1.5"][t]["queries_to_1pct"]
    w(f"| {t} | {Q['mlae_ratio1.5'][t]['a']:.2e} | " + " | ".join(f"{x:.2e}" for x in v.values())
      + f" | {q:.2e} | {min(v.values())/q:.2f} |")
w("\nMLAE fitted slopes: " + ", ".join(f"{Q['mlae_ratio1.5'][t]['slope']:.2f}" for t in T))
w("MLAE constant C = calls x eps x sqrt(a): " + ", ".join(f"{Q['mlae_ratio1.5'][t]['constant_C']:.1f}" for t in T))
for n, s in B["samplers"].items():
    w(f"{n} slopes: " + ", ".join(f"{x:.2f}" for x in s["slope"])
      + f"; rel RMSE at 1,048,576 paths on exhaustion: {s['rel_rmse'][-1][4]:.2%}")
w("\n## Doubling schedule on exhaustion (aliasing)\n")
for n, d in Q["mlae_doubling_exhaustion"].items():
    w(f"- {n} powers, {d['calls']:,} calls: rel RMSE {d['rel_rmse']:.1%}, runs >20% off {d['outlier_rate_gt20pct']:.1%}, "
      f"median outlier bias {d['median_outlier_bias']:+.0%}")
r15 = Q["mlae_ratio1.5"][T[4]]
w(f"- ratio-1.5: runs >20% off at the three largest budgets: {r15['outlier_rate_gt20pct'][-3:]}")
w(f"\n## Equal budget across scenarios ({Q['scenarios']['calls']:,} queries)\n")
w("| Scenario | Exh MLAE | Exh IS | Exh MC | Stop-loss MLAE | Stop-loss IS |\n|---|---|---|---|---|---|")
for s, row in Q["scenarios"]["rel_rmse"].items():
    c = B["scenarios"]["rel_rmse"][s]
    w(f"| {s} | {row[4]:.2%} | {c['Importance sampling'][4]:.1%} | {c['Monte Carlo'][4]:.0%} | {row[3]:.3%} | {c['Importance sampling'][3]:.2%} |")
w("\n## Noise (joint exhaustion, ~1e6 oracle calls)\n")
w("| eps 2q | usable depth | max power used | rel RMSE |\n|---|---|---|---|")
for r in Q["noise_exhaustion_1e6"]:
    w(f"| {r['eps2q']:g} | {r['usable_depth']} | {r['max_power_used']} | {r['rel_rmse']:.1%} |")
w("\n## Tensor network\n")
w("Table 5 max rel diff: " + ", ".join(f"{k} {v['max_rel_diff']:.1e}" for k, v in MP["table5"].items()))
for r in MP["brute_force"]:
    w(f"- n={r['n']}: brute {r['brute_s']:.3f}s, MPS {r['mps_s']*1e3:.1f} ms, max rel diff {r['max_rel_diff']:.1e}")
for r in MP["scaling"]:
    w(f"- n={r['n']}: {r['seconds']*1e3:.1f} ms; all pay {r['all_pay_probability']:.2e}; "
      f"loss > half aggregate {r['loss_gt_half_aggregate_probability']:.2e}")
w("\n## Resources\n")
w(f"Demo circuit: {RS['demo']}")
w("| n | logical qubits | rotations | ANDs | T gates | rotation share |\n|---|---|---|---|---|---|")
for r in RS["logical"]:
    w(f"| {r['n']} | {r['logical_qubits']} | {r['rotations']} | {r['toffoli_ands']} | {r['t_gates']:.2e} | {r['rotation_t_share']:.0%} |")
w("\n| n | target | a | calls | total T | d | physical qubits | runtime |\n|---|---|---|---|---|---|---|---|")
for r in RS["footprint"]:
    rt = r["runtime_s"]
    rts = f"{rt/3600:.1f} h" if rt < 86400 * 3 else f"{rt/86400:.0f} days"
    w(f"| {r['n']} | {r['target']} | {r['a']:.2e} | {r['oracle_calls']:.2e} | {r['total_t']:.2e} | "
      f"{r['code_distance']} | {r['physical_qubits']:.2e} | {rts} |")
w(f"\nOracle time at n=10: {RS['oracle_time_10_s']*1e3:.0f} ms; C={RS['C_exhaustion']:.1f}; "
  f"kappa_IS={RS['kappa_is']:.3f}")
w("Break-even a*: " + ", ".join(f"{k} {v:.1e}" for k, v in RS["break_even"].items()))
(ROOT / "results/SUMMARY.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
