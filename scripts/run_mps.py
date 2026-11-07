"""Track B: validate the MPS contraction (Table 5 at n=2, brute force for n=2..5) and time it."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adbq import model as M, mps  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def timed(f, *a, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f(*a)
        best = min(best, time.perf_counter() - t0)
    return r, best


def main():
    res = {"table5": {}, "brute_force": [], "scaling": []}
    for name, sc in M.SCENARIOS.items():
        t = mps.targets(mps.Chain.from_scenario(sc))
        e = M.exact_targets(sc)
        v = [t["expected_loss_usd"], t["all_pay_probability"], t["loss_gt_principal_probability"],
             t["stop_loss_usd"], t["all_exhausted_probability"]]
        res["table5"][name] = {"mps": v, "exact": list(e.values()),
                               "max_rel_diff": max(abs(a / b - 1) for a, b in zip(v, e.values()))}
    for n in (2, 3, 4, 5):
        ch = mps.Chain(n)
        b, tb = timed(mps.brute_force_targets, ch, reps=1)
        m, tm = timed(mps.targets, ch)
        res["brute_force"].append({"n": n, "brute_s": tb, "mps_s": tm,
                                   "max_rel_diff": max(abs(m[k] / b[k] - 1) for k in b)})
        print(res["brute_force"][-1])
    for n in (2, 5, 10, 20, 50, 100, 200, 500):
        r, t = timed(mps.targets, mps.Chain(n))
        res["scaling"].append({"n": n, "seconds": t, **r})
        print(n, f"{t*1e3:.1f} ms", f"all pay {r['all_pay_probability']:.2e}",
              f"loss > half aggregate {r['loss_gt_half_aggregate_probability']:.2e}")
    (OUT / "mps.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
