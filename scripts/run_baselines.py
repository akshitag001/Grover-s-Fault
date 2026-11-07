"""Classical baselines: relative RMSE vs budget for all four samplers (base scenario),
plus equal-budget accuracy across all five scenarios. Writes results/baselines.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adbq import model as M, samplers as S  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
BUDGETS = [2**k for k in range(8, 21)]          # 256 ... 1,048,576 paths
SCEN_BUDGET = 886_100                            # equal-budget comparison with MLAE
SCEN_REPS = 60


def reps(N):
    return int(np.clip(2**25 // N, 30, 400))


def main(seed=20261107):
    rng = np.random.default_rng(seed)
    P = S.Problem(M.SCENARIOS["adb-base"])
    res = {"budgets": BUDGETS, "targets": P.targets, "exact": P.exact.tolist(),
           "reps": [reps(N) for N in BUDGETS], "samplers": {}}
    for name, f in S.SAMPLERS.items():
        t0 = time.time()
        err = S.error_curve(P, f, BUDGETS, rng, reps)
        fits = [S.queries_to_error(BUDGETS, err[:, j]) for j in range(len(P.targets))]
        if name == "Monte Carlo":
            # Monte Carlo variance is known exactly from the exact term distribution
            d = M.term_distribution_convolution(P.sc)
            var = (P.F ** 2) @ d - P.exact ** 2
            exact_n = (var / (0.01 * P.exact) ** 2).tolist()
        res["samplers"][name] = {
            "rel_rmse": err.tolist(),
            "queries_to_1pct": [x[0] for x in fits],
            "slope": [x[1] for x in fits],
            "extrapolated": [x[2] for x in fits],
            "seconds": time.time() - t0,
        }
        if name == "Monte Carlo":
            res["samplers"][name]["queries_to_1pct_fit"] = res["samplers"][name]["queries_to_1pct"]
            res["samplers"][name]["queries_to_1pct"] = exact_n
        print(f"{name:20s} done in {time.time()-t0:.0f}s; slopes {np.round([x[1] for x in fits], 2)}")

    # equal-budget comparison across scenarios (Monte Carlo and importance sampling)
    scen = {}
    for sname, sc in M.SCENARIOS.items():
        Ps = S.Problem(sc)
        row = {}
        for name in ("Monte Carlo", "Importance sampling"):
            est = np.array([S.SAMPLERS[name](Ps, SCEN_BUDGET, rng) for _ in range(SCEN_REPS)])
            row[name] = (np.sqrt(np.mean((est - Ps.exact) ** 2, axis=0)) / np.abs(Ps.exact)).tolist()
        scen[sname] = row
    res["scenarios"] = {"budget": SCEN_BUDGET, "reps": SCEN_REPS, "rel_rmse": scen}
    (OUT / "baselines.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
