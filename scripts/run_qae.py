"""Track A experiments: Grover-probability check, MLAE error curves (doubling vs ratio-1.5),
equal-budget scenario comparison, and the depolarising-noise study. Writes results/qae.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adbq import model as M, qae as Q, samplers as S  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
SHOTS = 100
REPS = 200
N_POWERS = list(range(8, 21))            # ratio-1.5 schedules, 2.9e3 ... 1.3e6 oracle calls
SCEN_POWERS = 19                         # 886,100 oracle calls
NOISE_BUDGET = 1_000_000
NOISE_RATES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]


def rel_rmse(est, a):
    return float(np.sqrt(np.mean((np.asarray(est) / a - 1) ** 2)))


def budget_schedule(kcap, budget):
    """Ratio-1.5 powers up to kcap that fit the budget; leftover shots go to the top power."""
    ks = []
    for k in Q.schedule("ratio1.5", 40):
        if k > kcap or Q.calls(ks + [k], [SHOTS] * (len(ks) + 1)) > budget:
            break
        ks.append(int(k))
    shots = [SHOTS] * len(ks)
    left = budget - Q.calls(ks, shots)
    shots[-1] += left // (2 * ks[-1] + 1)
    return np.array(ks), np.array(shots)


def main(seed=20261108):
    rng = np.random.default_rng(seed)
    sc = M.SCENARIOS["adb-base"]
    res = {"targets": list(M.TARGETS), "shots_per_power": SHOTS, "reps": REPS}

    # 1. statevector check of Grover probabilities
    ks = np.arange(0, 64)
    chk = {}
    for t in M.TARGETS:
        a = Q.amplitude(sc, t)
        chk[t] = float(np.max(np.abs(Q.grover_probs(Q.build_A(sc, t), ks)
                                     - np.sin((2 * ks + 1) * np.arcsin(np.sqrt(a))) ** 2)))
    res["grover_check_max_abs_diff"] = chk
    print("Grover check max diff:", max(chk.values()))

    # 2. MLAE error curves, ratio-1.5 schedule, all targets
    curves = {}
    for t in M.TARGETS:
        a = Q.amplitude(sc, t)
        calls, errs, outl = [], [], []
        for n in N_POWERS:
            k = Q.schedule("ratio1.5", n)
            est = [Q.mlae(a, k, SHOTS, rng)[0] for _ in range(REPS)]
            calls.append(Q.calls(k, [SHOTS] * n))
            errs.append(rel_rmse(est, a))
            outl.append(float(np.mean(np.abs(np.asarray(est) / a - 1) > 0.2)))
        n1, slope, extrap = S.queries_to_error(calls, errs)
        curves[t] = {"a": a, "calls": calls, "rel_rmse": errs, "outlier_rate_gt20pct": outl,
                     "queries_to_1pct": n1, "slope": slope, "extrapolated": extrap,
                     "constant_C": n1 * 0.01 * np.sqrt(a)}
        print(f"{t:45s} a={a:.2e} calls@1%={n1:.2e} slope={slope:.2f} C={n1*0.01*np.sqrt(a):.1f}")
    res["mlae_ratio1.5"] = curves

    # 3. doubling schedule on the rarest target (aliasing failure mode)
    a = Q.amplitude(sc, M.TARGETS[4])
    dbl = {}
    for n in (12, 14, 16):
        k = Q.schedule("doubling", n)
        est = np.array([Q.mlae(a, k, SHOTS, rng)[0] for _ in range(REPS)])
        dbl[str(n)] = {"calls": Q.calls(k, [SHOTS] * n), "rel_rmse": rel_rmse(est, a),
                       "outlier_rate_gt20pct": float(np.mean(np.abs(est / a - 1) > 0.2)),
                       "median_outlier_bias": float(np.median(est[np.abs(est / a - 1) > 0.2] / a - 1))
                       if np.any(np.abs(est / a - 1) > 0.2) else 0.0}
    res["mlae_doubling_exhaustion"] = dbl
    print("doubling:", dbl)

    # 4. equal budget across scenarios
    k = Q.schedule("ratio1.5", SCEN_POWERS)
    scen = {"calls": Q.calls(k, [SHOTS] * SCEN_POWERS), "rel_rmse": {}}
    for sname, s in M.SCENARIOS.items():
        row = []
        for t in M.TARGETS:
            a_s = Q.amplitude(s, t)
            row.append(rel_rmse([Q.mlae(a_s, k, SHOTS, rng)[0] for _ in range(REPS)], a_s))
        scen["rel_rmse"][sname] = row
    res["scenarios"] = scen
    print("scenarios:", {s: np.round(v, 4).tolist() for s, v in scen["rel_rmse"].items()})

    # 5. noise study on joint exhaustion at 1e6 oracle calls
    a = Q.amplitude(sc, M.TARGETS[4])
    noise = []
    for eps in NOISE_RATES:
        kcap = Q.usable_depth(eps)
        ks_n, shots_n = budget_schedule(kcap, NOISE_BUDGET)
        est = [Q.mlae(a, ks_n, shots_n, rng, eps2q=eps)[0] for _ in range(REPS)]
        noise.append({"eps2q": eps, "usable_depth": None if np.isinf(kcap) else int(kcap),
                      "max_power_used": int(ks_n[-1]), "calls": Q.calls(ks_n, shots_n),
                      "rel_rmse": rel_rmse(est, a)})
        print("noise", noise[-1])
    res["noise_exhaustion_1e6"] = noise
    (OUT / "qae.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
