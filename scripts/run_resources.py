"""Resource tables (Section 6) from analytic counts + measured MLAE constant and sampler efficiency."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adbq import mps, resources as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    qae = json.loads((ROOT / "results/qae.json").read_text())
    base = json.loads((ROOT / "results/baselines.json").read_text())
    exh = "term_both_bonds_exhausted_probability"
    C = qae["mlae_ratio1.5"][exh]["constant_C"]
    j = base["targets"].index(exh)
    n_mc = base["samplers"]["Monte Carlo"]["queries_to_1pct"][j]
    n_is = base["samplers"]["Importance sampling"]["queries_to_1pct"][j]
    kappa_is = n_is / n_mc
    res = {"C_exhaustion": C, "kappa_is": kappa_is, "t_per_rotation": R.T_PER_ROT,
           "demo": R.demo_circuit_counts(),
           "logical": [R.logical_counts(n) for n in (2, 5, 10, 20, 50, 100)]}
    rows = [(2, 3.163e-5, "both exhausted")]
    for n in (5, 10, 20):
        a = mps.targets(mps.Chain(n))["loss_gt_half_aggregate_probability"]
        rows.append((n, a, "loss > half aggregate principal"))
    res["footprint"] = [{**R.ft_footprint(n, a, C=C), "target": lbl} for n, a, lbl in rows]
    tq = R.oracle_time_s(10)
    res["oracle_time_10_s"] = tq
    res["break_even"] = {f"{name}@{tc*1e6:.0f}us": R.break_even(tq, tc, k, C=C)
                         for name, k in (("Monte Carlo", 1.0), ("Importance sampling", kappa_is))
                         for tc in (1e-6, 1e-5)}
    (ROOT / "results/resources.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
