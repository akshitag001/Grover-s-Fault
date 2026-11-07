"""Figures from results/*.json -> figures/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
LABELS = ["Expected loss", "Joint attachment", "Loss > USD 80m", "Stop-loss above USD 80m",
          "Both bonds exhausted"]
STYLE = {"Monte Carlo": dict(marker="o", mfc="white", mec="#9a9a96", ms=9),
         "Scrambled Sobol": dict(marker="s", color="#bdbcb5", ms=8),
         "Stratified": dict(marker="^", color="#bdbcb5", ms=9),
         "Importance sampling": dict(marker="D", color="#4a4a48", ms=8),
         "MLAE (quantum)": dict(marker="o", color="#2a6fdb", ms=11)}


def queries_figure(base, qae):
    tg = base["targets"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    maxb = base["budgets"][-1]
    for i, t in enumerate(tg):
        y = len(tg) - 1 - i
        vals = {n: s["queries_to_1pct"][i] for n, s in base["samplers"].items()}
        best_name = min(vals, key=vals.get)
        q = qae["mlae_ratio1.5"][t]["queries_to_1pct"]
        ax.plot([q, vals[best_name]], [y, y], color="#c9dbf7", lw=4, zorder=1)
        for n, v in vals.items():
            st = dict(STYLE[n])
            extrap = v > maxb and n != "Monte Carlo"
            if extrap:
                st.update(alpha=0.45)
            ax.plot(v, y, ls="none", zorder=2, label=n if i == 0 else None, **st)
        ax.plot(q, y, ls="none", zorder=3, label="MLAE (quantum)" if i == 0 else None,
                **STYLE["MLAE (quantum)"])
        r = vals[best_name] / q
        txt = f"{r:.1f}x fewer" if r >= 1 else f"{1/r:.1f}x more"
        ax.text(3e9, y, txt, va="center", fontsize=11, weight="bold" if r >= 1 else None)
        a = qae["mlae_ratio1.5"][t]["a"]
        ax.text(1.2e2, y, f"{LABELS[i]}\na = {a:.1e}", va="center", fontsize=10)
    ax.axvline(maxb, color="#999", ls=":", lw=1)
    ax.text(maxb * 1.1, len(tg) - 0.45, "largest classical budget run", fontsize=8, color="#666")
    ax.set_xscale("log")
    ax.set_xlim(1e2, 1e10)
    ax.set_ylim(-0.6, len(tg) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    ax.set_xlabel("queries (paths or oracle calls) to 1% relative RMSE, base scenario (fewer is better)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=5, frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "Faded classical markers: extrapolated beyond 1,048,576 paths along the fitted "
             "error slope. Monte Carlo: exact variance from the exact term distribution.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig1_queries_to_1pct.png", dpi=200)
    plt.close(fig)


def curves_figure(base, qae):
    tg = base["targets"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharey=True)
    for i, t in enumerate(tg):
        ax = axes[i]
        for n, s in base["samplers"].items():
            ax.loglog(base["budgets"], np.array(s["rel_rmse"])[:, i], "-", marker=STYLE[n]["marker"],
                      ms=4, label=n, color=STYLE[n].get("color", "#9a9a96"))
        m = qae["mlae_ratio1.5"][t]
        ax.loglog(m["calls"], m["rel_rmse"], "-o", ms=4, color="#2a6fdb", label="MLAE")
        ax.axhline(0.01, color="#999", ls=":", lw=1)
        ax.set_title(LABELS[i], fontsize=10)
        ax.set_xlabel("queries")
    axes[0].set_ylabel("relative RMSE")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_error_curves.png", dpi=160)
    plt.close(fig)


def mps_figure(m):
    n = [r["n"] for r in m["scaling"]]
    t = [r["seconds"] for r in m["scaling"]]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.loglog(n, t, "-o", color="#2a6fdb")
    ax.set_xlabel("countries in chain")
    ax.set_ylabel("exact contraction time (s)")
    ax.set_title("Tensor-network engine, one CPU core", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_mps_scaling.png", dpi=160)
    plt.close(fig)


def main():
    base = json.loads((ROOT / "results/baselines.json").read_text())
    qae = json.loads((ROOT / "results/qae.json").read_text())
    queries_figure(base, qae)
    curves_figure(base, qae)
    mps_figure(json.loads((ROOT / "results/mps.json").read_text()))


if __name__ == "__main__":
    main()
