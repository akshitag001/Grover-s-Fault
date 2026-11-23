"""Figure 4: wall-clock crossover map from results/resources.json."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
R = json.load(open(ROOT / "results" / "resources.json"))

C = R["C_exhaustion"]
KAPPA_IS = R["kappa_is"]
TQ10 = R["oracle_time_10_s"]
EPS = 0.01


def a_star(tq, kappa, tc):
    return (kappa * tc / (C * EPS * tq)) ** 2


tq = np.logspace(-4, 1, 400)
fig, ax = plt.subplots(figsize=(8.2, 5.0))

ax.fill_between(tq, 1e-18, a_star(tq, 1.0, 1e-6), color="#2a6fdb", alpha=0.10, lw=0)
ax.loglog(tq, a_star(tq, 1.0, 1e-6), color="#2a6fdb", lw=2.4,
          label="vs Monte Carlo (1 $\\mu$s/path)")
ax.loglog(tq, a_star(tq, KAPPA_IS, 1e-6), color="#4a4a48", lw=2.0, ls="--",
          label="vs importance sampling (1 $\\mu$s/path)")

ax.axvline(TQ10, color="#9a9a96", lw=1.1, ls=":")
ax.annotate(f"modelled oracle,\n10 countries ({TQ10*1e3:.0f} ms)",
            xy=(TQ10, 4e-17), xytext=(TQ10 * 1.45, 4e-17),
            fontsize=8.5, color="#4a4a48", va="bottom")

for a, lab in [(3.163e-5, "both bonds exhausted, 2 countries ($a$ = 3.2e-05)"),
               (5.91e-6, "loss > half principal, 10 countries ($a$ = 5.9e-06)")]:
    ax.axhline(a, color="#bdbcb5", lw=1.0, ls="-")
    ax.annotate(lab, xy=(8.0, a * 1.6), fontsize=8.5, color="#6b6b68", ha="right")

ax.set_xlabel("time per quantum oracle call $t_q$ (s)")
ax.set_ylabel("crossover probability $a^{*}$  (QAE faster below the curve)")
ax.set_title("Where query advantage becomes wall-clock advantage ($\\varepsilon$ = 1%)",
             fontsize=11)
ax.set_xlim(1e-4, 10)
ax.set_ylim(1e-17, 1e-2)
ax.grid(True, which="major", color="#ebebe8", lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "fig4_crossover.png", dpi=200)
print("wrote", FIG / "fig4_crossover.png")
