"""Exact ADB two-country catastrophe-bond model (challenge brief, Section 3b).

Two exact routes are provided:
  * ``term_distribution_convolution`` - the brief's method (3 annual convolutions, capped)
  * ``term_distribution_exp3g``       - structural result 1: one exp(3G) Markov run
State order everywhere: (a, b, r1, r2) with a, b in 0..4 and r1, r2 in 0..1,
flattened as ((a*5 + b)*2 + r1)*2 + r2  -> 100 states.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

# Table 1: annual earthquake payout exceedance probabilities (>=25%, >=50%, >=75%, 100%)
TAIL_KG = (0.03883, 0.02331, 0.01790, 0.01184)
TAIL_TJ = (0.03925, 0.01743, 0.01332, 0.01043)
COND_TJ_GIVEN_KG = 0.075
COND_KG_GIVEN_TJ = 0.074
TERM_YEARS = 3
N_STATES = 100


@dataclass(frozen=True)
class Scenario:
    name: str = "adb-base"
    m: float = 1.0                 # shared-earthquake multiplier
    rain_shared_share: float = 0.25
    rain_mult: float = 1.0
    q_rain: float = 0.02
    eq_limit: float = 70e6
    rain_limit: float = 10e6
    principal: float = 80e6


SCENARIOS = {
    "adb-base": Scenario(),
    "adb-independent": Scenario(name="adb-independent", m=0.0, rain_shared_share=0.0),
    "adb-common-shock": Scenario(name="adb-common-shock", m=2.0),
    "adb-rain-stress": Scenario(name="adb-rain-stress", rain_mult=2.0),
    "adb-rain-cover": Scenario(name="adb-rain-cover", eq_limit=60e6, rain_limit=20e6),
}

# Table 5 exact reference values (three-year term), as published.
REFERENCE = {
    "adb-base":         (10.06e6, 0.04414, 0.01205, 282_355, 0.000032),
    "adb-independent":  (10.06e6, 0.02716, 0.00719, 171_266, 0.000004),
    "adb-common-shock": (10.06e6, 0.05053, 0.01607, 385_043, 0.000044),
    "adb-rain-stress":  (11.17e6, 0.07046, 0.01322, 306_286, 0.000069),
    "adb-rain-cover":   (9.97e6,  0.04414, 0.00700, 170_927, 0.000032),
}
TARGETS = (
    "term_expected_principal_loss_usd",
    "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability",
    "term_portfolio_stop_loss_usd",
    "term_both_bonds_exhausted_probability",
)


def compound_poisson_rates(tail):
    """Eqs (2)-(3): per-year rates lambda_1..4 and total Lambda from four tail probabilities."""
    t1, t2, t3, t4 = tail
    p0, p1, p2, p3 = 1 - t1, t1 - t2, t2 - t3, t3 - t4
    lam_tot = -np.log(p0)
    l1 = p1 / p0
    l2 = p2 / p0 - l1**2 / 2
    l3 = p3 / p0 - l1 * l2 - l1**3 / 6
    l4 = lam_tot - l1 - l2 - l3
    return np.array([l1, l2, l3, l4]), lam_tot


def shared_eq_rate(sc: Scenario, L1: float, L2: float) -> float:
    return sc.m * 0.5 * (COND_TJ_GIVEN_KG * L1 + COND_KG_GIVEN_TJ * L2)  # eq (4)


def eq_generator(sc: Scenario, tails=(TAIL_KG, TAIL_TJ)) -> np.ndarray:
    """25x25 CTMC generator on capped quarter steps (a, b)."""
    (r1, L1), (r2, L2) = (compound_poisson_rates(t) for t in tails)
    c = shared_eq_rate(sc, L1, L2)
    w1, w2 = r1 / L1, r2 / L2
    G = np.zeros((25, 25))
    for a in range(5):
        for b in range(5):
            s = a * 5 + b
            for k in range(4):
                G[s, min(a + k + 1, 4) * 5 + b] += (L1 - c) * w1[k]
                G[s, a * 5 + min(b + k + 1, 4)] += (L2 - c) * w2[k]
                for j in range(4):
                    G[s, min(a + k + 1, 4) * 5 + min(b + j + 1, 4)] += c * w1[k] * w2[j]
    G[np.diag_indices(25)] = 0.0          # capped self-transitions are no-ops
    G[np.diag_indices(25)] = -G.sum(axis=1)
    return G


def rain_probs(sc: Scenario, years: float = 1.0) -> np.ndarray:
    """Eqs (5)-(6): [P00, P01, P10, P11] indexed as r1*2 + r2, over `years` years."""
    mu = -np.log(1 - sc.q_rain) * sc.rain_mult
    cr = sc.rain_shared_share * mu  # both countries have the same rate
    mu1 = mu2 = mu
    P00 = np.exp(-years * (mu1 + mu2 - cr))
    P10 = np.exp(-years * mu2) - P00   # country 1 only
    P01 = np.exp(-years * mu1) - P00   # country 2 only
    P11 = 1 - P00 - P10 - P01
    out = np.zeros(4)
    out[0b00], out[0b01], out[0b10], out[0b11] = P00, P01, P10, P11
    return out


def annual_distribution(sc: Scenario) -> np.ndarray:
    eq = expm(eq_generator(sc))[0]
    return np.kron(eq, rain_probs(sc))  # state order (a,b) major, (r1,r2) minor


def _decode(s):
    ab, r = divmod(s, 4)
    a, b = divmod(ab, 5)
    return a, b, r >> 1, r & 1


_CAP_ADD = np.zeros((N_STATES, N_STATES), dtype=np.int64)
for _s in range(N_STATES):
    _a, _b, _x, _y = _decode(_s)
    for _d in range(N_STATES):
        _da, _db, _dx, _dy = _decode(_d)
        _CAP_ADD[_s, _d] = (((min(_a + _da, 4) * 5 + min(_b + _db, 4)) * 2
                             + min(_x + _dx, 1)) * 2 + min(_y + _dy, 1))


def capped_add_index(s, d):
    return _CAP_ADD[s, d]


def term_distribution_convolution(sc: Scenario) -> np.ndarray:
    """The brief's exact method: three passes over 100 x 100 state pairs."""
    ann = annual_distribution(sc)
    dist = np.zeros(N_STATES)
    dist[0] = 1.0
    for _ in range(TERM_YEARS):
        new = np.zeros(N_STATES)
        np.add.at(new, _CAP_ADD.ravel(), np.outer(dist, ann).ravel())
        dist = new
    return dist


def term_distribution_exp3g(sc: Scenario) -> np.ndarray:
    """Structural result 1: capped term state = first row of exp(3G) x 3-year rain law."""
    return np.kron(expm(TERM_YEARS * eq_generator(sc))[0], rain_probs(sc, TERM_YEARS))


def state_losses(sc: Scenario):
    """Per-state (L1, L2) principal losses in USD, eq (7)."""
    L1 = np.zeros(N_STATES)
    L2 = np.zeros(N_STATES)
    for s in range(N_STATES):
        a, b, x, y = _decode(s)
        L1[s] = sc.eq_limit * a / 4 + sc.rain_limit * x
        L2[s] = sc.eq_limit * b / 4 + sc.rain_limit * y
    return L1, L2


def payoffs(sc: Scenario) -> dict[str, np.ndarray]:
    """Per-state payoff f(x) for each of the five targets."""
    L1, L2 = state_losses(sc)
    L = L1 + L2
    T = sc.principal
    return {
        TARGETS[0]: L,
        TARGETS[1]: ((L1 > 0) & (L2 > 0)).astype(float),
        TARGETS[2]: (L > T).astype(float),
        TARGETS[3]: np.maximum(L - T, 0.0),
        TARGETS[4]: ((L1 >= T) & (L2 >= T)).astype(float),
    }


def payoff_max(sc: Scenario) -> dict[str, float]:
    """Normalisers f_max used for amplitude encoding."""
    return {TARGETS[0]: 2 * sc.principal, TARGETS[1]: 1.0, TARGETS[2]: 1.0,
            TARGETS[3]: sc.principal, TARGETS[4]: 1.0}


def exact_targets(sc: Scenario, dist: np.ndarray | None = None) -> dict[str, float]:
    if dist is None:
        dist = term_distribution_convolution(sc)
    return {k: float(dist @ f) for k, f in payoffs(sc).items()}


def scenario(name: str, **kw) -> Scenario:
    return replace(SCENARIOS[name], **kw)
