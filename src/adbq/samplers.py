"""The four sampling baselines of the brief, step 9.

A path is three annual states; the term state is their capped sum.
Every estimator returns one estimate per target for a given budget N (paths).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from . import model as M

YEARS = M.TERM_YEARS


class Problem:
    """Precomputed annual law and term payoffs for one scenario."""

    def __init__(self, sc: M.Scenario):
        self.sc = sc
        self.p = M.annual_distribution(sc)
        self.p = self.p / self.p.sum()
        self.cdf = np.cumsum(self.p)
        self.cdf[-1] = 1.0
        pay = M.payoffs(sc)
        self.targets = list(pay)
        self.F = np.stack([pay[t] for t in self.targets])          # (5, 100) term payoffs
        self.exact = self.F @ M.term_distribution_convolution(sc)
        L1, L2 = M.state_losses(sc)
        self.annual_loss = L1 + L2                                   # annual-state principal loss
        self.p0 = self.p[0]

    def inv(self, u):
        return np.minimum(np.searchsorted(self.cdf, u, side="right"), M.N_STATES - 1)

    def term_state(self, s):                                        # s: (N, 3) annual states
        t = s[:, 0]
        for y in range(1, YEARS):
            t = M._CAP_ADD[t, s[:, y]]
        return t


def monte_carlo(P: Problem, N: int, rng) -> np.ndarray:
    s = P.inv(rng.random((N, YEARS)))
    return P.F[:, P.term_state(s)].mean(axis=1)


def sobol(P: Problem, N: int, rng, n_scrambles: int = 8) -> np.ndarray:
    """Eight independent Owen scrambles, averaged (N paths in total)."""
    n = max(N // n_scrambles, 2)
    m = int(np.round(np.log2(n)))
    ests = []
    for _ in range(n_scrambles):
        u = qmc.Sobol(d=YEARS, scramble=True, seed=rng).random_base2(m)
        ests.append(P.F[:, P.term_state(P.inv(u))].mean(axis=1))
    return np.mean(ests, axis=0)


def sobol_interval(P: Problem, N: int, rng, n_scrambles: int = 8):
    """Mean and half-width of the t-interval on n_scrambles-1 dof (95%)."""
    from scipy.stats import t as tdist
    n = max(N // n_scrambles, 2)
    m = int(np.round(np.log2(n)))
    ests = np.array([P.F[:, P.term_state(P.inv(qmc.Sobol(d=YEARS, scramble=True, seed=rng).random_base2(m)))].mean(axis=1)
                     for _ in range(n_scrambles)])
    hw = tdist.ppf(0.975, n_scrambles - 1) * ests.std(axis=0, ddof=1) / np.sqrt(n_scrambles)
    return ests.mean(axis=0), hw


def stratified(P: Problem, N: int, rng) -> np.ndarray:
    """Strata = first year with any payout (1, 2, 3); all-zero paths contribute 0."""
    p_pay = P.p.copy()
    p_pay[0] = 0.0
    p_pay /= p_pay.sum()
    cdf_pay = np.cumsum(p_pay)
    cdf_pay[-1] = 1.0
    n_k = max(N // YEARS, 1)
    total = np.zeros(P.F.shape[0])
    for k in range(1, YEARS + 1):
        s = np.zeros((n_k, YEARS), dtype=np.int64)
        s[:, k - 1] = np.minimum(np.searchsorted(cdf_pay, rng.random(n_k), side="right"), M.N_STATES - 1)
        for y in range(k, YEARS):
            s[:, y] = P.inv(rng.random(n_k))
        w = P.p0 ** (k - 1) * (1 - P.p0)
        total += w * P.F[:, P.term_state(s)].mean(axis=1)
    return total


def importance(P: Problem, N: int, rng) -> np.ndarray:
    """Proposal per year = 0.5 * p + 0.5 * (p tilted by annual principal loss)."""
    tilt = P.p * P.annual_loss
    tilt /= tilt.sum()
    q = 0.5 * P.p + 0.5 * tilt
    cdf_q = np.cumsum(q)
    cdf_q[-1] = 1.0
    s = np.minimum(np.searchsorted(cdf_q, rng.random((N, YEARS)), side="right"), M.N_STATES - 1)
    w = np.prod(P.p[s] / q[s], axis=1)
    return (P.F[:, P.term_state(s)] * w).mean(axis=1)


SAMPLERS = {
    "Monte Carlo": monte_carlo,
    "Scrambled Sobol": sobol,
    "Stratified": stratified,
    "Importance sampling": importance,
}


def error_curve(P: Problem, sampler, budgets, rng, reps_fn):
    """Relative RMSE per target for each budget; returns array (len(budgets), 5)."""
    out = []
    for N in budgets:
        R = reps_fn(N)
        est = np.array([sampler(P, N, rng) for _ in range(R)])
        out.append(np.sqrt(np.mean((est - P.exact) ** 2, axis=0)) / np.abs(P.exact))
    return np.array(out)


def queries_to_error(budgets, rel_rmse, eps=0.01, fit_last=5):
    """Fit log(err) = a + s log(N) on the largest `fit_last` budgets; solve for err = eps.

    Returns (N_at_eps, slope, extrapolated?)."""
    b = np.log(np.asarray(budgets, float)[-fit_last:])
    e = np.log(np.asarray(rel_rmse, float)[-fit_last:])
    s, a = np.polyfit(b, e, 1)
    n = float(np.exp((np.log(eps) - a) / s))
    return n, float(s), bool(n > budgets[-1])
