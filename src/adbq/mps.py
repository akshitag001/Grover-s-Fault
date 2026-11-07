"""Track B: exact matrix-product-state contraction for an n-country chain (quantum-inspired).

Countries 1..n sit on a chain; each neighbouring pair shares an earthquake process at rate
c_e = m * 0.5 * (0.075 L_i + 0.074 L_{i+1}) and a precipitation process at
share * min(mu_i, mu_{i+1}). Each country keeps its own total rates. At n = 2 this is exactly
the ADB model. Over the term a bond matters only through (min(N_eq, 4), rain fired) -> 10 states.

Also contains an independent brute-force reference (5^n CTMC + Poisson enumeration).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from . import model as M

UNIT = 2.5e6           # loss accounting unit (USD)


@dataclass(frozen=True)
class Chain:
    n: int
    m: float = 1.0
    rain_share: float = 0.25
    rain_mult: float = 1.0
    q_rain: float = 0.02
    eq_limit: float = 70e6
    rain_limit: float = 10e6
    principal: float = 80e6
    years: float = M.TERM_YEARS

    @staticmethod
    def from_scenario(sc: M.Scenario, n: int = 2) -> "Chain":
        return Chain(n=n, m=sc.m, rain_share=sc.rain_shared_share, rain_mult=sc.rain_mult,
                     q_rain=sc.q_rain, eq_limit=sc.eq_limit, rain_limit=sc.rain_limit,
                     principal=sc.principal)

    def tails(self):
        return [M.TAIL_KG if i % 2 == 0 else M.TAIL_TJ for i in range(self.n)]

    def rates(self):
        """Per-country (lambda_k vector, Lambda), shared eq rates c_e, rain rates."""
        rl = [M.compound_poisson_rates(t) for t in self.tails()]
        c = [self.m * 0.5 * (M.COND_TJ_GIVEN_KG * rl[i][1] + M.COND_KG_GIVEN_TJ * rl[i + 1][1])
             for i in range(self.n - 1)]
        mu = -np.log(1 - self.q_rain) * self.rain_mult
        cr = [self.rain_share * mu for _ in range(self.n - 1)]
        return rl, c, mu, cr

    def units(self):
        q = self.eq_limit / 4 / UNIT
        r = self.rain_limit / UNIT
        assert abs(q - round(q)) < 1e-9 and abs(r - round(r)) < 1e-9
        return int(round(q)), int(round(r)), int(round(self.principal / UNIT))


def _capped_compound(rates_k, t):
    """Distribution of min(total quarter steps, 4) of a compound Poisson over time t."""
    G = np.zeros((5, 5))
    for a in range(5):
        for k in range(4):
            G[a, min(a + k + 1, 4)] += rates_k[k]
    G[np.diag_indices(5)] = 0.0
    G[np.diag_indices(5)] = -G.sum(axis=1)
    return expm(t * G)[0]


def _cap_conv(p, q):
    out = np.zeros(5)
    for i in range(5):
        for j in range(5):
            out[min(i + j, 4)] += p[i] * q[j]
    return out


def _bond_prior(c, cr, t):
    """Pr(min(N,4) = j) x Pr(rain fired) over 10 bond states (j*2 + fired)."""
    lam = c * t
    pn = np.array([np.exp(-lam) * lam**j / np.prod(range(1, j + 1)) for j in range(4)])
    pn = np.append(pn, 1 - pn.sum())
    pr = np.array([np.exp(-cr * t), 1 - np.exp(-cr * t)])
    return np.kron(pn, pr)


def site_tensors(ch: Chain):
    """T[i][bl, br, x] with x = steps*2 + rain (10 outcomes); dummy bond of size 1 at the ends."""
    rl, c, mu, cr = ch.rates()
    t = ch.years
    T, cache = [], {}
    for i in range(ch.n):
        key = (i % 2, i > 0, i < ch.n - 1, c[i - 1] if i > 0 else None, c[i] if i < ch.n - 1 else None)
        if key in cache:
            T.append(cache[key])
            continue
        lam_k, Lam = rl[i]
        w = lam_k / Lam
        c_l = c[i - 1] if i > 0 else 0.0
        c_r = c[i] if i < ch.n - 1 else 0.0
        own_eq = _capped_compound(lam_k * (Lam - c_l - c_r) / Lam, t)
        mu_own = mu - (cr[i - 1] if i > 0 else 0.0) - (cr[i] if i < ch.n - 1 else 0.0)
        p_own_rain = 1 - np.exp(-mu_own * t)
        wcap = np.zeros(5)
        wcap[1:] = w                                  # size distribution of one shared event
        shared = [np.eye(5)[0]]
        for _ in range(4):
            shared.append(_cap_conv(shared[-1], wcap))   # sum of j shared events, capped
        nl = 10 if i > 0 else 1
        nr = 10 if i < ch.n - 1 else 1
        Ti = np.zeros((nl, nr, 10))
        for bl in range(nl):
            for br in range(nr):
                jl, fl = divmod(bl, 2) if i > 0 else (0, 0)
                jr, fr = divmod(br, 2) if i < ch.n - 1 else (0, 0)
                steps = _cap_conv(own_eq, shared[min(jl + jr, 4)])
                rain = np.array([0.0, 1.0]) if (fl or fr) else np.array([1 - p_own_rain, p_own_rain])
                Ti[bl, br] = np.kron(steps, rain)
        cache[key] = Ti
        T.append(Ti)
    priors = [_bond_prior(c[i], cr[i], t) for i in range(ch.n - 1)]
    return T, priors


def contract(ch: Chain, weight=None, track_loss=True, tensors=None):
    """Forward contraction. Returns the distribution of total loss in UNITs (len 32n+1),
    with outcomes optionally masked per country by weight[i][x] (e.g. indicators)."""
    T, priors = tensors if tensors is not None else site_tensors(ch)
    qu, ru, pu = ch.units()
    loss_x = np.array([(x // 2) * qu + (x % 2) * ru for x in range(10)])
    Lmax = pu * ch.n if track_loss else 0
    alpha = np.zeros((1, Lmax + 1))
    alpha[0, 0] = 1.0
    for i in range(ch.n):
        Ti = T[i] * (priors[i][None, :, None] if i < ch.n - 1 else 1.0)
        if weight is not None:
            Ti = Ti * weight[i][None, None, :]
        nr = Ti.shape[1]
        new = np.zeros((nr, Lmax + 1))
        for x in range(10):
            M_x = Ti[:, :, x]                          # (bl, br)
            if not M_x.any():
                continue
            contrib = M_x.T @ alpha                     # (br, L)
            sh = loss_x[x] if track_loss else 0
            if sh:
                new[:, sh:] += contrib[:, : Lmax + 1 - sh]
            else:
                new += contrib
        alpha = new
    return alpha[0]


def targets(ch: Chain) -> dict[str, float]:
    """n-country generalisation of the five focus targets (equal to Table 3 at n = 2)."""
    qu, ru, pu = ch.units()
    loss_x = np.array([(x // 2) * qu + (x % 2) * ru for x in range(10)])
    tens = site_tensors(ch)
    Ldist = contract(ch, tensors=tens)
    Lu = np.arange(len(Ldist))
    pays = [(loss_x > 0).astype(float)] * ch.n
    exhausted = [(loss_x >= pu).astype(float)] * ch.n
    return {
        "expected_loss_usd": float(Ldist @ Lu) * UNIT,
        "all_pay_probability": float(contract(ch, pays, track_loss=False, tensors=tens).sum()),
        "loss_gt_principal_probability": float(Ldist[Lu > pu].sum()),     # L > 80m (Table 3)
        "stop_loss_usd": float(Ldist @ np.maximum(Lu - pu, 0)) * UNIT,
        "all_exhausted_probability": float(contract(ch, exhausted, track_loss=False, tensors=tens).sum()),
        "loss_gt_half_aggregate_probability": float(Ldist[Lu > pu * ch.n / 2].sum()),
    }


# --------------------------------------------------------------------------- brute force
def brute_force_targets(ch: Chain) -> dict[str, float]:
    """Independent reference: CTMC on all 5^n joint earthquake states (own + shared events,
    shared sizes drawn independently per country) x enumeration of every rain process."""
    rl, c, mu, cr = ch.rates()
    n, t = ch.n, ch.years
    qu, ru, pu = ch.units()
    states = list(itertools.product(range(5), repeat=n))
    idx = {s: k for k, s in enumerate(states)}
    G = np.zeros((len(states), len(states)))
    for s in states:
        i0 = idx[s]
        for i in range(n):
            lam_k, Lam = rl[i]
            own = Lam - (c[i - 1] if i > 0 else 0) - (c[i] if i < n - 1 else 0)
            for k in range(4):
                s2 = list(s)
                s2[i] = min(s[i] + k + 1, 4)
                G[i0, idx[tuple(s2)]] += own * lam_k[k] / Lam
        for e in range(n - 1):
            w1 = rl[e][0] / rl[e][1]
            w2 = rl[e + 1][0] / rl[e + 1][1]
            for k in range(4):
                for j in range(4):
                    s2 = list(s)
                    s2[e] = min(s[e] + k + 1, 4)
                    s2[e + 1] = min(s[e + 1] + j + 1, 4)
                    G[i0, idx[tuple(s2)]] += c[e] * w1[k] * w2[j]
    G[np.diag_indices(len(states))] = 0.0
    G[np.diag_indices(len(states))] = -G.sum(axis=1)
    peq = expm(t * G)[0]
    # rain: n own processes + (n-1) shared processes, each fires or not
    own_mu = [mu - (cr[i - 1] if i > 0 else 0) - (cr[i] if i < n - 1 else 0) for i in range(n)]
    rain = {}
    for fires in itertools.product((0, 1), repeat=2 * n - 1):
        pr = 1.0
        for i in range(n):
            pf = 1 - np.exp(-own_mu[i] * t)
            pr *= pf if fires[i] else 1 - pf
        for e in range(n - 1):
            pf = 1 - np.exp(-cr[e] * t)
            pr *= pf if fires[n + e] else 1 - pf
        r = tuple(int(fires[i] or (i > 0 and fires[n + i - 1]) or (i < n - 1 and fires[n + i]))
                  for i in range(n))
        rain[r] = rain.get(r, 0.0) + pr
    out = dict.fromkeys(["expected_loss_usd", "all_pay_probability", "loss_gt_principal_probability",
                         "stop_loss_usd", "all_exhausted_probability",
                         "loss_gt_half_aggregate_probability"], 0.0)
    for s, pe in zip(states, peq):
        for r, prr in rain.items():
            p = pe * prr
            li = [s[i] * qu + r[i] * ru for i in range(n)]
            L = sum(li)
            out["expected_loss_usd"] += p * L * UNIT
            out["all_pay_probability"] += p * all(x > 0 for x in li)
            out["loss_gt_principal_probability"] += p * (L > pu)
            out["stop_loss_usd"] += p * max(L - pu, 0) * UNIT
            out["all_exhausted_probability"] += p * all(x >= pu for x in li)
            out["loss_gt_half_aggregate_probability"] += p * (L > pu * n / 2)
    return out
