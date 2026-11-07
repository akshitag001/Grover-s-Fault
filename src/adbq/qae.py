"""Quantum amplitude estimation for the ADB targets (Track A).

* ``build_A``         - explicit 256x256 unitary A on 7 state qubits + 1 objective qubit
* ``grover_probs``    - exact statevector Pr(objective=1) after k Grover steps
* ``mlae``            - Maximum-Likelihood Amplitude Estimation (Suzuki et al. 2020) with
                        binomial shot sampling from the verified Grover probabilities
* noise model         - global depolarising: visibility v(k) = (1-eps)^(G_A * (2k+1))

Oracle-call accounting: one shot at Grover power k costs 2k+1 applications of A.
"""
from __future__ import annotations

import numpy as np

from . import model as M

N_STATE_QUBITS = 7          # 128 >= 100 term states
DIM_STATE = 2 ** N_STATE_QUBITS
DIM = 2 * DIM_STATE         # + objective qubit (least significant)


def amplitude(sc: M.Scenario, target: str) -> float:
    p = M.term_distribution_exp3g(sc)
    return float(p @ (M.payoffs(sc)[target] / M.payoff_max(sc)[target]))


def build_A(sc: M.Scenario, target: str) -> np.ndarray:
    """A = R_obj (U_prep (x) I); U_prep is a Householder reflection with first column sqrt(p)."""
    p = np.zeros(DIM_STATE)
    p[: M.N_STATES] = M.term_distribution_exp3g(sc)
    p = np.clip(p, 0, None)
    p /= p.sum()
    v = np.sqrt(p)
    e0 = np.zeros(DIM_STATE)
    e0[0] = 1.0
    w = e0 - v
    U = np.eye(DIM_STATE) - 2 * np.outer(w, w) / (w @ w) if np.linalg.norm(w) > 1e-15 else np.eye(DIM_STATE)
    g = np.zeros(DIM_STATE)
    g[: M.N_STATES] = M.payoffs(sc)[target] / M.payoff_max(sc)[target]
    th = np.arcsin(np.sqrt(np.clip(g, 0, 1)))
    R = np.zeros((DIM, DIM))
    for x in range(DIM_STATE):                   # controlled Ry(2 th_x) on objective qubit
        c, s = np.cos(th[x]), np.sin(th[x])
        R[2 * x:2 * x + 2, 2 * x:2 * x + 2] = [[c, -s], [s, c]]
    return R @ np.kron(U, np.eye(2))


def grover_probs(A: np.ndarray, ks) -> np.ndarray:
    """Exact Pr(objective = 1) after Q^k A|0>, Q = -A S_0 A^dag S_chi."""
    good = np.zeros(DIM)
    good[1::2] = 1.0
    psi = A[:, 0].copy()
    out, k_now = {}, 0
    for k in sorted(set(int(k) for k in ks)):
        while k_now < k:
            psi = psi * (1 - 2 * good)                    # S_chi
            phi = A.T @ psi                               # A^dag (A is real orthogonal)
            phi[0] = -phi[0]                              # S_0 = I - 2|0><0|
            psi = -(A @ phi)
            k_now += 1
        out[k] = float(psi[1::2] @ psi[1::2])
    return np.array([out[int(k)] for k in ks])


# --------------------------------------------------------------------------- schedules
def schedule(kind: str, n_powers: int) -> np.ndarray:
    if kind == "doubling":
        ks = [0] + [2 ** j for j in range(n_powers - 1)]
    elif kind.startswith("ratio"):
        r = float(kind[5:]) if len(kind) > 5 else 1.5
        ks, j = [0], 0
        while len(ks) < n_powers:
            k = int(np.floor(r ** j))
            if k > ks[-1]:
                ks.append(k)
            j += 1
    else:
        raise ValueError(kind)
    return np.array(ks[:n_powers])


def calls(ks, shots) -> int:
    return int(np.sum(np.asarray(shots) * (2 * np.asarray(ks) + 1)))


# --------------------------------------------------------------------------- estimator
def _loglik(theta, ks, hits, shots, vis):
    p = np.sin((2 * ks[:, None] + 1) * theta[None, :]) ** 2
    p = 0.5 + vis[:, None] * (p - 0.5)
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return (hits[:, None] * np.log(p) + (shots - hits)[:, None] * np.log1p(-p)).sum(axis=0)


def mle_theta(ks, hits, shots, vis=None, grid0=20000, grid=400):
    """Sequential windowed MLE: each new power refines the window of the previous estimate."""
    ks = np.asarray(ks, float)
    hits = np.asarray(hits, float)
    shots = np.asarray(shots, float)
    vis = np.ones_like(ks) if vis is None else np.asarray(vis, float)
    lo, hi = 0.0, np.pi / 2
    th = None
    for j in range(len(ks)):
        n = grid0 if j == 0 else grid
        tg = np.linspace(lo, hi, n)
        ll = _loglik(tg, ks[: j + 1], hits[: j + 1], shots[: j + 1], vis[: j + 1])
        i = int(np.argmax(ll))
        th = tg[i]
        # next window: a few periods of the next power's oscillation around the estimate
        k_next = ks[j + 1] if j + 1 < len(ks) else ks[j]
        half = 1.5 * np.pi / (2 * (2 * k_next + 1))
        lo, hi = max(0.0, th - half), min(np.pi / 2, th + half)
    # final local refinement
    tg = np.linspace(max(0, th - 2 * (hi - lo) / grid), min(np.pi / 2, th + 2 * (hi - lo) / grid), 201)
    th = tg[np.argmax(_loglik(tg, ks, hits, shots, vis))]
    return float(th)


def mlae(a_true: float, ks, shots, rng, eps2q: float = 0.0, gates_per_call: int = 420,
         probs=None):
    """One MLAE run. Returns (a_hat, oracle_calls)."""
    ks = np.asarray(ks)
    shots = np.broadcast_to(np.asarray(shots), ks.shape).astype(int)
    theta = np.arcsin(np.sqrt(a_true))
    p = np.sin((2 * ks + 1) * theta) ** 2 if probs is None else np.asarray(probs)
    vis = visibility(ks, eps2q, gates_per_call)
    p_noisy = 0.5 + vis * (p - 0.5)
    hits = rng.binomial(shots, p_noisy)
    th = mle_theta(ks, hits, shots, vis)
    return np.sin(th) ** 2, calls(ks, shots)


def visibility(ks, eps2q, gates_per_call=420):
    return (1.0 - eps2q) ** (gates_per_call * (2 * np.asarray(ks, float) + 1))


def usable_depth(eps2q, gates_per_call=420, v_min=0.05):
    if eps2q == 0:
        return np.inf
    return int(np.floor((np.log(v_min) / (gates_per_call * np.log1p(-eps2q)) - 1) / 2))
