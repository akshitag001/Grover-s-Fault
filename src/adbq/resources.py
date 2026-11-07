"""Analytic logical resource counts for the structured n-country QAE circuit (Section 3.2)
and the fault-tolerant footprint / break-even model (Section 6).

All counts are per oracle call A, for an indicator target (probability). Assumptions are
explicit constants below; Phase 2 replaces them with compiled counts.

Register layout (logical qubits):
  per bond:     3 (min(N_eq, 4)) + 1 (shared rain)          = 4
  per country:  3 (capped quarter steps) + 1 (rain)         = 4
  shared:       3 (reused m ancilla) + B (loss register, B = ceil(log2(32n+1)))
                + B (adder / comparator workspace) + 1 (flag) + 1 (objective)
Gate model (Clifford+T):
  * state preparation by uniformly controlled Ry rotations (Mottonen et al. 2004):
      bond: 7 + 1 rotations; country: 7 x 8 multiplexed (on m) + 1 rotations
  * one objective rotation (indicator)
  * additions with Gidney's temporary-AND adder: B-1 ANDs per B-bit addition, 4 T each,
    uncomputation by measurement (0 T)
  * per country: m = min(N_l + N_r, 4) (5 ANDs), loss += 7*steps as 8s - s (2 additions),
    loss += 4*rain as a controlled addition (counted as 2 additions)
  * final comparator (B ANDs) and an n-input AND for flags (n-1 ANDs)
  * rotation synthesis: repeat-until-success circuits, 1.15 log2(1/delta) + 9.2 T
    (Bocharov, Roetteler and Svore 2015), delta = 1e-9 -> 43.6 T per rotation
"""
from __future__ import annotations

import math

T_PER_AND = 4
DELTA = 1e-9
T_PER_ROT = 1.15 * math.log2(1 / DELTA) + 9.2

# fault-tolerant cost model
P_PHYS = 1e-3
P_TH = 1e-2
CYCLE_S = 1e-6
N_FACTORIES = 20
FACTORY_QUBITS = 6000
T_STATE_PERIOD_S = 50e-6
ROUTING_OVERHEAD = 1.5
FAILURE_BUDGET = 0.01


def loss_bits(n: int) -> int:
    return math.ceil(math.log2(32 * n + 1))


def logical_counts(n: int) -> dict:
    B = loss_bits(n)
    qubits = 4 * (n - 1) + 4 * n + 3 + 2 * B + 2
    rotations = 8 * (n - 1) + (7 * 8 + 1) * n + 1
    ands = n * (5 + 4 * (B - 1)) + B + (n - 1)
    t = T_PER_AND * ands + T_PER_ROT * rotations
    return {"n": n, "loss_bits": B, "logical_qubits": qubits, "rotations": rotations,
            "toffoli_ands": ands, "t_gates": t, "rotation_t_share": T_PER_ROT * rotations / t}


def demo_circuit_counts() -> dict:
    """2-country direct-loading demo: 7 state qubits + objective (+1 reflection ancilla)."""
    prep_cnot = 2 ** 7 - 2            # uniformly controlled Ry state preparation on 7 qubits
    obj_cnot = 2 ** 7                 # 7-control uniformly controlled Ry on the objective
    a_cnot = prep_cnot + obj_cnot
    reflection_cnot = 36              # 8-control Z with 1 ancilla (Barenco-style, ~6(k-2) + ...)
    return {"qubits": 8, "ancilla": 1, "two_qubit_per_A": a_cnot,
            "two_qubit_per_A_routed": round(a_cnot * 1.65),
            "two_qubit_per_Q": 2 * a_cnot + reflection_cnot}


def oracle_time_s(n: int) -> float:
    return logical_counts(n)["t_gates"] * T_STATE_PERIOD_S / N_FACTORIES


def ft_footprint(n: int, a: float, eps: float = 0.01, C: float = 26.0) -> dict:
    """One eps-accurate estimate of probability a with oracle calls = C / (eps sqrt(a))."""
    lc = logical_counts(n)
    calls = C / (eps * math.sqrt(a))
    total_t = calls * lc["t_gates"]
    runtime = total_t * T_STATE_PERIOD_S / N_FACTORIES
    cycles = runtime / CYCLE_S
    q = lc["logical_qubits"] * ROUTING_OVERHEAD
    d = 3
    while q * cycles * 0.1 * (P_PHYS / P_TH) ** ((d + 1) / 2) > FAILURE_BUDGET:
        d += 2
    phys = 2 * d * d * q + N_FACTORIES * FACTORY_QUBITS
    return {"n": n, "a": a, "oracle_calls": calls, "total_t": total_t, "code_distance": d,
            "physical_qubits": phys, "runtime_s": runtime}


def break_even(t_q: float, t_c: float, kappa: float, eps: float = 0.01, C: float = 26.0) -> float:
    """Probability a* below which QAE beats a sampler on wall-clock: (kappa t_c / (C eps t_q))^2."""
    return (kappa * t_c / (C * eps * t_q)) ** 2
