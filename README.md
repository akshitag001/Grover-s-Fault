# Grover's Fault

**Team TuringIRL — ADB Quantum Innovation Challenge, Phase 1 prototype**

Quantum and tensor-network joint-loss modelling for the Kyrgyz Republic–Tajikistan
Disaster Relief Bond benchmark (ADB Quantum Innovation Challenge brief, Section 3).

## Reproduce everything

```bash
pip install -r requirements.lock
./run_all.sh          # ~4 minutes on one CPU core
```

`run_all.sh` runs the tests, every experiment, the figures, and writes
`results/SUMMARY.md` with every headline number used in the proposal.

## Layout

| Path | What it does |
| --- | --- |
| `src/adbq/model.py` | Exact ADB model: compound Poisson rates, 25-state CTMC, rain law, the brief's 3-year convolution, and the exp(3G) shortcut (structural result 1) |
| `src/adbq/samplers.py` | The brief's four baselines: Monte Carlo, scrambled Sobol (8 scrambles), first-payout-year stratification, half-loss-tilted importance sampling |
| `src/adbq/qae.py` | Track A: explicit 8-qubit unitary A, statevector Grover check, MLAE with doubling / ratio-1.5 schedules, depolarising-noise model |
| `src/adbq/mps.py` | Track B: exact MPS contraction for n-country chains, plus an independent 5^n CTMC brute-force reference |
| `src/adbq/resources.py` | Analytic logical counts for the structured circuit, fault-tolerant footprint and break-even model (all assumptions listed in the module docstring) |
| `scripts/` | One script per experiment; results go to `results/*.json`, figures to `figures/` |
| `tests/` | All 25 Table 5 values, exp(3G) = convolution, Grover probabilities, MPS vs brute force, sampler sanity |

## What is and is not simulated

* QAE runs use exact Grover probabilities (verified against the explicit statevector to 1e-14) with
  binomial shot noise. The 2-country circuit loads the exact term distribution (ideal loading).
* The structured n-country circuit is costed analytically, not compiled or simulated end to end.
* Noise is a global depolarising model: visibility (1 - eps)^(420 (2k + 1)).
* n-country portfolios are synthetic chains built from the Kyrgyz and Tajik tail tables.

## AI use

Code drafted with Claude (Anthropic) under the team's direction; all results come from running this code.
