# TuringIRL Phase 1 results summary

## Queries to 1% relative RMSE (base scenario)

| Target | a | Monte Carlo | Scrambled Sobol | Stratified | Importance sampling | MLAE | best classical / MLAE |
|---|---|---|---|---|---|---|---|
| term_expected_principal_loss_usd | 6.29e-02 | 4.60e+04 | 2.48e+03 | 7.43e+03 | 9.97e+03 | 6.30e+03 | 0.39 |
| term_joint_attachment_probability | 4.41e-02 | 2.17e+05 | 3.57e+04 | 6.01e+04 | 1.07e+05 | 1.56e+04 | 2.29 |
| term_portfolio_loss_exceedance_probability | 1.20e-02 | 8.20e+05 | 1.06e+05 | 2.44e+05 | 1.13e+05 | 2.56e+04 | 4.16 |
| term_portfolio_stop_loss_usd | 3.53e-03 | 1.43e+06 | 1.09e+05 | 4.47e+05 | 1.46e+05 | 7.32e+04 | 1.49 |
| term_both_bonds_exhausted_probability | 3.16e-05 | 3.16e+08 | 3.26e+08 | 4.16e+07 | 1.55e+07 | 4.59e+05 | 33.71 |

MLAE fitted slopes: -0.92, -1.08, -1.06, -1.27, -1.04
MLAE constant C = calls x eps x sqrt(a): 15.8, 32.8, 28.1, 43.5, 25.8
Monte Carlo slopes: -0.49, -0.44, -0.48, -0.54, -0.47; rel RMSE at 1,048,576 paths on exhaustion: 18.54%
Scrambled Sobol slopes: -0.86, -0.83, -0.83, -0.74, -0.50; rel RMSE at 1,048,576 paths on exhaustion: 16.95%
Stratified slopes: -0.55, -0.56, -0.49, -0.49, -0.57; rel RMSE at 1,048,576 paths on exhaustion: 7.47%
Importance sampling slopes: -0.49, -0.55, -0.51, -0.55, -0.48; rel RMSE at 1,048,576 paths on exhaustion: 3.73%

## Doubling schedule on exhaustion (aliasing)

- 12 powers, 410,600 calls: rel RMSE 6.2%, runs >20% off 2.5%, median outlier bias +39%
- 14 powers, 1,639,600 calls: rel RMSE 10.1%, runs >20% off 6.5%, median outlier bias +40%
- 16 powers, 6,555,000 calls: rel RMSE 10.5%, runs >20% off 7.0%, median outlier bias +40%
- ratio-1.5: runs >20% off at the three largest budgets: [0.0, 0.0, 0.0]

## Equal budget across scenarios (886,100 queries)

| Scenario | Exh MLAE | Exh IS | Exh MC | Stop-loss MLAE | Stop-loss IS |
|---|---|---|---|---|---|
| adb-base | 0.44% | 3.8% | 20% | 0.044% | 0.43% |
| adb-independent | 1.24% | 6.3% | 52% | 0.102% | 0.43% |
| adb-common-shock | 0.39% | 3.6% | 14% | 0.038% | 0.40% |
| adb-rain-stress | 0.31% | 2.7% | 13% | 0.039% | 0.38% |
| adb-rain-cover | 0.42% | 2.8% | 20% | 0.051% | 0.44% |

## Noise (joint exhaustion, ~1e6 oracle calls)

| eps 2q | usable depth | max power used | rel RMSE |
|---|---|---|---|
| 0 | None | 1477 | 0.4% |
| 1e-06 | 3565 | 1477 | 1.4% |
| 1e-05 | 356 | 291 | 11.7% |
| 0.0001 | 35 | 25 | 42.1% |
| 0.001 | 3 | 3 | 1204.8% |

## Tensor network

Table 5 max rel diff: adb-base 3.9e-14, adb-independent 3.8e-14, adb-common-shock 4.4e-15, adb-rain-stress 5.3e-14, adb-rain-cover 3.9e-14
- n=2: brute 0.001s, MPS 0.8 ms, max rel diff 4.6e-14
- n=3: brute 0.006s, MPS 3.0 ms, max rel diff 3.0e-14
- n=4: brute 0.114s, MPS 4.6 ms, max rel diff 3.2e-14
- n=5: brute 6.205s, MPS 5.3 ms, max rel diff 5.5e-14
- n=2: 0.8 ms; all pay 4.41e-02; loss > half aggregate 1.20e-02
- n=5: 5.4 ms; all pay 5.29e-04; loss > half aggregate 1.01e-03
- n=10: 6.6 ms; all pay 3.58e-07; loss > half aggregate 5.91e-06
- n=20: 9.0 ms; all pay 1.62e-13; loss > half aggregate 2.75e-10
- n=50: 24.7 ms; all pay 1.51e-32; loss > half aggregate 7.78e-23
- n=100: 46.3 ms; all pay 2.88e-64; loss > half aggregate 1.40e-43
- n=200: 169.9 ms; all pay 1.05e-127; loss > half aggregate 6.21e-85
- n=500: 1780.5 ms; all pay 5.11e-318; loss > half aggregate 9.65e-209

## Resources

Demo circuit: {'qubits': 8, 'ancilla': 1, 'two_qubit_per_A': 254, 'two_qubit_per_A_routed': 419, 'two_qubit_per_Q': 544}
| n | logical qubits | rotations | ANDs | T gates | rotation share |
|---|---|---|---|---|---|
| 2 | 31 | 123 | 66 | 5.62e+03 | 95% |
| 5 | 57 | 318 | 177 | 1.46e+04 | 95% |
| 10 | 99 | 643 | 388 | 2.96e+04 | 95% |
| 20 | 181 | 1293 | 849 | 5.97e+04 | 94% |
| 50 | 423 | 3243 | 2310 | 1.51e+05 | 94% |
| 100 | 825 | 6493 | 5011 | 3.03e+05 | 93% |

| n | target | a | calls | total T | d | physical qubits | runtime |
|---|---|---|---|---|---|---|---|
| 2 | both exhausted | 3.16e-05 | 4.59e+05 | 2.58e+09 | 25 | 1.78e+05 | 1.8 h |
| 5 | loss > half aggregate principal | 1.01e-03 | 8.13e+04 | 1.18e+09 | 25 | 2.27e+05 | 0.8 h |
| 10 | loss > half aggregate principal | 5.91e-06 | 1.06e+06 | 3.14e+10 | 29 | 3.70e+05 | 21.8 h |
| 20 | loss > half aggregate principal | 2.75e-10 | 1.56e+08 | 9.30e+12 | 33 | 7.11e+05 | 269 days |

Oracle time at n=10: 74 ms; C=25.8; kappa_IS=0.049
Break-even a*: Monte Carlo@1us 2.7e-09, Monte Carlo@10us 2.7e-07, Importance sampling@1us 6.6e-12, Importance sampling@10us 6.6e-10
