import numpy as np
import pytest

from adbq import model as M, mps, qae as Q, samplers as S


@pytest.mark.parametrize("target", M.TARGETS)
def test_grover_probabilities_match_theory(target):
    sc = M.SCENARIOS["adb-base"]
    A = Q.build_A(sc, target)
    assert np.abs(A.T @ A - np.eye(Q.DIM)).max() < 1e-12
    a = Q.amplitude(sc, target)
    ks = np.arange(0, 40)
    th = np.arcsin(np.sqrt(a))
    assert np.abs(Q.grover_probs(A, ks) - np.sin((2 * ks + 1) * th) ** 2).max() < 1e-12


def test_mlae_ratio_schedule_converges():
    rng = np.random.default_rng(0)
    a = Q.amplitude(M.SCENARIOS["adb-base"], M.TARGETS[4])
    ks = Q.schedule("ratio1.5", 18)
    est = np.array([Q.mlae(a, ks, 100, rng)[0] for _ in range(50)])
    assert np.sqrt(np.mean((est / a - 1) ** 2)) < 0.03


@pytest.mark.parametrize("name", list(M.SCENARIOS))
def test_mps_reproduces_table5(name):
    sc = M.SCENARIOS[name]
    t = mps.targets(mps.Chain.from_scenario(sc))
    v = [t["expected_loss_usd"], t["all_pay_probability"], t["loss_gt_principal_probability"],
         t["stop_loss_usd"], t["all_exhausted_probability"]]
    np.testing.assert_allclose(v, list(M.exact_targets(sc).values()), rtol=1e-12)


@pytest.mark.parametrize("n", [3, 4])
def test_mps_matches_brute_force(n):
    ch = mps.Chain(n)
    b = mps.brute_force_targets(ch)
    m = mps.targets(ch)
    for k in b:
        assert m[k] == pytest.approx(b[k], rel=1e-12)


@pytest.mark.parametrize("name", list(S.SAMPLERS))
def test_samplers_unbiased(name):
    rng = np.random.default_rng(1)
    P = S.Problem(M.SCENARIOS["adb-base"])
    est = S.SAMPLERS[name](P, 2**18, rng)
    assert abs(est[0] / P.exact[0] - 1) < 0.02      # expected loss within 2%
