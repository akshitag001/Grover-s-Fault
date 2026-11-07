import numpy as np
import pytest
from scipy.linalg import expm

from adbq import model as M


@pytest.mark.parametrize("name", list(M.SCENARIOS))
def test_table5_reproduced(name):
    sc = M.SCENARIOS[name]
    got = M.exact_targets(sc)
    ref = M.REFERENCE[name]
    el, joint, exc, sl, exh = (got[t] for t in M.TARGETS)
    assert round(el / 1e4) * 1e4 == pytest.approx(ref[0], abs=1)       # USD x.xx m
    assert round(joint, 5) == pytest.approx(ref[1], abs=1e-12)
    assert round(exc, 5) == pytest.approx(ref[2], abs=1e-12)
    assert round(sl) == ref[3]
    assert round(exh, 6) == pytest.approx(ref[4], abs=1e-12)


@pytest.mark.parametrize("name", list(M.SCENARIOS))
def test_exp3g_equals_convolution(name):
    sc = M.SCENARIOS[name]
    d1 = M.term_distribution_convolution(sc)
    d2 = M.term_distribution_exp3g(sc)
    assert np.max(np.abs(d1 - d2)) < 1e-14


def test_brief_checkpoints():
    sc = M.SCENARIOS["adb-base"]
    (_, L1), (_, L2) = M.compound_poisson_rates(M.TAIL_KG), M.compound_poisson_rates(M.TAIL_TJ)
    assert M.shared_eq_rate(sc, L1, L2) == pytest.approx(0.0029667, abs=5e-8)
    assert expm(M.eq_generator(sc))[0, 0] == pytest.approx(0.9262, abs=5e-5)
    assert M.annual_distribution(sc)[0] == pytest.approx(0.8940, abs=5e-5)
    r = M.rain_probs(sc)
    assert r[0] == pytest.approx(0.96526, abs=5e-6)
    assert r[3] == pytest.approx(0.00526, abs=5e-6)


def test_table2_rates():
    r, L = M.compound_poisson_rates(M.TAIL_KG)
    np.testing.assert_allclose(r, [0.016147, 0.005498, 0.006215, 0.011743], atol=5e-7)
    assert L == pytest.approx(0.039604, abs=5e-7)
