"""Port of printing_network.m: the 11-factor 3D-printing/FDF case study from
Section 4 of Roponen & Salo (2024). Node 9 alone needs ~2200 QP variables to
fit (6 parents x 3 states each) -- fine for cvxpy's bundled open-source
solvers, just slow enough that the network is built once per module and
shared across tests rather than refit for each assertion.
"""

import numpy as np
import pytest

from cross_impact import build_network, printing_network


@pytest.fixture(scope="module")
def built():
    return build_network(printing_network(), mode="exact")


def test_marginals_are_matched_exactly(built):
    network = printing_network()
    for i, marginal in enumerate(network.marginals):
        other_axes = tuple(a for a in range(built.joint.ndim) if a != i)
        recovered = built.joint.sum(axis=other_axes)
        assert np.allclose(recovered, marginal, atol=1e-6)


def test_joint_distribution_sums_to_one(built):
    assert built.joint.sum() == pytest.approx(1.0)


def test_most_likely_exogenous_partial_scenario_matches_paper(built):
    """Section 4.2: among the 3^7 = 2187 partial scenarios formed by the 7
    "exogenous" factors (global/Finnish industry, printing speed & cost,
    graduates, regulation, standardization), the most probable is every
    factor taking its second outcome, with probability 8.62% in the paper
    (computed there with MATLAB's lsqlin). The per-node QP is underdetermined
    (Section 3.2 notes the specific solution found depends on the solver and
    starting point), so cvxpy's solvers land on a nearby but not identical
    optimum -- checked here against a wider tolerance than a solver-precision
    difference alone would need, plus the more solver-independent claim that
    it's still the single most probable partial scenario."""
    partial = built.joint.sum(axis=(7, 8, 9, 10))
    state_2_prob = partial[1, 1, 1, 1, 1, 1, 1]
    assert state_2_prob == pytest.approx(0.0862, abs=0.01)
    assert partial[1, 1, 1, 1, 1, 1, 1] == partial.max()
