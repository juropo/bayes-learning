"""engine="ipf": Iterative Proportional Fitting as a dependency-free
alternative to cvxpy for loss="kl". Checks that it produces valid,
marginal-matching distributions -- the property IPF is specifically built to
guarantee, via its own alternating-projection hard-constraint sub-loop -- and
that it broadly agrees with the cvxpy "kl" solve despite being a completely
independent numerical method.
"""

import numpy as np
import pytest

from cross_impact import (
    build_network,
    cross_impacts_from_joint,
    energy_network,
    fit_joint_distribution,
    printing_network,
)
from cross_impact.model import CrossImpactNetwork, UncertaintyFactor


def _assert_matches_marginals(built, network, atol=1e-6):
    assert built.joint.sum() == pytest.approx(1.0, abs=atol)
    for i, marginal in enumerate(network.marginals):
        other_axes = tuple(a for a in range(built.joint.ndim) if a != i)
        recovered = built.joint.sum(axis=other_axes)
        assert np.allclose(recovered, marginal, atol=atol)


def test_ipf_matches_marginals_on_energy_network():
    network = energy_network()
    built = build_network(network, mode="exact", loss="kl", engine="ipf")
    _assert_matches_marginals(built, network)


def test_ipf_matches_marginals_on_printing_network():
    network = printing_network()
    built = build_network(network, mode="exact", loss="kl", engine="ipf")
    _assert_matches_marginals(built, network, atol=1e-8)


def test_ipf_broadly_agrees_with_cvxpy_kl():
    network = energy_network()
    cvxpy_built = build_network(network, mode="exact", loss="kl", engine="cvxpy")
    ipf_built = build_network(network, mode="exact", loss="kl", engine="ipf")

    total_abs_diff = np.abs(cvxpy_built.joint - ipf_built.joint).sum()
    # Two different numerical methods solving a (for this network) mildly
    # underdetermined, not-fully-consistent fitting problem -- exact
    # agreement isn't expected, but they should land in the same
    # neighborhood rather than materially different distributions.
    assert total_abs_diff < 0.3


def test_ipf_finds_the_same_most_likely_partial_scenario_as_cvxpy():
    built = build_network(printing_network(), mode="exact", loss="kl", engine="ipf")
    partial = built.joint.sum(axis=(7, 8, 9, 10))
    assert partial[1, 1, 1, 1, 1, 1, 1] == partial.max()


def test_ipf_direct_joint_fit_matches_marginals():
    rng = np.random.default_rng(4)
    truth = rng.exponential(1.0, size=(3, 3, 3))
    truth /= truth.sum()
    cross_impacts, marginals = cross_impacts_from_joint(truth)
    factors = [UncertaintyFactor(str(i), 3, ()) for i in range(3)]
    network = CrossImpactNetwork(factors=factors, marginals=marginals, cross_impacts=cross_impacts)

    fitted = fit_joint_distribution(network, loss="kl", engine="ipf")

    assert fitted.sum() == pytest.approx(1.0, abs=1e-8)
    for i, marginal in enumerate(marginals):
        other_axes = tuple(a for a in range(fitted.ndim) if a != i)
        assert np.allclose(fitted.sum(axis=other_axes), marginal, atol=1e-8)


def test_ipf_requires_kl_loss():
    network = energy_network()
    with pytest.raises(ValueError, match="engine=\"ipf\""):
        build_network(network, mode="exact", loss="least_squares", engine="ipf")
