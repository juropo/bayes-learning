"""The loss="kl" variant of the fitting objective (generalized KL divergence
from the cross-impact-implied targets, instead of least squares). Checks that
it still produces valid, marginal-matching distributions -- both through the
per-node iterative fit (build_network) and the direct whole-network fit
(fit_joint_distribution) -- and that it's actually a different fit from
least squares rather than a no-op.
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


def test_kl_fit_matches_marginals_on_energy_network():
    network = energy_network()
    built = build_network(network, mode="exact", loss="kl")

    assert built.joint.sum() == pytest.approx(1.0, abs=1e-5)
    for i, marginal in enumerate(network.marginals):
        other_axes = tuple(a for a in range(built.joint.ndim) if a != i)
        recovered = built.joint.sum(axis=other_axes)
        assert np.allclose(recovered, marginal, atol=1e-4)


def test_kl_fit_differs_from_least_squares():
    network = energy_network()
    least_squares = build_network(network, mode="exact", loss="least_squares")
    kl = build_network(network, mode="exact", loss="kl")

    # Same constraints, different objective -- the two fits shouldn't coincide.
    assert not np.allclose(least_squares.joint, kl.joint, atol=1e-3)


def test_kl_fit_matches_marginals_on_printing_network():
    network = printing_network()
    built = build_network(network, mode="exact", loss="kl")

    assert built.joint.sum() == pytest.approx(1.0, abs=1e-4)
    for i, marginal in enumerate(network.marginals):
        other_axes = tuple(a for a in range(built.joint.ndim) if a != i)
        recovered = built.joint.sum(axis=other_axes)
        assert np.allclose(recovered, marginal, atol=1e-3)


def test_kl_direct_joint_fit_matches_marginals():
    rng = np.random.default_rng(3)
    truth = rng.exponential(1.0, size=(3, 3, 3))
    truth /= truth.sum()
    cross_impacts, marginals = cross_impacts_from_joint(truth)
    factors = [UncertaintyFactor(str(i), 3, ()) for i in range(3)]
    network = CrossImpactNetwork(factors=factors, marginals=marginals, cross_impacts=cross_impacts)

    fitted = fit_joint_distribution(network, loss="kl")

    assert fitted.sum() == pytest.approx(1.0, abs=1e-5)
    for i, marginal in enumerate(marginals):
        other_axes = tuple(a for a in range(fitted.ndim) if a != i)
        assert np.allclose(fitted.sum(axis=other_axes), marginal, atol=1e-4)


def test_unknown_loss_raises():
    network = energy_network()
    with pytest.raises(ValueError, match="unknown loss"):
        build_network(network, mode="exact", loss="not_a_real_loss")
