"""build_network(..., initial_joint=...): seeding the running joint with an
already-known distribution over a prefix of factors (e.g. from an
already-learned pyAgrum network -- see test_pyagrum_roundtrip.py for the
full pyAgrum-integrated version), so those factors are treated as given
rather than fit from cross-impact estimates. This file checks the mechanic
in isolation with a hand-built correlated joint, so expected values can be
computed by hand rather than only cross-checked against another tool.
"""

import numpy as np
import pytest

from cross_impact import CrossImpactNetwork, UncertaintyFactor, build_network

# A (2 states) and B (3 states), deliberately correlated (not independent).
TRUE_JOINT_AB = np.array(
    [
        [0.10, 0.05, 0.15],
        [0.30, 0.10, 0.30],
    ]
)
MARGINAL_A = TRUE_JOINT_AB.sum(axis=1)
MARGINAL_B = TRUE_JOINT_AB.sum(axis=0)


def _network_with_new_factor_c():
    factors = [
        UncertaintyFactor("A", 2, ()),
        UncertaintyFactor("B", 3, (0,)),
        UncertaintyFactor("C", 2, (0, 1)),
    ]
    marginal_c = np.array([0.4, 0.6])
    cross_impacts = {
        (0, 2): np.array([[1.5, 0.7], [0.7, 1.2]]),  # rows=A, cols=C
        (1, 2): np.array([[1.2, 1.0, 0.8], [0.8, 1.0, 1.2]]).T,  # stored rows=B, cols=C
    }
    network = CrossImpactNetwork(
        factors=factors, marginals=[MARGINAL_A, MARGINAL_B, marginal_c], cross_impacts=cross_impacts
    )
    return network, marginal_c


def test_given_prefix_cpts_are_derived_exactly_from_the_seeded_joint():
    network, _ = _network_with_new_factor_c()
    built = build_network(network, mode="exact", initial_joint=TRUE_JOINT_AB)

    assert np.allclose(built.cpts[0], MARGINAL_A)
    expected_b_given_a = (TRUE_JOINT_AB / MARGINAL_A[:, None]).T  # shape (n_B, n_A)
    assert np.allclose(built.cpts[1], expected_b_given_a)


def test_seeded_prefix_correlation_survives_into_the_final_joint_unchanged():
    """The whole point: fitting the new factor C must not distort A and B's
    (already-known, correlated) joint at all."""
    network, marginal_c = _network_with_new_factor_c()
    built = build_network(network, mode="exact", initial_joint=TRUE_JOINT_AB)

    assert np.allclose(built.joint.sum(axis=2), TRUE_JOINT_AB)
    assert np.allclose(built.joint.sum(axis=(0, 1)), marginal_c, atol=1e-5)
    assert built.joint.sum() == pytest.approx(1.0)


def test_mismatched_marginal_raises():
    network, _ = _network_with_new_factor_c()
    network.marginals[0] = np.array([0.9, 0.1])  # inconsistent with TRUE_JOINT_AB's own A-marginal
    with pytest.raises(ValueError, match="doesn't match"):
        build_network(network, mode="exact", initial_joint=TRUE_JOINT_AB)


def test_initial_joint_requires_exact_mode():
    network, _ = _network_with_new_factor_c()
    with pytest.raises(ValueError, match='mode="exact"'):
        build_network(network, mode="independent", initial_joint=TRUE_JOINT_AB)
