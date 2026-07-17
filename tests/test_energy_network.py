"""Port of energy_network.m: the small 3-factor Soisalo example worked
through in Section 3 of Roponen & Salo (2024). Expected values are read
directly off Tables 1, 2 and 4 of the paper."""

import numpy as np
import pytest

from cross_impact import build_network, energy_network, scenario_probability

# Table 4, "2nd iteration": p(renewables | nuclear), columns = nuclear state.
TABLE_4_RENEWABLES_GIVEN_NUCLEAR = np.array(
    [
        [0.10, 0.50, 0.50],
        [0.17, 0.25, 0.21],
        [0.37, 0.20, 0.28],
        [0.36, 0.06, 0.02],
    ]
)


def test_marginals_are_matched_exactly():
    network = energy_network()
    built = build_network(network, mode="exact")

    for i, marginal in enumerate(network.marginals):
        other_axes = tuple(a for a in range(built.joint.ndim) if a != i)
        recovered = built.joint.sum(axis=other_axes)
        assert np.allclose(recovered, marginal, atol=1e-6)


def test_joint_distribution_sums_to_one():
    built = build_network(energy_network(), mode="exact")
    assert built.joint.sum() == pytest.approx(1.0)


def test_renewables_cpt_matches_paper_table_4():
    built = build_network(energy_network(), mode="exact")
    # cpts[1] has shape (n_renewables, n_nuclear) since node 1's only parent is node 0.
    assert np.allclose(built.cpts[1], TABLE_4_RENEWABLES_GIVEN_NUCLEAR, atol=0.02)


def test_scenario_probability_matches_joint_lookup():
    built = build_network(energy_network(), mode="exact")
    scenario = (1, 2, 0)
    assert scenario_probability(built, scenario) == pytest.approx(built.joint[scenario], abs=1e-9)
