"""build_influence_diagram: extending a fitted CrossImpactNetwork with
decision and utility nodes.

Covers the two ways decision nodes enter the diagram:
- Decision.promote: an already cross-impact-fitted chance node becomes a
  decision, keeping every downstream child's CPT unchanged (verified
  against build_network's own built.cpts, not just "it runs").
- Decision(): a brand-new decision, with a hand-specified CPT
  (ManualChanceNode) for any chance node that depends on it, since
  cross-impact fitting doesn't apply to a decision parent.

MEU/optimal-decision checks use hand-computable utility tables so the
expected values aren't just "whatever pyAgrum returns".
"""

import numpy as np
import pyagrum as gum
import pytest

from cross_impact import build_network, energy_network
from cross_impact.influence_diagram import (
    Decision,
    InfluenceDiagramSpec,
    ManualChanceNode,
    Utility,
    build_influence_diagram,
)


@pytest.fixture
def built():
    return build_network(energy_network(), mode="exact")


def test_promoted_node_becomes_a_decision(built):
    spec = InfluenceDiagramSpec(built=built, promote=("Nuclear power",))
    id_ = build_influence_diagram(spec)

    assert id_.isDecisionNode("Nuclear power")
    assert id_.isChanceNode("Renewables")
    assert id_.isChanceNode("Energy storage")


def test_promoted_node_children_keep_their_fitted_cpts_unchanged(built):
    spec = InfluenceDiagramSpec(built=built, promote=("Nuclear power",))
    id_ = build_influence_diagram(spec)

    recovered = np.array([[id_.cpt("Renewables")[{"Nuclear power": a}][k] for a in range(3)] for k in range(4)])
    assert np.allclose(recovered, built.cpts[1])

    recovered_es = np.zeros(built.cpts[2].shape)
    for a in range(3):
        for r in range(4):
            for k in range(3):
                recovered_es[k, a, r] = id_.cpt("Energy storage")[{"Nuclear power": a, "Renewables": r}][k]
    assert np.allclose(recovered_es, built.cpts[2])


def test_promoted_node_has_no_incoming_arcs(built):
    """Nuclear power has no parents in the original network, but this also
    checks the general rule: a promoted decision never gets incoming arcs,
    since its value is chosen, not caused by anything."""
    spec = InfluenceDiagramSpec(built=built, promote=("Renewables",))  # Renewables *does* have a parent (Nuclear power)
    id_ = build_influence_diagram(spec)
    assert id_.isDecisionNode("Renewables")
    arcs_into_renewables = [a for a, b in id_.arcs() if id_.variable(b).name() == "Renewables"]
    assert arcs_into_renewables == []


def test_promoted_decision_meu_matches_hand_calculation(built):
    utility_table = np.array([100.0, 50.0, -20.0])  # indexed by Energy storage state
    spec = InfluenceDiagramSpec(
        built=built,
        promote=("Nuclear power",),
        utilities=(Utility(name="U", parents=("Energy storage",), table=utility_table),),
    )
    id_ = build_influence_diagram(spec)

    ie = gum.ShaferShenoyLIMIDInference(id_)
    ie.makeInference()

    # hand-computed expected utility per Nuclear power choice, using
    # built.cpts[2] marginalized over Renewables
    expected_utilities = []
    for a in range(3):
        p_storage = sum(built.cpts[2][:, a, r] * built.cpts[1][r, a] for r in range(4))
        expected_utilities.append(float(np.dot(p_storage, utility_table)))
    best = max(range(3), key=lambda a: expected_utilities[a])

    assert ie.MEU()["mean"] == pytest.approx(expected_utilities[best], abs=1e-3)
    policy = ie.optimalDecision("Nuclear power")
    assert policy[best] == pytest.approx(1.0)


def test_new_decision_with_manual_cpt_and_utility(built):
    decision = Decision(name="Investment", alternatives=["subsidize", "no_subsidy"])
    opinion_cpt = np.array(
        [
            [[0.8, 0.6, 0.3], [0.3, 0.2, 0.1]],  # positive
            [[0.2, 0.4, 0.7], [0.7, 0.8, 0.9]],  # negative
        ]
    )
    manual_node = ManualChanceNode(
        name="Public opinion", n_states=2, parents=("Investment", "Nuclear power"), cpt=opinion_cpt
    )
    utility_table = np.array([[50.0, -30.0], [10.0, 5.0]])  # rows=Investment, cols=Public opinion
    utility = Utility(name="U", parents=("Investment", "Public opinion"), table=utility_table)

    spec = InfluenceDiagramSpec(built=built, decisions=(decision,), manual_nodes=(manual_node,), utilities=(utility,))
    id_ = build_influence_diagram(spec)

    assert id_.isDecisionNode("Investment")
    assert id_.isChanceNode("Public opinion")
    assert id_.isChanceNode("Nuclear power")  # untouched, still a regular chance node

    recovered = np.zeros(opinion_cpt.shape)
    for inv in range(2):
        for nuc in range(3):
            for k in range(2):
                recovered[k, inv, nuc] = id_.cpt("Public opinion")[{"Investment": inv, "Nuclear power": nuc}][k]
    assert np.allclose(recovered, opinion_cpt)

    nuclear_marginal = built.network.marginals[0]
    p_positive = opinion_cpt[0] @ nuclear_marginal  # shape (2,) over Investment
    expected_utility_subsidize = p_positive[0] * 50.0 + (1 - p_positive[0]) * -30.0
    expected_utility_no_subsidy = p_positive[1] * 10.0 + (1 - p_positive[1]) * 5.0
    best = "subsidize" if expected_utility_subsidize > expected_utility_no_subsidy else "no_subsidy"

    ie = gum.ShaferShenoyLIMIDInference(id_)
    ie.makeInference()
    expected_meu = max(expected_utility_subsidize, expected_utility_no_subsidy)
    assert ie.MEU()["mean"] == pytest.approx(expected_meu, abs=1e-3)
    policy = ie.optimalDecision("Investment")
    assert policy[0 if best == "subsidize" else 1] == pytest.approx(1.0)


def test_zero_parent_utility_node(built):
    spec = InfluenceDiagramSpec(built=built, utilities=(Utility(name="U", parents=(), table=np.array(42.0)),))
    id_ = build_influence_diagram(spec)
    assert id_.utility("U")[:][0] == pytest.approx(42.0)


def test_duplicate_names_raise(built):
    spec = InfluenceDiagramSpec(built=built, decisions=(Decision(name="Nuclear power", alternatives=["a", "b"]),))
    with pytest.raises(ValueError, match="unique"):
        build_influence_diagram(spec)


def test_unknown_promotion_name_raises(built):
    spec = InfluenceDiagramSpec(built=built, promote=("Not A Real Factor",))
    with pytest.raises(ValueError, match="spec.promote"):
        build_influence_diagram(spec)


def test_wrong_utility_table_shape_raises(built):
    bad_utility = Utility(name="U", parents=("Nuclear power",), table=np.zeros(5))
    spec = InfluenceDiagramSpec(built=built, utilities=(bad_utility,))
    with pytest.raises(ValueError, match="expected"):
        build_influence_diagram(spec)


def test_wrong_manual_cpt_shape_raises(built):
    decision = Decision(name="Investment", alternatives=["a", "b"])
    bad_node = ManualChanceNode(name="X", n_states=2, parents=("Investment",), cpt=np.zeros((2, 5)))
    spec = InfluenceDiagramSpec(built=built, decisions=(decision,), manual_nodes=(bad_node,))
    with pytest.raises(ValueError, match="expected"):
        build_influence_diagram(spec)
