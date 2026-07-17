"""End-to-end: extract a joint from an already-"learned" gum.BayesNet
(marginals_and_joint_from_pyagrum), fit new cross-impact nodes conditioned
on it (build_network(..., initial_joint=...)), and graft them back onto the
original network (graft_onto_pyagrum) -- the full "learn part of a network
from data, add cross-impact-derived nodes downstream" workflow.

The network: X -> Y -> Z is the "learned" part; W is a new cross-impact node
depending on both Y and Z (exercising the multi-parent/correlated-joint
case). Checked against pyAgrum's own inference on the merged network, not
just build_network's internal bookkeeping -- the strongest available check
that the two pieces actually compose correctly.
"""

import numpy as np
import pyagrum as gum
import pytest

from cross_impact import CrossImpactNetwork, UncertaintyFactor, build_network
from cross_impact.pyagrum_export import graft_onto_pyagrum
from cross_impact.pyagrum_import import marginals_and_joint_from_pyagrum


@pytest.fixture
def learned_bn():
    bn = gum.BayesNet()
    bn.add(gum.LabelizedVariable("X", "X", ["x0", "x1"]))
    bn.add(gum.LabelizedVariable("Y", "Y", ["y0", "y1", "y2"]))
    bn.add(gum.LabelizedVariable("Z", "Z", ["z0", "z1"]))
    bn.addArc("X", "Y")
    bn.addArc("Y", "Z")
    bn.cpt("X")[:] = [0.3, 0.7]
    bn.cpt("Y")[{"X": 0}] = [0.6, 0.3, 0.1]
    bn.cpt("Y")[{"X": 1}] = [0.1, 0.2, 0.7]
    bn.cpt("Z")[{"Y": 0}] = [0.9, 0.1]
    bn.cpt("Z")[{"Y": 1}] = [0.5, 0.5]
    bn.cpt("Z")[{"Y": 2}] = [0.2, 0.8]
    return bn


def _joint(bn, names):
    ie = gum.LazyPropagation(bn)
    if len(names) > 1:
        ie.addJointTarget(set(names))
    ie.makeInference()
    if len(names) == 1:
        return ie.posterior(names[0]).toarray()
    pot = ie.jointPosterior(set(names))
    order = list(reversed(pot.names))
    perm = [order.index(n) for n in names]
    return np.transpose(pot.toarray(), axes=perm)


def _build_and_graft(learned_bn):
    marginals_given, joint_yz = marginals_and_joint_from_pyagrum(learned_bn, ["Y", "Z"])
    factors = [
        UncertaintyFactor("Y", 3, ()),
        UncertaintyFactor("Z", 2, (0,)),
        UncertaintyFactor("W", 2, (0, 1)),
    ]
    marginal_w = np.array([0.4, 0.6])
    network = CrossImpactNetwork(
        factors=factors,
        marginals=[marginals_given[0], marginals_given[1], marginal_w],
        cross_impacts={
            (0, 2): np.array([[1.3, 0.6], [1.0, 1.0], [0.7, 1.4]]),
            (1, 2): np.array([[1.4, 0.6], [0.6, 1.4]]),
        },
    )
    built = build_network(network, mode="exact", initial_joint=joint_yz)
    merged = graft_onto_pyagrum(learned_bn, built, n_given=2)
    return built, merged, joint_yz


def test_new_node_is_fit_against_the_true_correlated_parent_joint(learned_bn):
    built, _, joint_yz = _build_and_graft(learned_bn)
    assert np.allclose(built.joint.sum(axis=2), joint_yz)
    assert np.allclose(built.joint.sum(axis=(0, 1)), [0.4, 0.6])


def test_merged_network_keeps_upstream_node_and_arc(learned_bn):
    _, merged, _ = _build_and_graft(learned_bn)
    names = {merged.variable(i).name() for i in merged.nodes()}
    assert names == {"X", "Y", "Z", "W"}
    arcs = {(merged.variable(a).name(), merged.variable(b).name()) for a, b in merged.arcs()}
    assert ("X", "Y") in arcs
    assert ("Y", "Z") in arcs
    assert ("Y", "W") in arcs
    assert ("Z", "W") in arcs


def test_grafting_does_not_mutate_the_original_network(learned_bn):
    original_size = learned_bn.size()
    _build_and_graft(learned_bn)
    assert learned_bn.size() == original_size


def test_merged_inference_leaves_learned_distribution_unchanged(learned_bn):
    """The whole point: Y,Z's distribution in the merged network must be
    bit-for-bit what it was before grafting on W -- adding a downstream
    cross-impact node must not perturb the already-learned part at all."""
    original_yz = _joint(learned_bn, ["Y", "Z"])
    _, merged, _ = _build_and_graft(learned_bn)
    merged_yz = _joint(merged, ["Y", "Z"])
    assert np.allclose(merged_yz, original_yz)


def test_merged_inference_matches_build_network_joint_exactly(learned_bn):
    """The strongest check: pyAgrum's own exact inference on the merged
    network must reproduce build_network's computed joint over (Y, Z, W),
    not just each piece separately."""
    built, merged, _ = _build_and_graft(learned_bn)
    merged_yzw = _joint(merged, ["Y", "Z", "W"])
    assert np.allclose(merged_yzw, built.joint)
