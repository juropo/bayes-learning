"""marginals_and_joint_from_pyagrum: extracting exact marginals/joints from
an already-fitted gum.BayesNet, for use as build_network's `initial_joint`.
Ground truth is a hand-built 3-node chain (X -> Y -> Z) with known CPTs, so
every extracted value can be checked against a manual calculation.
"""

import numpy as np
import pyagrum as gum
import pytest

from cross_impact.pyagrum_import import marginals_and_joint_from_pyagrum


@pytest.fixture
def chain_bn():
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


def _manual_xy(bn):
    px = bn.cpt("X").toarray()
    py_given_x = np.array([bn.cpt("Y")[{"X": k}] for k in range(2)])
    return px[:, None] * py_given_x


def test_single_variable_marginal(chain_bn):
    marginals, joint = marginals_and_joint_from_pyagrum(chain_bn, ["X"])
    assert np.allclose(marginals[0], [0.3, 0.7])
    assert np.allclose(joint, [0.3, 0.7])


def test_adjacent_pair_joint_and_order(chain_bn):
    manual_xy = _manual_xy(chain_bn)

    marginals, joint = marginals_and_joint_from_pyagrum(chain_bn, ["X", "Y"])
    assert joint.shape == (2, 3)
    assert np.allclose(joint, manual_xy)
    assert np.allclose(marginals[0], manual_xy.sum(axis=1))
    assert np.allclose(marginals[1], manual_xy.sum(axis=0))

    # requesting the reverse order transposes the result accordingly
    _, joint_reversed = marginals_and_joint_from_pyagrum(chain_bn, ["Y", "X"])
    assert joint_reversed.shape == (3, 2)
    assert np.allclose(joint_reversed, manual_xy.T)


def test_non_adjacent_pair_joint(chain_bn):
    """X and Z are correlated only through Y, which isn't requested --
    exercises LazyPropagation's addJointTarget requirement for node sets
    that aren't already a clique."""
    manual_xy = _manual_xy(chain_bn)
    pz_given_y = np.array([chain_bn.cpt("Z")[{"Y": k}] for k in range(3)])
    manual_xz = np.einsum("xy,yz->xz", manual_xy, pz_given_y)

    _, joint = marginals_and_joint_from_pyagrum(chain_bn, ["X", "Z"])
    assert np.allclose(joint, manual_xz)


def test_three_variable_joint(chain_bn):
    marginals, joint = marginals_and_joint_from_pyagrum(chain_bn, ["X", "Y", "Z"])
    assert joint.shape == (2, 3, 2)
    assert joint.sum() == pytest.approx(1.0)
    assert np.allclose(joint.sum(axis=(1, 2)), marginals[0])
    assert np.allclose(joint.sum(axis=(0, 2)), marginals[1])
    assert np.allclose(joint.sum(axis=(0, 1)), marginals[2])


def test_evidence_conditions_the_extraction(chain_bn):
    _, joint_given_x0 = marginals_and_joint_from_pyagrum(chain_bn, ["Y", "Z"], evidence={"X": 0})
    py_given_x0 = chain_bn.cpt("Y")[{"X": 0}]
    pz_given_y = np.array([chain_bn.cpt("Z")[{"Y": k}] for k in range(3)])
    manual = py_given_x0[:, None] * pz_given_y
    assert np.allclose(joint_given_x0, manual)
