"""Extracting marginals/joints from an already-fitted gum.BayesNet (e.g. one
learned from data via pyAgrum's BNLearner), for use as the given/known
prefix of a CrossImpactNetwork -- see build_network's `initial_joint`
argument. This is what lets cross-impact-derived nodes be added *downstream*
of a data-learned sub-network without needing to re-learn or approximate
anything about the already-learned part: its exact joint (not just each
node's marginal treated independently) is pulled out via ordinary exact
inference.
"""

from __future__ import annotations

import numpy as np
import pyagrum as gum


def marginals_and_joint_from_pyagrum(
    bn: gum.BayesNet, names: list[str], evidence: dict | None = None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract the marginal distribution of each of `names` (in that order)
    and their exact joint distribution from a fitted/learned gum.BayesNet,
    via exact inference (gum.LazyPropagation).

    Args:
        bn: a trained/fitted pyAgrum network (e.g. from BNLearner, or
            cross_impact.to_pyagrum).
        names: variable names to extract, in the order they should become
            CrossImpactNetwork factors 0..len(names)-1.
        evidence: optional {name: label_or_index} to condition on before
            extracting (e.g. to build a network for a specific
            scenario/subpopulation rather than the unconditional model).

    Returns:
        (marginals, joint): marginals[i] is the 1-d marginal for names[i];
        joint has shape (n_states(names[0]), ..., n_states(names[-1])),
        suitable as build_network's `initial_joint`.
    """
    ie = gum.LazyPropagation(bn)
    if len(names) > 1:
        # LazyPropagation only answers joint queries it was told to prepare
        # for before inference runs -- without this, jointPosterior raises
        # UndefinedElement for any set of nodes that isn't already a clique
        # of its junction tree (e.g. two nodes correlated only through a
        # third, unrequested one).
        ie.addJointTarget(set(names))
    if evidence:
        ie.setEvidence(evidence)
    ie.makeInference()

    marginals = [ie.posterior(name).toarray() for name in names]

    if len(names) == 1:
        joint = marginals[0]
    else:
        joint_potential = ie.jointPosterior(set(names))
        # pyAgrum's Tensor.toarray() returns axes in the REVERSE of .names
        # order (a documented pyAgrum quirk, not a numpy convention).
        order = list(reversed(joint_potential.names))
        perm = [order.index(name) for name in names]
        joint = np.transpose(joint_potential.toarray(), axes=perm)

    return marginals, joint
