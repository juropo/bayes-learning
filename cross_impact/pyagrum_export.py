"""Export a fitted CrossImpactNetwork (from build_network) into a pyAgrum
gum.BayesNet, for inference/visualization -- the pyAgrum equivalent of what
genie_parser.m did for GeNIe in the original MATLAB code.

graft_onto_pyagrum handles the reverse-composition case: rather than
building a fresh BayesNet from scratch, it adds newly-fitted cross-impact
nodes onto an EXISTING one (e.g. a network learned from data via
BNLearner), leaving that network's own nodes/CPTs untouched -- the natural
counterpart to build_network's `initial_joint`, which is how those new
nodes get correctly conditioned on the existing network's nodes in the
first place (see cross_impact.pyagrum_import.marginals_and_joint_from_pyagrum).
"""

from __future__ import annotations

import itertools

import pyagrum as gum

from cross_impact.model import BuiltNetwork, CrossImpactNetwork


def _validate_labels(factor_index: int, n_states: int, state_labels: dict[int, list[str]] | None) -> list[str]:
    labels = (state_labels or {}).get(factor_index)
    if labels is None:
        return [str(s) for s in range(n_states)]
    if len(labels) != n_states:
        raise ValueError(f"state_labels[{factor_index}] has {len(labels)} labels, expected {n_states}")
    return labels


def _add_factor(
    bn: gum.BayesNet,
    network: CrossImpactNetwork,
    built: BuiltNetwork,
    i: int,
    names: list[str],
    state_labels: dict[int, list[str]] | None,
) -> None:
    """Add factor `i` (node, arcs from its parents, and CPT) to `bn` in
    place. `names` is the full factor-index-to-pyAgrum-name list -- parents
    may refer to nodes already present in `bn` before this call (as with
    graft_onto_pyagrum) or added earlier in the same pass (as with
    to_pyagrum)."""
    factor = network.factors[i]
    labels = _validate_labels(i, factor.n_states, state_labels)
    bn.add(gum.LabelizedVariable(factor.name, factor.name, labels))

    parents = sorted(factor.parents)
    for p in parents:
        bn.addArc(names[p], factor.name)

    cpt = built.cpts[i]  # shape (n_states, *parent_state_dims)
    if not parents:
        bn.cpt(factor.name)[:] = cpt.tolist()
        return
    parent_dims = [network.factors[p].n_states for p in parents]
    for config in itertools.product(*(range(d) for d in parent_dims)):
        # Assigning by {parent_name: state} dict rather than raw array
        # position sidesteps pyAgrum's internal (not insertion-order) axis
        # ordering for a CPT's parent variables.
        instantiation = {names[p]: s for p, s in zip(parents, config)}
        bn.cpt(factor.name)[instantiation] = cpt[(slice(None), *config)].tolist()


def to_pyagrum(built: BuiltNetwork, state_labels: dict[int, list[str]] | None = None) -> gum.BayesNet:
    """Build a gum.BayesNet with one LabelizedVariable per factor (named
    after factor.name), arcs matching the network's parent structure, and
    CPTs set from `built.cpts`.

    Args:
        built: a fitted network, as returned by build_network.
        state_labels: optional {factor_index: [label_for_state_0, ...]}
            giving human-readable outcome names. Factors not present (or if
            the argument is omitted entirely) default to "0", "1", ... --
            same convention as genie_parser.m's generic "State1", "State2", ...

    Returns:
        The constructed gum.BayesNet.
    """
    network = built.network
    names = [factor.name for factor in network.factors]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"factor names must be unique to export to pyAgrum; duplicates: {duplicates}")

    bn = gum.BayesNet()
    for i in range(len(network.factors)):
        _add_factor(bn, network, built, i, names, state_labels)
    return bn


def graft_onto_pyagrum(
    bn: gum.BayesNet, built: BuiltNetwork, n_given: int, state_labels: dict[int, list[str]] | None = None
) -> gum.BayesNet:
    """Add the newly-fitted factors (`built.network.factors[n_given:]`) onto
    an existing gum.BayesNet, e.g. one learned from data via BNLearner --
    the counterpart to build_network(..., initial_joint=...), whose first
    `n_given` factors must be the same nodes (by name) already present in
    `bn`. `bn` itself is left untouched; a modified copy is returned.

    Args:
        bn: an existing network (e.g. learned via BNLearner) that already
            contains a node for each of `built.network.factors[:n_given]`,
            by matching name.
        built: the result of build_network(network, initial_joint=...),
            where `network.factors[:n_given]` correspond to `bn`'s existing
            nodes and `network.factors[n_given:]` are the new ones to graft
            on.
        n_given: how many of `built.network.factors` are already in `bn`
            (must match the `initial_joint` used to produce `built`).
        state_labels: as in to_pyagrum, but indexed the same way (i.e.
            entries for indices < n_given are ignored, since those nodes
            already exist in `bn` with their own labels).

    Returns:
        A new gum.BayesNet: a copy of `bn` plus the grafted-on nodes/arcs/CPTs.
    """
    network = built.network
    names = [factor.name for factor in network.factors]

    given_names = names[:n_given]
    existing_names = {bn.variable(i).name() for i in bn.nodes()}
    missing = [n for n in given_names if n not in existing_names]
    if missing:
        raise ValueError(f"bn is missing nodes for the given factors: {missing}")

    new_names = names[n_given:]
    if len(set(new_names)) != len(new_names):
        duplicates = sorted({n for n in new_names if new_names.count(n) > 1})
        raise ValueError(f"new factor names must be unique; duplicates: {duplicates}")
    collisions = sorted(existing_names & set(new_names))
    if collisions:
        raise ValueError(f"new factor names collide with bn's existing nodes: {collisions}")

    merged = gum.BayesNet(bn)
    for i in range(n_given, len(network.factors)):
        _add_factor(merged, network, built, i, names, state_labels)
    return merged
