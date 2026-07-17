"""Extending a fitted cross-impact network into a pyAgrum influence diagram
with decision and utility nodes.

Decision and utility nodes are deliberately kept outside build_network's
fitting machinery entirely: a decision node has no probability distribution
to fit (its value is chosen, not random, so there's no marginal to match and
no cross-impact multiplier to fit against), and a utility node's value table
is a payoff you specify directly, not something cross-impact analysis
produces. Both are supplied directly here rather than fit.

The one subtlety is promoting an already-fitted chance node into a decision
(Decision.promote): a fitted CPT q(child | parent=l) is valid for every
state l of the parent regardless of whether l was drawn from a distribution
or chosen by a decision maker -- nothing about what a CPT means depends on
how the parent's value arose. So promoting a node just means dropping its
own marginal/CPT (a decision doesn't need one) while keeping every
downstream child's already-fitted CPT completely unchanged: no refitting
required, and it correctly captures how outcomes change under each possible
choice. The promoted node's own incoming arcs (its former cross-impact
parents) are dropped, since a decision's value isn't probabilistically
caused by anything -- it's simply set.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pyagrum as gum

from cross_impact.model import BuiltNetwork
from cross_impact.pyagrum_export import _validate_labels


@dataclass(frozen=True)
class Decision:
    """A brand-new decision node: no probability distribution, just a set of
    alternatives chosen by the decision maker."""

    name: str
    alternatives: list[str]


@dataclass(frozen=True)
class ManualChanceNode:
    """A brand-new chance node whose conditional distribution is supplied
    directly rather than fit via cross-impact -- for when a parent is a
    decision (which has no marginal/cross-impact multiplier to fit against),
    or whenever you simply want to hand-specify a CPT."""

    name: str
    n_states: int
    parents: tuple[str, ...]
    cpt: np.ndarray  # shape (n_states, *parent_state_dims), parents in the given order


@dataclass(frozen=True)
class Utility:
    """A utility (value) node: a deterministic payoff table over its
    parents' states (chance and/or decision nodes) -- elicited directly, not
    fit from cross-impact estimates."""

    name: str
    parents: tuple[str, ...]
    table: np.ndarray  # shape = parents' state counts, in the given order; one utility value per parent config


@dataclass(frozen=True)
class InfluenceDiagramSpec:
    """Everything needed to assemble a gum.InfluenceDiagram on top of a
    fitted CrossImpactNetwork: which of its factors become decisions instead
    of chance nodes, plus any brand-new decision/manual-CPT/utility nodes."""

    built: BuiltNetwork
    promote: tuple[str, ...] = ()
    decisions: tuple[Decision, ...] = ()
    manual_nodes: tuple[ManualChanceNode, ...] = ()
    utilities: tuple[Utility, ...] = ()


def _add_utility_table(id_: gum.InfluenceDiagram, name: str, parents: tuple[str, ...], table: np.ndarray) -> None:
    if not parents:
        id_.utility(name)[:] = [float(table)]
        return
    parent_dims = table.shape
    for config in itertools.product(*(range(d) for d in parent_dims)):
        instantiation = {p: s for p, s in zip(parents, config)}
        id_.utility(name)[instantiation] = [float(table[config])]


def build_influence_diagram(spec: InfluenceDiagramSpec, state_labels: dict[int, list[str]] | None = None) -> gum.InfluenceDiagram:
    """Assemble a gum.InfluenceDiagram from a fitted CrossImpactNetwork plus
    decision/utility nodes (see InfluenceDiagramSpec).

    Args:
        spec: the fitted network plus decision/utility additions.
        state_labels: as in to_pyagrum -- {factor_index: [labels]} for
            built.network.factors; ignored for any factor named in
            spec.promote (decisions don't have state-probability labels in
            the same sense, though they still get labels for their
            alternatives -- use Decision/promote's own naming for that).

    Returns:
        The constructed gum.InfluenceDiagram.
    """
    built = spec.built
    network = built.network
    names = [factor.name for factor in network.factors]

    all_names = (
        names
        + [d.name for d in spec.decisions]
        + [n.name for n in spec.manual_nodes]
        + [u.name for u in spec.utilities]
    )
    if len(set(all_names)) != len(all_names):
        duplicates = sorted({n for n in all_names if all_names.count(n) > 1})
        raise ValueError(f"node names must be unique across the whole diagram; duplicates: {duplicates}")

    promote = set(spec.promote)
    unknown_promotions = promote - set(names)
    if unknown_promotions:
        raise ValueError(f"spec.promote names not found among built.network.factors: {sorted(unknown_promotions)}")

    id_ = gum.InfluenceDiagram()

    # 1. The fitted cross-impact network's own factors -- as chance nodes,
    # or as decisions for whichever are being promoted (dropping their own
    # CPT/marginal and incoming arcs, but keeping every child's CPT as-is).
    for i, factor in enumerate(network.factors):
        if factor.name in promote:
            id_.addDecisionNode(gum.LabelizedVariable(factor.name, factor.name, factor.n_states))
            continue
        labels = _validate_labels(i, factor.n_states, state_labels)
        id_.addChanceNode(gum.LabelizedVariable(factor.name, factor.name, labels))

    for i, factor in enumerate(network.factors):
        if factor.name in promote:
            continue
        for p in sorted(factor.parents):
            id_.addArc(names[p], factor.name)
        parents = sorted(factor.parents)
        cpt = built.cpts[i]
        if not parents:
            id_.cpt(factor.name)[:] = cpt.tolist()
            continue
        parent_dims = [network.factors[p].n_states for p in parents]
        for config in itertools.product(*(range(d) for d in parent_dims)):
            instantiation = {names[p]: s for p, s in zip(parents, config)}
            id_.cpt(factor.name)[instantiation] = cpt[(slice(None), *config)].tolist()

    # 2. Brand-new decision nodes.
    for d in spec.decisions:
        id_.addDecisionNode(gum.LabelizedVariable(d.name, d.name, d.alternatives))

    # 3. Brand-new chance nodes with a hand-specified CPT (e.g. depending on
    # a decision). Parents must already be present in the diagram, so these
    # must be given in dependency order.
    for node in spec.manual_nodes:
        if node.cpt.shape[0] != node.n_states:
            raise ValueError(f"{node.name}: cpt's first axis has {node.cpt.shape[0]} states, expected {node.n_states}")
        expected_shape = (node.n_states, *(id_.variable(p).domainSize() for p in node.parents))
        if node.cpt.shape != expected_shape:
            raise ValueError(f"{node.name}: cpt has shape {node.cpt.shape}, expected {expected_shape}")

        id_.addChanceNode(gum.LabelizedVariable(node.name, node.name, node.n_states))
        for p in node.parents:
            id_.addArc(p, node.name)
        if not node.parents:
            id_.cpt(node.name)[:] = node.cpt.tolist()
            continue
        parent_dims = node.cpt.shape[1:]
        for config in itertools.product(*(range(d) for d in parent_dims)):
            instantiation = {p: s for p, s in zip(node.parents, config)}
            id_.cpt(node.name)[instantiation] = node.cpt[(slice(None), *config)].tolist()

    # 4. Utility nodes.
    for u in spec.utilities:
        expected_shape = tuple(id_.variable(p).domainSize() for p in u.parents)
        if u.table.shape != expected_shape:
            raise ValueError(f"{u.name}: table has shape {u.table.shape}, expected {expected_shape}")
        id_.addUtilityNode(gum.LabelizedVariable(u.name, u.name, 1))
        for p in u.parents:
            id_.addArc(p, u.name)
        _add_utility_table(id_, u.name, u.parents, u.table)

    return id_
