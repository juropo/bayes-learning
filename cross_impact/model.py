"""Data structures and joint-distribution bookkeeping for the cross-impact
method of Roponen & Salo (2024), "A probabilistic cross-impact methodology
for explorative scenario analysis", https://doi.org/10.1002/ffo2.165.

An UncertaintyFactor is a discrete random variable (Section 3 of the paper);
a scenario is one outcome per factor. Rather than eliciting probabilities for
every scenario directly (infeasible once there are more than a handful of
factors), the method elicits marginal probabilities per factor plus pairwise
cross-impact multipliers C^ij_kl = P(X^i=k | X^j=l) / P(X^i=k) for selected
pairs, then fits a Bayesian network's conditional probability tables to those
estimates one node at a time in topological order (fit_conditional_distribution
in fitting.py implements the per-node QP; build_network below drives the
iteration and tracks the joint distribution needed to condition each new
node on its already-processed parents).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class UncertaintyFactor:
    """One node of the network. `parents` are indices into the owning
    CrossImpactNetwork.factors list and must all be smaller than this
    factor's own index -- factors are processed in list order, which must
    therefore already be a topological order of the dependency graph.
    """

    name: str
    n_states: int
    parents: tuple[int, ...] = ()


@dataclass
class CrossImpactNetwork:
    """factors[i].parents give the DAG; marginals[i] is p_hat^i (shape
    (n_states_i,)); cross_impacts holds the elicited multipliers Ĉ^ij_kl for
    every (i, j) pair with i < j that appears as a parent-child edge (or,
    for fit_joint_distribution's direct-fit formulation, any pair at all)."""

    factors: list[UncertaintyFactor]
    marginals: list[np.ndarray]
    cross_impacts: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def cross_impact(self, i: int, j: int) -> np.ndarray:
        """Ĉ^ij_kl as an (n_states_i, n_states_j) array, oriented so that
        cross_impact(i, j) is always the transpose of cross_impact(j, i)."""
        if i == j:
            raise ValueError("cross-impact is only defined between distinct factors")
        lo, hi = (i, j) if i < j else (j, i)
        block = self.cross_impacts.get((lo, hi))
        if block is None:
            raise KeyError(f"no cross-impact estimate stored for factors {i} and {j}")
        return block if i < j else block.T


@dataclass
class BuiltNetwork:
    """Result of build_network: cpts[i] is q(k | s_parents), shape
    (n_states_i, *parent_state_dims); joint is the full running joint
    distribution over all factors (only populated when mode="exact")."""

    network: CrossImpactNetwork
    cpts: list[np.ndarray]
    joint: np.ndarray | None


def parent_joint_distribution(joint: np.ndarray | None, parents: tuple[int, ...]) -> np.ndarray:
    """Marginalize `joint` (one axis per already-processed factor, in factor-
    index order) down to just the axes in `parents`, matching parentPD.m."""
    if not parents:
        return np.array(1.0)
    non_parent_axes = tuple(ax for ax in range(joint.ndim) if ax not in parents)
    return joint.sum(axis=non_parent_axes) if non_parent_axes else joint


def independent_parent_distribution(marginals: list[np.ndarray], parents: tuple[int, ...]) -> np.ndarray:
    """Outer product of the parents' own marginals, i.e. the parent joint
    distribution *assuming* the parents are mutually independent. This is the
    scalability shortcut tree_network.m uses (via independentPD.m) once
    tracking the true running joint becomes infeasible; it trades accuracy
    for a parent-distribution computation whose cost no longer grows with the
    number of already-processed factors."""
    if not parents:
        return np.array(1.0)
    result = marginals[parents[0]]
    for p in parents[1:]:
        result = np.multiply.outer(result, marginals[p])
    return result


def extend_joint(joint: np.ndarray | None, parents: tuple[int, ...], cpt: np.ndarray) -> np.ndarray:
    """Fold a newly fitted CPT (shape (n_states_new,) + parent_state_dims,
    parents given in increasing factor-index order) into the running joint,
    appending the new factor as the last axis. Broadcasts across any
    already-processed factor that isn't among `parents`, since the CPT makes
    the new factor conditionally independent of everything else given its
    parents (matches updatePD.m)."""
    n_new = cpt.shape[0]
    if joint is None:
        return cpt.reshape(n_new)

    n_prior = joint.ndim
    full_shape = [1] * (n_prior + 1)
    full_shape[0] = n_new
    for cpt_axis, p in enumerate(parents, start=1):
        full_shape[p + 1] = cpt.shape[cpt_axis]
    cpt_broadcastable = cpt.reshape(full_shape)

    extended = joint[np.newaxis, ...] * cpt_broadcastable
    return np.moveaxis(extended, 0, -1)


def cross_impacts_from_joint(joint: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """The exact pairwise cross-impact multipliers implied by a fully known
    joint distribution (generalizes eq. 15's "implied cross-impacts" from a
    fitted q(s) to any joint). Useful as ground truth in tests: generate a
    random joint, derive its marginals and cross-impacts, and check that
    fitting a network back from those estimates recovers a consistent result."""
    n = joint.ndim
    marginals = [joint.sum(axis=tuple(a for a in range(n) if a != i)) for i in range(n)]
    cross_impacts: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair_joint = joint.sum(axis=tuple(a for a in range(n) if a not in (i, j)))
            outer = np.outer(marginals[i], marginals[j])
            cross_impacts[(i, j)] = np.divide(
                pair_joint, outer, out=np.ones_like(pair_joint), where=outer > 0
            )
    return cross_impacts, marginals


def _conditional_from_joint(joint: np.ndarray, own_axis: int, parent_axes: tuple[int, ...]) -> np.ndarray:
    """Extract q(own | parents) from `joint`, marginalizing out every axis
    not in {own_axis} | parent_axes and reordering to the (own-state-first)
    CPT convention used throughout this package, regardless of how
    `own_axis`/`parent_axes` happen to be ordered within `joint`."""
    keep = (own_axis, *parent_axes)
    other = tuple(a for a in range(joint.ndim) if a not in keep)
    restricted = joint.sum(axis=other) if other else joint

    kept_sorted = sorted(keep)
    perm = [kept_sorted.index(own_axis)] + [kept_sorted.index(p) for p in parent_axes]
    restricted = np.transpose(restricted, axes=perm)

    n_own = restricted.shape[0]
    parent_marginal = restricted.sum(axis=0)
    return np.divide(
        restricted,
        parent_marginal[np.newaxis, ...],
        out=np.full_like(restricted, 1.0 / n_own),
        where=parent_marginal[np.newaxis, ...] > 0,
    )


def scenario_probability(built: BuiltNetwork, scenario: tuple[int, ...]) -> float:
    """P(s) via the chain rule (eq. 28), from the fitted CPTs alone -- does
    not require the full joint to have been materialized."""
    factors = built.network.factors
    prob = 1.0
    for i, state in enumerate(scenario):
        parents = tuple(sorted(factors[i].parents))
        index = (state,) + tuple(scenario[p] for p in parents)
        prob *= built.cpts[i][index]
    return prob


def build_network(
    network: CrossImpactNetwork,
    mode: str = "exact",
    loss: str = "least_squares",
    engine: str = "cvxpy",
    initial_joint: np.ndarray | None = None,
) -> BuiltNetwork:
    """Fit every factor's conditional probability table in topological order
    (Section 3.2/3.4 of the paper). mode="exact" tracks the true running
    joint over already-processed factors (as energy_network.m/
    printing_network.m do); mode="independent" instead approximates each
    node's parent distribution as if its parents were mutually independent
    (as tree_network.m does), which is far cheaper but only appropriate when
    the true parent joint is unavailable or infeasible to track. loss/engine
    select the per-node fitting objective/algorithm -- see
    fitting.fit_conditional_distribution for the choices.

    initial_joint: optional (mode="exact" only), seeds the running joint with
        an already-known distribution over a PREFIX of `network.factors` --
        e.g. extracted from an already-learned gum.BayesNet via
        cross_impact.pyagrum_import.marginals_and_joint_from_pyagrum -- so
        those factors are treated as given rather than fit from cross-impact
        estimates. Its `ndim` gives how many leading factors it covers
        (`network.factors[:initial_joint.ndim]`); their conditional
        distributions (given whichever of their declared `parents` are also
        within this prefix) are derived directly from `initial_joint` rather
        than fit, and no cross_impacts entries are needed for pairs that are
        both within the prefix.
    """
    from cross_impact.fitting import fit_conditional_distribution

    if mode not in ("exact", "independent"):
        raise ValueError(f"unknown mode {mode!r}")
    if initial_joint is not None and mode != "exact":
        raise ValueError('initial_joint requires mode="exact"')

    if initial_joint is not None:
        joint: np.ndarray | None = np.asarray(initial_joint, dtype=float)
        n_given = joint.ndim
        for i in range(n_given):
            other_axes = tuple(a for a in range(joint.ndim) if a != i)
            implied_marginal = joint.sum(axis=other_axes)
            if not np.allclose(implied_marginal, network.marginals[i], atol=1e-6):
                raise ValueError(
                    f"initial_joint's marginal for factor {i} ({network.factors[i].name!r}) "
                    f"doesn't match network.marginals[{i}]"
                )
    else:
        joint = None
        n_given = 0

    cpts: list[np.ndarray] = []
    for i, factor in enumerate(network.factors):
        parents = tuple(sorted(factor.parents))
        if any(p >= i for p in parents):
            raise ValueError(f"factor {i} ({factor.name!r}) has a parent that isn't already processed")

        if i < n_given:
            cpts.append(_conditional_from_joint(joint, own_axis=i, parent_axes=parents))
            continue

        if mode == "exact":
            parent_joint = parent_joint_distribution(joint, parents)
        else:
            parent_joint = independent_parent_distribution(network.marginals, parents)

        cpt = fit_conditional_distribution(
            marginal=network.marginals[i],
            parent_marginals=[network.marginals[p] for p in parents],
            parent_joint=parent_joint,
            cross_impacts=[network.cross_impact(i, p) for p in parents],
            loss=loss,
            engine=engine,
        )
        cpts.append(cpt)

        if mode == "exact":
            joint = extend_joint(joint, parents, cpt)

    return BuiltNetwork(network=network, cpts=cpts, joint=joint)
