"""Iterative Proportional Fitting (IPF, aka raking) as a fast, dependency-free
alternative to the cvxpy exponential-cone solve for loss="kl" in fitting.py.

IPF fits a nonnegative table to a set of target marginals by cyclically
rescaling the table, for each target in turn, so that the marginal it
constrains matches that target exactly (a multiplicative -- i.e. KL --
projection onto that one constraint), then moving on to the next target.
When the targets are jointly achievable, this cyclic projection is known to
converge to the minimum-KL-divergence table (relative to the starting point)
that satisfies all of them exactly (Csiszar's iterative scaling theorem) --
which is exactly what fit_conditional_distribution's/fit_joint_distribution's
"kl" loss is a relaxation of, for the case where the cross-impact-implied
targets are NOT all jointly achievable (they need not be mutually
consistent -- see the paper's Table 3). In that case this cyclic-projection
heuristic no longer has a global-optimality guarantee (it can settle into a
limit cycle rather than a fixed point), but it remains the standard, fast,
numerically robust practical algorithm for this kind of table-fitting
problem, and empirically tracks the cvxpy "kl" solution closely (see
tests/test_ipf.py).

Two constraints are *hard* rather than soft (own-marginal, and the
already-known parent/other-factor joint -- eqs. 25/26, or eq. 30 for the
direct whole-network fit) and must hold exactly, not just approximately.
Naively cycling through the hard and soft targets in one flat rotation
doesn't work: whenever a soft target happens to span every axis of the table
(e.g. a node with a single parent, where "the cross-impact target for that
parent" already covers the whole table -- there's nothing left to
marginalize over), projecting onto it is a full overwrite, not a gentle
nudge, and it would silently undo the hard constraints on every pass rather
than just perturbing them. So instead, the hard constraints are re-imposed
to *full* convergence (their own inner alternating-projection loop) after
every single soft-target nudge, not interleaved as just another item in the
same rotation.
"""

from __future__ import annotations

import numpy as np

from cross_impact.model import CrossImpactNetwork


def _rescale(table: np.ndarray, target: np.ndarray, keep_axes: int | tuple[int, ...]) -> np.ndarray:
    """Multiplicatively rescale `table` so that summing over every axis NOT
    in `keep_axes` reproduces `target` exactly -- a KL projection onto that
    one marginal constraint. `target`'s shape must match `table`'s shape at
    `keep_axes`, in increasing axis order. If `keep_axes` covers every axis
    of `table`, there is nothing to marginalize over and this reduces to
    `table = target`."""
    if isinstance(keep_axes, int):
        keep_axes = (keep_axes,)
    other_axes = tuple(a for a in range(table.ndim) if a not in keep_axes)
    current = table.sum(axis=other_axes) if other_axes else table
    scale = np.divide(target, current, out=np.ones_like(target, dtype=float), where=current > 0)
    shape = [1] * table.ndim
    for ax in keep_axes:
        shape[ax] = table.shape[ax]
    return table * scale.reshape(shape)


def _enforce_hard_marginals(
    table: np.ndarray,
    targets_by_axes: list[tuple[int | tuple[int, ...], np.ndarray]],
    max_iter: int = 200,
    tol: float = 1e-13,
) -> np.ndarray:
    """Alternately project `table` onto each (keep_axes, target) hard
    constraint in turn until all of them hold simultaneously (or `max_iter`
    is reached) -- N-way Sinkhorn/IPF run to convergence, for constraints
    that are known to be jointly achievable (their targets' totals agree)."""
    for _ in range(max_iter):
        prev = table
        for keep_axes, target in targets_by_axes:
            table = _rescale(table, target, keep_axes=keep_axes)
        if np.abs(table - prev).max() < tol:
            break
    return table


def fit_conditional_distribution_ipf(
    marginal: np.ndarray,
    parent_marginals: list[np.ndarray],
    parent_joint: np.ndarray,
    cross_impacts: list[np.ndarray],
    max_iter: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """IPF counterpart to fitting.fit_conditional_distribution(loss="kl"):
    fits the joint table y(own state, parent config) by cyclically rescaling
    towards each parent's cross-impact-implied target -- restoring the two
    hard constraints (own marginal, parent joint) to full convergence after
    each one -- until the table stops changing overall. Returns
    q(k | s_parents), shape (n_states, *parent_state_dims) -- same signature
    and shape contract as fitting.fit_conditional_distribution.
    """
    n_states = marginal.shape[0]
    parent_dims = tuple(pm.shape[0] for pm in parent_marginals)
    parent_joint = np.asarray(parent_joint, dtype=float).reshape(parent_dims)
    parent_axes = tuple(range(1, 1 + len(parent_dims)))
    hard_constraints = [(0, marginal), (parent_axes, parent_joint)]

    targets = [C * marginal[:, None] * p_hat_j[None, :] for C, p_hat_j in zip(cross_impacts, parent_marginals)]

    # Independence starting point: exactly satisfies both hard constraints already.
    y = marginal.reshape((n_states,) + (1,) * len(parent_dims)) * parent_joint.reshape((1, *parent_dims))

    for _ in range(max_iter):
        y_prev = y
        for p_idx, target in enumerate(targets):
            y = _rescale(y, target, keep_axes=(0, parent_axes[p_idx]))
            y = _enforce_hard_marginals(y, hard_constraints)

        if np.abs(y - y_prev).max() < tol:
            break

    x = np.divide(y, parent_joint, out=np.full_like(y, 1.0 / n_states), where=parent_joint > 0)
    return x


def fit_joint_distribution_ipf(network: CrossImpactNetwork, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """IPF counterpart to fitting.fit_joint_distribution(loss="kl"): fits the
    full joint table by cyclically rescaling towards every elicited
    pairwise cross-impact target -- restoring every factor's own marginal to
    full convergence after each one -- until the table stabilizes. Returns
    an array of shape (n_states_0, ..., n_states_{N-1})."""
    dims = [f.n_states for f in network.factors]
    n = len(dims)
    marginals = network.marginals
    hard_constraints = [(i, m) for i, m in enumerate(marginals)]

    q = np.ones(())
    for i, m in enumerate(marginals):
        shape = [1] * n
        shape[i] = dims[i]
        q = q * m.reshape(shape)

    for _ in range(max_iter):
        q_prev = q
        for (i, j), C in network.cross_impacts.items():
            target = C * marginals[i][:, None] * marginals[j][None, :]
            q = _rescale(q, target, keep_axes=(i, j))
            q = _enforce_hard_marginals(q, hard_constraints)

        if np.abs(q - q_prev).max() < tol:
            break

    return q
