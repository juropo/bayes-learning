"""The constrained fits from Roponen & Salo (2024), solved with cvxpy in
place of MATLAB's lsqlin.

fit_conditional_distribution implements the per-node problem (eqs. 11-14,
generalized to arbitrary parent sets by eqs. 24-27) that build_network calls
once per factor. fit_joint_distribution implements the direct whole-network
problem (eqs. 29-31) that Section 3.5 uses only to sanity-check the iterative
approach on small networks, since its variable count is the full product of
all factor state counts.

Both problems are built as matrix expressions (indicator/selection matrices
against a variable, combined via matrix multiplication) rather than by
looping over every (k, l) outcome pair in Python -- both for speed and
because it lets cvxpy's DCP analysis see a small number of large affine
expressions instead of thousands of scalar ones.

Both also accept a `loss` argument choosing how deviation from the
cross-impact-implied targets is penalized:
- "least_squares" (default, matches the paper): sum of squared differences.
- "kl": generalized KL divergence (aka I-divergence) from each target to the
  corresponding implied probability, sum(target*log(target/lhs) - target +
  lhs). Unlike ordinary KL divergence this doesn't require `target` or `lhs`
  to individually sum to 1 (they generally won't -- cross-impact estimates
  need not be mutually consistent, and a target here is only one pairwise
  slice of the fitted distribution), so it's really the elementwise Bregman
  divergence associated with the Poisson/count-data likelihood, applied to
  probability values rather than counts. Compared to least squares, it
  penalizes relative rather than absolute deviation, and it requires a
  solver that supports the exponential cone (cvxpy will pick one
  automatically -- Clarabel or SCS rather than OSQP).

For loss="kl", `engine="ipf"` is also available: Iterative Proportional
Fitting (cross_impact.ipf), a dependency-free multiplicative-update
algorithm that is the classical way to solve exactly this kind of
KL/table-fitting problem, and is typically much faster than routing through
cvxpy's exponential-cone solvers at scale. See cross_impact/ipf.py for how it
relates to (and differs from) the cvxpy "kl" solve.
"""

from __future__ import annotations

import itertools

import cvxpy as cp
import numpy as np

from cross_impact.model import CrossImpactNetwork


def _indicator_matrix(configs: list[tuple[int, ...]], axis: int, n_axis_states: int) -> np.ndarray:
    """Shape (n_axis_states, len(configs)): entry [k, c] is 1 if configs[c]
    has state k along `axis`, else 0."""
    matrix = np.zeros((n_axis_states, len(configs)))
    for c, cfg in enumerate(configs):
        matrix[cfg[axis], c] = 1.0
    return matrix


def _residual_term(lhs: cp.Expression, target: np.ndarray, loss: str) -> cp.Expression:
    if loss == "least_squares":
        return cp.sum_squares(lhs - target)
    if loss == "kl":
        return cp.sum(cp.kl_div(target, lhs))
    raise ValueError(f"unknown loss {loss!r}; expected 'least_squares' or 'kl'")


def _solve(problem: cp.Problem) -> None:
    # cvxpy's default solver choice (e.g. Clarabel for an exponential-cone
    # "kl" problem) is usually fine, but can fail to converge on the larger
    # per-node problems (many parents/states) that the "kl" loss produces.
    # SCS is slower but consistently robust at that scale, so it's the
    # fallback rather than the default.
    try:
        problem.solve()
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return
    except cp.error.SolverError:
        pass
    problem.solve(solver=cp.SCS)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"cvxpy failed to find an optimal solution (status {problem.status!r})")


def fit_conditional_distribution(
    marginal: np.ndarray,
    parent_marginals: list[np.ndarray],
    parent_joint: np.ndarray,
    cross_impacts: list[np.ndarray],
    loss: str = "least_squares",
    engine: str = "cvxpy",
) -> np.ndarray:
    """Fit q(k | s_parents) -- one factor's conditional probability table --
    to minimize deviation from the cross-impact-implied joint probabilities
    Ĉ^ij_kl * p_hat^i_k * p_hat^j_l (summed over every parent j and every
    pair of outcomes k, l), subject to matching the factor's own marginal
    p_hat exactly (eq. 25) and every conditional distribution summing to one
    (eq. 26). See the module docstring for the `loss`/`engine` choices.

    Args:
        marginal: p_hat^i, shape (n_states,).
        parent_marginals: p_hat^j for each parent, in the same order as
            `cross_impacts`.
        parent_joint: q(s_parents), the (already-known) joint distribution
            over the parents, shape matching the product of parent state
            counts (a 0-d array of 1.0 if there are no parents).
        cross_impacts: Ĉ^ij_kl for each parent, shape (n_states, n_states_j).
        loss: "least_squares" (default) or "kl".
        engine: "cvxpy" (default, supports both losses) or "ipf" (requires
            loss="kl"; see cross_impact.ipf).

    Returns:
        Array of shape (n_states, *parent_state_dims).
    """
    if engine == "ipf":
        if loss != "kl":
            raise ValueError('engine="ipf" only supports loss="kl"')
        from cross_impact.ipf import fit_conditional_distribution_ipf

        return fit_conditional_distribution_ipf(marginal, parent_marginals, parent_joint, cross_impacts)
    if engine != "cvxpy":
        raise ValueError(f"unknown engine {engine!r}; expected 'cvxpy' or 'ipf'")

    n_states = marginal.shape[0]
    parent_dims = [pm.shape[0] for pm in parent_marginals]
    n_configs = int(np.prod(parent_dims, dtype=int)) if parent_dims else 1
    configs = list(itertools.product(*(range(d) for d in parent_dims))) or [()]
    parent_weight = np.asarray(parent_joint, dtype=float).reshape(n_configs)

    x = cp.Variable((n_states, n_configs), nonneg=True)

    residual_terms = []
    for p_idx, (C, p_hat_j) in enumerate(zip(cross_impacts, parent_marginals)):
        n_p = parent_dims[p_idx]
        # weighted[l, c] = parent_weight[c] if configs[c] has parent p_idx at
        # state l, else 0 -- so x @ weighted.T sums, per (k, l), q(k|config) *
        # q(config) over exactly the configs where parent p_idx = l.
        weighted = _indicator_matrix(configs, p_idx, n_p) * parent_weight
        lhs = x @ weighted.T  # shape (n_states, n_p)
        target = C * marginal[:, None] * p_hat_j[None, :]
        residual_terms.append(_residual_term(lhs, target, loss))

    objective = cp.Minimize(cp.sum(residual_terms) if residual_terms else 0)
    constraints = [
        cp.sum(x, axis=0) == 1,
        x @ parent_weight == marginal,
    ]
    _solve(cp.Problem(objective, constraints))

    return np.asarray(x.value).reshape((n_states, *parent_dims))


def fit_joint_distribution(network: CrossImpactNetwork, loss: str = "least_squares", engine: str = "cvxpy") -> np.ndarray:
    """Fit q(s) directly over the full scenario space (eqs. 29-31), matching
    every factor's marginal exactly and minimizing deviation from every
    elicited cross-impact estimate. Unlike build_network's per-node iterative
    fit, this needs no parent/topological structure at all, but its variable
    count is the full product of all factor state counts, so it is only
    practical for small networks (used here mainly to check that the
    iterative and direct approaches agree, per Section 3.5). See the module
    docstring for the `loss`/`engine` choices."""
    if engine == "ipf":
        if loss != "kl":
            raise ValueError('engine="ipf" only supports loss="kl"')
        from cross_impact.ipf import fit_joint_distribution_ipf

        return fit_joint_distribution_ipf(network)
    if engine != "cvxpy":
        raise ValueError(f"unknown engine {engine!r}; expected 'cvxpy' or 'ipf'")

    factors = network.factors
    dims = [f.n_states for f in factors]
    configs = list(itertools.product(*(range(d) for d in dims)))

    q = cp.Variable(len(configs), nonneg=True)

    indicators = [_indicator_matrix(configs, i, dims[i]) for i in range(len(factors))]

    residual_terms = []
    for (i, j), C in network.cross_impacts.items():
        # lhs[k, l] = sum_c indicators[i][k, c] * indicators[j][l, c] * q[c]:
        # the joint probability of factor i = k and factor j = l implied by q.
        weighted_i = cp.multiply(indicators[i], q)  # (n_i, n_configs), affine in q
        lhs = weighted_i @ indicators[j].T  # (n_i, n_j)
        target = C * network.marginals[i][:, None] * network.marginals[j][None, :]
        residual_terms.append(_residual_term(lhs, target, loss))

    objective = cp.Minimize(cp.sum(residual_terms) if residual_terms else 0)
    constraints = [indicators[i] @ q == network.marginals[i] for i in range(len(factors))]
    _solve(cp.Problem(objective, constraints))

    return np.asarray(q.value).reshape(dims)
