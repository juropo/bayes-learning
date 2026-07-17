"""Port of tree_network.m / Appendix B: a scalability stress test on a
3-regular directed tree of depth 4 (121 factors, each with 3 outcomes; see
Figure B1 in the paper). Marginals and cross-impacts are random -- the point
isn't correctness of the output (the paper says as much: "both the
parameters and the output of the computation are nonsense") but that the
network can be built at all, quickly, once conditional independence keeps
each node's fitting problem small regardless of how many factors precede it.

Matches tree_network.m's shortcut of using independentPD.m instead of the
true running joint (mode="independent" here): tracking an exact joint over
121 binary/ternary factors is infeasible (3**121 possible scenarios), so
each node's parent distribution is approximated as the outer product of its
parents' own marginals.
"""

import time

import numpy as np

from cross_impact import CrossImpactNetwork, UncertaintyFactor, build_network


def _tree_structure(n: int = 121) -> list[tuple[int, ...]]:
    """3-regular directed tree of depth 4: nodes 0-80 are leaves; 81-107
    aggregate triples of 0-80; 108-116 aggregate triples of 81-107; 117-119
    aggregate triples of 108-116; 120 aggregates 117-119. Returns, for each
    node, the tuple of nodes that point up to it (its "parents")."""
    child_of = [None] * n
    for ci in range(81):
        child_of[ci] = 81 + ci // 3
    for i in range(27):
        child_of[81 + i] = 81 + 27 + i // 3
    for i in range(9):
        child_of[81 + 27 + i] = 81 + 27 + 9 + i // 3
    for i in range(3):
        child_of[81 + 27 + 9 + i] = 81 + 27 + 9 + 3
    return [tuple(j for j in range(n) if child_of[j] == i) for i in range(n)]


def _random_tree_network(seed: int = 0, n: int = 121) -> CrossImpactNetwork:
    rng = np.random.default_rng(seed)
    parents = _tree_structure(n)
    factors = [UncertaintyFactor(str(i), 3, parents[i]) for i in range(n)]

    marginals = []
    for _ in range(n):
        p01 = rng.random(2) / 2.01
        marginals.append(np.array([p01[0], p01[1], 1 - p01.sum()]))

    cross_impacts = {}
    for i in range(n):
        for p in parents[i]:
            lo, hi = (i, p) if i < p else (p, i)
            ordinal = rng.integers(-3, 4, size=(3, 3)).astype(float)
            multiplier = np.sqrt(2.0) ** ordinal
            cross_impacts[(lo, hi)] = multiplier if lo == i else multiplier.T

    return CrossImpactNetwork(factors=factors, marginals=marginals, cross_impacts=cross_impacts)


def test_builds_121_factor_tree_quickly():
    network = _random_tree_network()

    start = time.time()
    built = build_network(network, mode="independent")
    elapsed = time.time() - start

    assert elapsed < 30, f"took {elapsed:.1f}s -- conditional independence should keep this fast"
    assert len(built.cpts) == len(network.factors)


def test_every_cpt_is_a_valid_conditional_distribution():
    # Tolerances are solver precision, not correctness margin: cvxpy's default
    # solver for this problem shape converges to constraint satisfaction
    # within ~1e-4, not to machine precision.
    built = build_network(_random_tree_network(seed=1), mode="independent")
    for factor, cpt in zip(built.network.factors, built.cpts):
        assert cpt.shape[0] == factor.n_states
        assert np.all(cpt >= -1e-4)
        assert np.allclose(cpt.sum(axis=0), 1.0, atol=1e-3)
