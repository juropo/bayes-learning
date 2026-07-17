"""Port of randomized_data.m: cross-impacts only pin down pairwise marginals,
not the full joint, so fitting from them is inherently lossy and the fitted
result can depend on the order factors are processed in. This checks that
the loss stays small (as the paper's Appendix B reports: ~0.01-0.03 total
absolute difference across scenarios) rather than blowing up, by comparing
(a) two different topological orderings of the same fully-connected network,
and (b) the iterative per-node fit (build_network) against the direct
whole-joint fit (fit_joint_distribution, eqs. 29-31), which optimizes all
scenario probabilities at once instead of one node's CPT at a time.

Sized small (4 factors x 3 states) to keep the test fast -- a fully-connected
network's last node has n_factors - 1 parents, so the per-node QPs would
otherwise grow quickly.
"""

import numpy as np

from cross_impact import (
    CrossImpactNetwork,
    UncertaintyFactor,
    build_network,
    cross_impacts_from_joint,
    fit_joint_distribution,
)

N_FACTORS = 4
N_STATES = 3


def _random_ground_truth(seed: int):
    rng = np.random.default_rng(seed)
    joint = rng.exponential(1.0, size=(N_STATES,) * N_FACTORS)
    return joint / joint.sum()


def _fully_connected_network(order, marginals, cross_impacts):
    """order: processing order as a permutation of the ground-truth factor
    indices 0..N_FACTORS-1. Every factor depends on every factor before it
    in `order` (matches randomized_data.m's nodes(i).parents=1:i-1)."""
    position = {orig: new for new, orig in enumerate(order)}
    factors = [
        UncertaintyFactor(str(orig), N_STATES, tuple(sorted(position[o] for o in order[:new])))
        for new, orig in enumerate(order)
    ]
    new_marginals = [marginals[orig] for orig in order]
    new_cross_impacts = {}
    for (i, j), block in cross_impacts.items():
        ni, nj = position[i], position[j]
        lo, hi = (ni, nj) if ni < nj else (nj, ni)
        new_cross_impacts[(lo, hi)] = block if ni < nj else block.T
    return CrossImpactNetwork(factors=factors, marginals=new_marginals, cross_impacts=new_cross_impacts), position


def _joint_in_ground_truth_order(joint, position):
    perm = [position[i] for i in range(N_FACTORS)]
    return np.transpose(joint, axes=perm)


def test_fitted_joint_is_a_valid_distribution():
    truth = _random_ground_truth(seed=0)
    cross_impacts, marginals = cross_impacts_from_joint(truth)
    network, _ = _fully_connected_network(list(range(N_FACTORS)), marginals, cross_impacts)

    built = build_network(network, mode="exact")

    assert built.joint.sum() == 1.0 or abs(built.joint.sum() - 1.0) < 1e-6
    assert np.all(built.joint >= -1e-9)


def test_processing_order_has_only_a_small_effect():
    truth = _random_ground_truth(seed=1)
    cross_impacts, marginals = cross_impacts_from_joint(truth)

    order_a = list(range(N_FACTORS))
    order_b = [2, 0, 3, 1]

    net_a, pos_a = _fully_connected_network(order_a, marginals, cross_impacts)
    net_b, pos_b = _fully_connected_network(order_b, marginals, cross_impacts)

    joint_a = _joint_in_ground_truth_order(build_network(net_a, mode="exact").joint, pos_a)
    joint_b = _joint_in_ground_truth_order(build_network(net_b, mode="exact").joint, pos_b)

    total_abs_diff = np.abs(joint_a - joint_b).sum()
    # Appendix B reports 0.013-0.027 total absolute difference for an
    # 8-factor/3-state network (6561 scenarios); with only 4 factors here (81
    # scenarios, kept small for test speed) there's much less redundancy to
    # average discrepancies out over, so the gap is larger -- but it should
    # still be a small fraction of the maximum possible total absolute
    # difference between two distributions (2.0).
    assert total_abs_diff < 0.5


def test_iterative_fit_agrees_with_direct_joint_fit():
    truth = _random_ground_truth(seed=2)
    cross_impacts, marginals = cross_impacts_from_joint(truth)
    network, _ = _fully_connected_network(list(range(N_FACTORS)), marginals, cross_impacts)

    iterative = build_network(network, mode="exact").joint
    direct = fit_joint_distribution(network)

    total_abs_diff = np.abs(iterative - direct).sum()
    assert total_abs_diff < 0.5
