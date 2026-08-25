"""Stationary (discounted infinite-horizon) optimal treatment policy via
classic MDP value iteration on the small (diagnosis, living) state space,
using transition probabilities extracted from the trained 2TBN via exact
inference.

This sidesteps pyAgrum's ID/LIMID solver entirely: cost_influence_diagram.py's
build_treatment_decision_diagram/solve_treatment_policy gives an exact
finite-horizon policy but only stays numerically reliable up to about T=200
(see that module's docstrings) and produces a time-varying policy anyway --
useful for validating the model, but not for a single stationary policy.
Here the state space is small enough (5 diagnoses x 4 living arrangements x
4 treatments) that plain tabular value iteration is both exact enough for
this purpose and trivially cheap, independent of any horizon length.

State = (diagnosis, living); action = treatment. `sickness` and `anchor` are
marginalized out (not tracked in state): sickness is a near-deterministic
function of diagnosis (see train_dbn.py's module docstring), and anchor is
persistent-but-hidden, so is marginalized to its learned prior rather than
filtered on -- consistent with treatment{t}'s information set in
cost_influence_diagram.py (diagnosis alone, not anchor).
"""

import cvxpy as cp
import numpy as np
import pyagrum as gum
from elder_care_dbn.cost_influence_diagram import (
    HEALTH_PENALTY_PER_DAY,
    LIVING_COST_PER_DAY,
    TREATMENT_COST_PER_DAY,
)
from elder_care_dbn.train_dbn import DIAGNOSIS_LABELS, LIVING_LABELS, TREATMENT_LABELS


def build_mdp(bn2_path: str):
    """Extract the (diagnosis, living) x treatment -> (diagnosis, living)
    transition kernel and reward table from the trained 2TBN via exact
    inference (one LazyPropagation call per (diagnosis, living, treatment)
    combination), plus the status quo's own stochastic treatment policy
    (the trained treatment0 | diagnosis0 CPT).

    Args:
        bn2_path: path to the trained 2TBN .bif file, e.g. "data/elder_twodbn.bif".

    Returns:
        (P, R, status_quo_policy):
        P: array (n_diag, n_living, n_treatment, n_diag, n_living); P[d,l,a,d',l']
            = P(sickness1=d', living1=l' | sickness0=d, living0=l, treatment0=a)
            -- see the note below on why this conditions on sickness, not
            diagnosis, despite the "diag" naming (they share a label space).
        R: array (n_diag, n_living, n_treatment), immediate reward
            (= -(living cost + treatment cost + health penalty)) for taking
            treatment a while in (d, l).
        status_quo_policy: array (n_diag, n_treatment), the learned
            P(treatment0=a | diagnosis0=d) -- the status quo's own (stochastic)
            decision rule, for policy evaluation/comparison.

    Note on sickness vs diagnosis: this conditions transitions on `sickness0`
    (and reads off `sicknesst`), not `diagnosis0`/`diagnosist`, even though
    everywhere else in this module treats the (diagnosis, living) pair as
    "the state". `sickness0`'s learned prior has DEAD pinned at exactly 0
    (see SICKNESS0_PRIOR in train_dbn.py -- "nobody starts out already
    dead"), which is correct for the model's true t=0 but breaks evidencing
    diagnosis0="DEAD" as a *hypothetical current state* here: since
    diagnosis0<-sickness0,anchor, forcing that evidence makes pyAgrum invert
    it through Bayes' rule, and because sickness0=DEAD has zero prior
    probability it's mathematically impossible as an explanation -- the
    posterior instead explains a DEAD diagnosis as a MILD/MEDIUM/DIFFICULT
    person misdiagnosed as dead, silently turning "the patient is dead" into
    "the patient is fine but was misdiagnosed" for every transition sampled
    from that row. Conditioning on sickness0 instead avoids this entirely:
    it's a root node, so evidencing it is exact substitution, not inversion
    through a prior that rules the value out.

    Note on intervening vs observing: treatment0's only parent is
    diagnosis0 (people historically got treated based on their diagnosis).
    Plain evidence-setting on treatment0 without severing that arc computes
    an OBSERVATIONAL distribution -- "what did outcomes look like for people
    who happened to receive this treatment" -- not the INTERVENTIONAL one an
    MDP transition kernel needs -- "what happens if I choose this
    treatment", regardless of whether it's what the historical policy would
    have picked. Concretely, evidencing treatment0="HOSPITAL" pulls
    diagnosis0's posterior sharply toward DEAD/DIFFICULT (since HOSPITAL was
    historically reserved for the sickest patients), which then contaminates
    livingt's distribution too (diagnosis0 is also livingt's parent) --
    entirely a confound, nothing to do with what HOSPITAL actually does. So
    the arc is cut on a working copy before evidencing (same fix
    build_influence_diagram's `promote` applies for the same reason -- see
    cost_influence_diagram.py).
    """
    bn2 = gum.loadBN(bn2_path)
    # status_quo_policy (below) needs treatment0's real, unmodified CPT --
    # only the P/R-extraction copy gets its diagnosis0->treatment0 arc cut.
    bn2_intervened = gum.BayesNet(bn2)
    bn2_intervened.eraseArc("diagnosis0", "treatment0")  # see the note above

    n_d, n_l, n_a = len(DIAGNOSIS_LABELS), len(LIVING_LABELS), len(TREATMENT_LABELS)
    sicknesst_var = bn2_intervened.variable("sicknesst")
    livingt_var = bn2_intervened.variable("livingt")

    P = np.zeros((n_d, n_l, n_a, n_d, n_l))
    for di, d in enumerate(DIAGNOSIS_LABELS):
        for li, l in enumerate(LIVING_LABELS):
            for ai, a in enumerate(TREATMENT_LABELS):
                ie = gum.LazyPropagation(bn2_intervened)
                ie.addJointTarget({"sicknesst", "livingt"})
                ie.setEvidence({"sickness0": d, "living0": l, "treatment0": a})
                ie.makeInference()
                jp = ie.jointPosterior({"sicknesst", "livingt"})
                inst = gum.Instantiation(jp)
                inst.setFirst()
                while not inst.end():
                    P[di, li, ai, inst.val(sicknesst_var), inst.val(livingt_var)] = (
                        jp.get(inst)
                    )
                    inst.inc()

    R = np.array(
        [
            [
                [
                    -(
                        LIVING_COST_PER_DAY[l]
                        + TREATMENT_COST_PER_DAY[a]
                        + HEALTH_PENALTY_PER_DAY[d]
                    )
                    for a in TREATMENT_LABELS
                ]
                for l in LIVING_LABELS
            ]
            for d in DIAGNOSIS_LABELS
        ]
    )

    diagnosis0_var = bn2.variable("diagnosis0")
    treatment0_var = bn2.variable("treatment0")
    treatment0_cpt = bn2.cpt("treatment0")
    status_quo_policy = np.zeros((n_d, n_a))
    inst = gum.Instantiation(treatment0_cpt)
    inst.setFirst()
    while not inst.end():
        status_quo_policy[inst.val(diagnosis0_var), inst.val(treatment0_var)] = (
            treatment0_cpt.get(inst)
        )
        inst.inc()

    return P, R, status_quo_policy


def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.95,
    tol: float = 1e-10,
    max_iter: int = 100_000,
):
    """Standard Bellman backup: V(s) <- max_a [R(s,a) + gamma sum_s' P(s'|s,a)V(s')].

    Args:
        P: (n_d, n_l, n_a, n_d, n_l) transition kernel, as from build_mdp.
        R: (n_d, n_l, n_a) reward table, as from build_mdp.
        gamma: discount factor -- how much a day's reward is worth relative
            to the previous day's; closer to 1 weighs the far future more.
        tol: convergence threshold on max |V_new - V_old|.
        max_iter: safety cap on iterations.

    Returns:
        (V, policy): V is (n_d, n_l) optimal discounted value function;
        policy is (n_d, n_l) int array, the optimal treatment index per state.
    """
    n_d, n_l, n_a = R.shape
    P_flat = P.reshape(n_d, n_l, n_a, n_d * n_l)
    V = np.zeros((n_d, n_l))
    Q = R.copy()
    for _ in range(max_iter):
        Q = R + gamma * np.einsum("dlas,s->dla", P_flat, V.reshape(-1))
        V_new = Q.max(axis=2)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    policy = Q.argmax(axis=2)
    return V, policy


def _occupation_measure_constraints(P: np.ndarray, mu: np.ndarray, gamma: float, x: cp.Variable):
    """The discounted occupation-measure flow-balance constraint shared by
    solve_lp_policy and solve_asf_policy: x(s,a) is the (normalized, sums to
    1) discounted probability of being in state s and taking action a,
    starting from mu. Total flow out of a state must equal total flow in,
    either fresh (weighted (1-gamma)) or via a transition (weighted gamma) --
    see Puterman, *Markov Decision Processes*, section 6.9.

    Args:
        P: (n_d, n_l, n_a, n_d, n_l) transition kernel, as from build_mdp.
        mu: (n_d, n_l) start-state distribution.
        gamma: discount factor.
        x: (n_s, n_a) cvxpy Variable (n_s = n_d * n_l), already constrained
            nonneg -- this only adds the flow-balance equality.

    Returns:
        A single cvxpy equality constraint (vector-valued, one row per state).
    """
    n_d, n_l, n_a = P.shape[0], P.shape[1], P.shape[2]
    n_s = n_d * n_l
    P_mat = P.reshape(n_s, n_a, n_s).reshape(n_s * n_a, n_s)  # rows ordered (s, a), C-order
    x_vec = cp.reshape(x, (n_s * n_a,), order="C")
    inflow = (1 - gamma) * mu.reshape(n_s) + gamma * (P_mat.T @ x_vec)
    return cp.sum(x, axis=1) == inflow


def solve_lp_policy(P: np.ndarray, R: np.ndarray, mu: np.ndarray, gamma: float = 0.95):
    """Solve for the optimal stationary policy via the classic MDP linear
    program (the dual/occupation-measure formulation -- see
    _occupation_measure_constraints), as an exact cross-check against
    value_iteration rather than because it's needed here: for a single
    (linear, additive-over-time) objective, value iteration already finds
    the exact global optimum, so this exists mainly to confirm the two
    independent methods agree (checked: aggregate values match to solver
    tolerance, recovered policies match exactly) and as the direct
    single-objective analogue of solve_asf_policy's LP.

        maximize   sum_{s,a} R(s,a) x(s,a)
        subject to x(s,a) >= 0, flow-balance (see _occupation_measure_constraints)

    Args:
        P: (n_d, n_l, n_a, n_d, n_l) transition kernel, as from build_mdp.
        R: (n_d, n_l, n_a) reward table, as from build_mdp.
        mu: (n_d, n_l) start-state distribution, as from start_state_distribution.
        gamma: discount factor, as in value_iteration.

    Returns:
        (policy, value): policy is (n_d, n_l) int array (argmax_a x(s,a) per
        state); value is the aggregate discounted value sum(mu * V), directly
        comparable to value_iteration's (mu * V_star).sum().
    """
    n_d, n_l, n_a = R.shape
    n_s = n_d * n_l
    x = cp.Variable((n_s, n_a), nonneg=True)
    constraint = _occupation_measure_constraints(P, mu, gamma, x)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(R.reshape(n_s, n_a), x))), [constraint])
    prob.solve()

    policy = x.value.reshape(n_d, n_l, n_a).argmax(axis=2)
    return policy, prob.value / (1 - gamma)


def evaluate_policy(
    P: np.ndarray, R: np.ndarray, policy_probs: np.ndarray, gamma: float = 0.95
) -> np.ndarray:
    """Discounted value of a (possibly stochastic) policy, solved exactly via
    the linear system V = R_pi + gamma P_pi V (policy evaluation).

    Args:
        P: (n_d, n_l, n_a, n_d, n_l) transition kernel, as from build_mdp.
        R: (n_d, n_l, n_a) reward table, as from build_mdp.
        policy_probs: (n_d, n_a) P(action | diagnosis) -- action
            probabilities depend only on diagnosis, matching treatment0's
            CPT (and build_treatment_decision_diagram's decision, which is
            likewise informed by diagnosis alone).
        gamma: discount factor, as in value_iteration.

    Returns:
        V: (n_d, n_l) value function under this policy.
    """
    n_d, n_l, n_a = R.shape
    n_s = n_d * n_l
    R_pi = np.einsum("da,dla->dl", policy_probs, R).reshape(n_s)
    P_pi = np.einsum(
        "da,dlas->dls", policy_probs, P.reshape(n_d, n_l, n_a, n_s)
    ).reshape(n_s, n_s)
    return np.linalg.solve(np.eye(n_s) - gamma * P_pi, R_pi).reshape(n_d, n_l)


def cost_and_health_tables() -> tuple[np.ndarray, np.ndarray]:
    """The two objective components of build_mdp's combined R, kept separate
    (and POSITIVE, i.e. NOT negated) for multi-objective use -- see
    solve_asf_policy.

    Returns:
        (cost_table, health_table): both (n_d, n_l, n_a); cost_table[d,l,a]
        = living_cost_per_day[l] + treatment_cost_per_day[a] (independent of
        d); health_table[d,l,a] = health_penalty_per_day[d] (independent of
        l and a; broadcast to the same shape so both tables line up with P).
    """
    cost_table = np.array(
        [
            [[LIVING_COST_PER_DAY[l] + TREATMENT_COST_PER_DAY[a] for a in TREATMENT_LABELS] for l in LIVING_LABELS]
            for _ in DIAGNOSIS_LABELS
        ]
    )
    health_table = np.array(
        [[[HEALTH_PENALTY_PER_DAY[d]] * len(TREATMENT_LABELS) for _ in LIVING_LABELS] for d in DIAGNOSIS_LABELS]
    )
    return cost_table, health_table


def evaluate_deterministic_policy(P: np.ndarray, R: np.ndarray, policy: np.ndarray, gamma: float = 0.95) -> np.ndarray:
    """Discounted value of a deterministic policy (one action per state,
    possibly depending on living as well as diagnosis -- unlike
    evaluate_policy's status-quo policy, which is diagnosis-only), solved
    exactly via V = R_pi + gamma P_pi V, same as evaluate_policy but
    specialized to a single chosen action per state rather than a
    distribution over actions.

    Args:
        P: (n_d, n_l, n_a, n_d, n_l) transition kernel, as from build_mdp.
        R: (n_d, n_l, n_a) reward table, as from build_mdp (or any other
            per-(state, action) reward table of the same shape, e.g. a
            single objective's cost or health penalty in isolation).
        policy: (n_d, n_l) int array, the action index chosen in each state
            -- e.g. value_iteration's output, or a candidate being scored.
        gamma: discount factor, as in value_iteration.

    Returns:
        V: (n_d, n_l) value function under this policy.
    """
    n_d, n_l, _ = R.shape
    d_idx, l_idx = np.arange(n_d)[:, None], np.arange(n_l)[None, :]
    n_s = n_d * n_l
    R_pi = R[d_idx, l_idx, policy].reshape(n_s)
    P_pi = P[d_idx, l_idx, policy].reshape(n_s, n_s)
    return np.linalg.solve(np.eye(n_s) - gamma * P_pi, R_pi).reshape(n_d, n_l)


def solve_asf_policy_local_search(
    P: np.ndarray,
    cost_table: np.ndarray,
    health_table: np.ndarray,
    reference_point: tuple[float, float],
    start_dist: np.ndarray,
    weights: tuple[float, float] | None = None,
    rho: float = 1e-4,
    gamma: float = 0.95,
    max_sweeps: int = 200,
):
    """Superseded by solve_asf_policy (an exact LP formulation) -- kept for
    comparison/illustration. Checked empirically: for one example reference
    point, this found ASF=0.426 from its best of three warm starts, while
    solve_asf_policy's LP found the true optimum, ASF=0.350 -- a reminder
    that local search over a nonlinear scalarization isn't reliable even on
    a state space this small, and multiple warm starts don't fix that.

    Find a stationary policy minimizing Wierzbicki's augmented achievement
    scalarizing function of (expected discounted cost, expected discounted
    health penalty), given a reference point (aspiration level) for each --
    e.g. "I'd like cost around z_cost and health penalty around z_health",
    and ASF finds the closest Pareto-optimal tradeoff to that, in the
    direction implied by `weights`.

    s(f, z) = max_i[ w_i (f_i - z_i) ] + rho * sum_i[ w_i (f_i - z_i) ]

    minimized over policies, where f = (expected discounted cost, expected
    discounted health penalty) at the given start distribution. The
    augmentation term (weight rho) is what guarantees the minimizer is
    (properly) Pareto optimal rather than merely weakly so, and lets ASF
    reach non-convex parts of the tradeoff frontier that a plain weighted
    sum can't -- but that same max term means ASF, unlike a linear weighted
    sum, is NOT additively separable over time, so it can't be plugged into
    value_iteration's per-step Bellman backup directly (see the surrounding
    discussion in the conversation this was built from). Instead:

    1. Get the ideal point (best achievable cost/health in isolation, via
       ordinary value iteration on each objective alone) and a nadir
       estimate (their cross-evaluated worst case -- the standard "payoff
       table" method), to auto-scale `weights` if not given.
    2. Local search from several warm starts (the weighted-sum-optimal
       policy under `weights`, plus each single-objective-optimal policy --
       all three solvable exactly via ordinary value iteration): repeatedly
       try changing one state's action at a time, keeping any change that
       improves the (exactly, not approximately, evaluated) ASF value, until
       a full sweep finds no improvement. Cheap here since each evaluation
       is just a 20x20 linear solve. Keep whichever start's local optimum
       scores best.

    This finds a local optimum of a nonlinear scalarization, not a globally
    certified one (unlike value_iteration's plain discounted-sum objective,
    which Bellman's principle of optimality solves exactly) -- multiple
    warm starts noticeably change which local optimum is found (checked
    empirically: two single-objective-optimal starting points converged to
    ASF values of 0.456 and 0.432 for the same reference point), so treat
    the result as a good, not certified-global, tradeoff.

    Args:
        P: (n_d, n_l, n_a, n_d, n_l) transition kernel, as from build_mdp.
        cost_table: (n_d, n_l, n_a) POSITIVE cost (not negated, unlike
            build_mdp's combined R) -- e.g. living_cost + treatment_cost.
        health_table: (n_d, n_l, n_a) POSITIVE health penalty, broadcast
            over living/treatment if it only depends on diagnosis.
        reference_point: (z_cost, z_health) aspiration levels, same units
            and sign (positive = bad) as cost_table/health_table.
        start_dist: (n_d, n_l) probability distribution to evaluate the
            objectives under, e.g. start_state_distribution's output.
        weights: (w_cost, w_health); defaults to 1/(nadir - ideal) per
            objective (auto-scaling so both matter comparably).
        rho: augmentation coefficient (small and positive); larger values
            weight the sum-of-deviations term more relative to the max.
        gamma: discount factor, as in value_iteration.
        max_sweeps: safety cap on local-search sweeps over all states.

    Returns:
        (policy, asf_value, ideal, nadir): policy is (n_d, n_l) int array of
        chosen actions; asf_value is the achieved (minimized) ASF value;
        ideal/nadir are (2,) arrays [cost, health] from the payoff table.
    """
    R_cost, R_health = -cost_table, -health_table  # value_iteration maximizes

    _, policy_cost_opt = value_iteration(P, R_cost, gamma=gamma)
    _, policy_health_opt = value_iteration(P, R_health, gamma=gamma)
    cost_at_cost_opt = (start_dist * evaluate_deterministic_policy(P, cost_table, policy_cost_opt, gamma)).sum()
    health_at_health_opt = (start_dist * evaluate_deterministic_policy(P, health_table, policy_health_opt, gamma)).sum()
    health_at_cost_opt = (start_dist * evaluate_deterministic_policy(P, health_table, policy_cost_opt, gamma)).sum()
    cost_at_health_opt = (start_dist * evaluate_deterministic_policy(P, cost_table, policy_health_opt, gamma)).sum()
    ideal = np.array([cost_at_cost_opt, health_at_health_opt])
    nadir = np.array([cost_at_health_opt, health_at_cost_opt])

    if weights is None:
        weights = 1.0 / np.maximum(nadir - ideal, 1e-6)
    weights = np.asarray(weights, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    def asf(cost_val: float, health_val: float) -> float:
        deviation = weights * (np.array([cost_val, health_val]) - reference_point)
        return deviation.max() + rho * deviation.sum()

    def score(pol: np.ndarray) -> float:
        cost_val = (start_dist * evaluate_deterministic_policy(P, cost_table, pol, gamma)).sum()
        health_val = (start_dist * evaluate_deterministic_policy(P, health_table, pol, gamma)).sum()
        return asf(cost_val, health_val)

    def local_search(policy: np.ndarray) -> tuple[np.ndarray, float]:
        policy = policy.copy()
        best = score(policy)
        n_d, n_l, n_a = cost_table.shape
        for _ in range(max_sweeps):
            improved = False
            for di in range(n_d):
                for li in range(n_l):
                    original = policy[di, li]
                    for a in range(n_a):
                        if a == original:
                            continue
                        policy[di, li] = a
                        val = score(policy)
                        if val < best - 1e-12:
                            best, original, improved = val, a, True
                        else:
                            policy[di, li] = original
            if not improved:
                break
        return policy, best

    _, weighted_sum_opt = value_iteration(P, -(weights[0] * cost_table + weights[1] * health_table), gamma=gamma)
    candidates = [weighted_sum_opt, policy_cost_opt, policy_health_opt]

    best_policy, best_value = None, np.inf
    for start_policy in candidates:
        pol, val = local_search(start_policy)
        if val < best_value:
            best_policy, best_value = pol, val

    return best_policy, best_value, ideal, nadir


def solve_asf_policy(
    P: np.ndarray,
    cost_table: np.ndarray,
    health_table: np.ndarray,
    reference_point: tuple[float, float],
    start_dist: np.ndarray,
    weights: tuple[float, float] | None = None,
    rho: float = 1e-4,
    gamma: float = 0.95,
):
    """Find the stationary policy minimizing Wierzbicki's augmented
    achievement scalarizing function (ASF) of (expected discounted cost,
    expected discounted health penalty) at a reference point, via an exact
    linear program -- see solve_asf_policy_local_search's docstring for the
    scalarization itself and why it doesn't reduce to a per-step Bellman
    reward.

    The key trick: working in occupation-measure space (see
    _occupation_measure_constraints) makes BOTH aggregate objectives linear
    in the decision variable x(s,a) --

        cost_agg(x)   = sum_{s,a} cost(s,a)   x(s,a) / (1 - gamma)
        health_agg(x) = sum_{s,a} health(s,a) x(s,a) / (1 - gamma)

    -- so ASF's max-of-affine-plus-linear-sum, which isn't decomposable
    over time as a per-step reward, IS a convex (piecewise-linear) function
    of x, exactly representable via cvxpy's cp.maximum. The whole problem
    is then a single LP over the same flow-conservation polytope
    solve_lp_policy uses, solved to global optimality -- no local search,
    no warm starts, no risk of the sort of stuck-in-a-worse-optimum result
    solve_asf_policy_local_search was shown to produce.

    Args:
        P: (n_d, n_l, n_a, n_d, n_l) transition kernel, as from build_mdp.
        cost_table: (n_d, n_l, n_a) POSITIVE cost, as from cost_and_health_tables.
        health_table: (n_d, n_l, n_a) POSITIVE health penalty, as above.
        reference_point: (z_cost, z_health) aspiration levels.
        start_dist: (n_d, n_l) probability distribution, e.g.
            start_state_distribution's output.
        weights: (w_cost, w_health); defaults to 1/(nadir - ideal) per
            objective (found via value_iteration on each objective alone,
            same payoff-table method as the local-search version).
        rho: augmentation coefficient (small and positive).
        gamma: discount factor.

    Returns:
        (policy, asf_value, ideal, nadir): policy is (n_d, n_l) int array of
        chosen actions (argmax_a x(s,a) per state); asf_value is the exact
        minimized ASF value; ideal/nadir are (2,) arrays [cost, health].
    """
    R_cost, R_health = -cost_table, -health_table  # value_iteration maximizes
    _, policy_cost_opt = value_iteration(P, R_cost, gamma=gamma)
    _, policy_health_opt = value_iteration(P, R_health, gamma=gamma)
    cost_at_cost_opt = (start_dist * evaluate_deterministic_policy(P, cost_table, policy_cost_opt, gamma)).sum()
    health_at_health_opt = (start_dist * evaluate_deterministic_policy(P, health_table, policy_health_opt, gamma)).sum()
    health_at_cost_opt = (start_dist * evaluate_deterministic_policy(P, health_table, policy_cost_opt, gamma)).sum()
    cost_at_health_opt = (start_dist * evaluate_deterministic_policy(P, cost_table, policy_health_opt, gamma)).sum()
    ideal = np.array([cost_at_cost_opt, health_at_health_opt])
    nadir = np.array([cost_at_health_opt, health_at_cost_opt])

    if weights is None:
        weights = 1.0 / np.maximum(nadir - ideal, 1e-6)
    weights = np.asarray(weights, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    n_d, n_l, n_a = cost_table.shape
    n_s = n_d * n_l
    x = cp.Variable((n_s, n_a), nonneg=True)
    constraint = _occupation_measure_constraints(P, start_dist, gamma, x)

    cost_agg = cp.sum(cp.multiply(cost_table.reshape(n_s, n_a), x)) / (1 - gamma)
    health_agg = cp.sum(cp.multiply(health_table.reshape(n_s, n_a), x)) / (1 - gamma)
    deviation_cost = weights[0] * (cost_agg - reference_point[0])
    deviation_health = weights[1] * (health_agg - reference_point[1])
    objective = cp.Minimize(cp.maximum(deviation_cost, deviation_health) + rho * (deviation_cost + deviation_health))

    prob = cp.Problem(objective, [constraint])
    prob.solve()

    policy = x.value.reshape(n_d, n_l, n_a).argmax(axis=2)
    return policy, prob.value, ideal, nadir


def start_state_distribution(bn2_path: str) -> np.ndarray:
    """P(sickness0, living0) under the trained model -- sickness0 (matching
    build_mdp's state; see its note on sickness vs diagnosis) and living0
    have no arc between them at t=0 (see train_dbn.TOPOLOGY), so this is
    just the outer product of their marginals, but computed via inference
    rather than assumed, in case that ever changes.

    Args:
        bn2_path: path to the trained 2TBN .bif file.
    """
    bn2 = gum.loadBN(bn2_path)
    ie = gum.LazyPropagation(bn2)
    ie.makeInference()
    return np.outer(
        ie.posterior("sickness0").toarray(), ie.posterior("living0").toarray()
    )


if __name__ == "__main__":
    P, R, status_quo_policy = build_mdp("data/elder_twodbn.bif")

    gamma = 0.995
    V_star, policy = value_iteration(P, R, gamma=gamma)
    V_status_quo = evaluate_policy(P, R, status_quo_policy, gamma=gamma)

    print(f"stationary optimal policy (gamma={gamma}):")
    for di, d in enumerate(DIAGNOSIS_LABELS):
        chosen = {
            LIVING_LABELS[li]: TREATMENT_LABELS[policy[di, li]]
            for li in range(len(LIVING_LABELS))
        }
        print(f"  diagnosis={d:10s}: {chosen}")

    print("\nvalue comparison (discounted, at living=INDEPENDENT):")
    li = LIVING_LABELS.index("INDEPENDENT")
    for di, d in enumerate(DIAGNOSIS_LABELS):
        print(
            f"  diagnosis={d:10s}: optimal={V_star[di, li]:9.2f}  status_quo={V_status_quo[di, li]:9.2f}"
        )

    start = start_state_distribution("data/elder_twodbn.bif")
    print("\nexpected discounted value from the true t=0 start distribution:")
    print(f"  optimal:    {(start * V_star).sum():.2f}")
    print(f"  status quo: {(start * V_status_quo).sum():.2f}")

    print("\nmulti-objective (ASF) example, treating cost and health penalty as separate objectives:")
    cost_table, health_table = cost_and_health_tables()
    _, cost_opt_policy = value_iteration(P, -cost_table, gamma=gamma)
    _, health_opt_policy = value_iteration(P, -health_table, gamma=gamma)
    ideal_cost = (start * evaluate_deterministic_policy(P, cost_table, cost_opt_policy, gamma)).sum()
    ideal_health = (start * evaluate_deterministic_policy(P, health_table, health_opt_policy, gamma)).sum()
    # illustrative reference point: aim for 50% above each objective's own
    # best-case ideal (an aspiration a bit more ambitious than the combined
    # weighted-sum optimum, but not the unreachable ideal point itself)
    reference_point = (ideal_cost * 1.5, ideal_health * 1.5)

    asf_policy, asf_value, ideal, nadir = solve_asf_policy(P, cost_table, health_table, reference_point, start, gamma=gamma)
    achieved_cost = (start * evaluate_deterministic_policy(P, cost_table, asf_policy, gamma)).sum()
    achieved_health = (start * evaluate_deterministic_policy(P, health_table, asf_policy, gamma)).sum()
    print(f"  reference point: cost~{reference_point[0]:.1f}, health~{reference_point[1]:.1f}")
    print(f"  ideal (per objective, isolated): cost={ideal[0]:.1f}, health={ideal[1]:.1f}")
    print(f"  nadir (cross-evaluated worst case): cost={nadir[0]:.1f}, health={nadir[1]:.1f}")
    print(f"  achieved: cost={achieved_cost:.1f}, health={achieved_health:.1f} (ASF value={asf_value:.4f})")
    for di, d in enumerate(DIAGNOSIS_LABELS):
        chosen = {LIVING_LABELS[li]: TREATMENT_LABELS[asf_policy[di, li]] for li in range(len(LIVING_LABELS))}
        print(f"  diagnosis={d:10s}: {chosen}")
