"""Build an InfluenceDiagram from the elder-care 2TBN, unrolled over T
timesteps, with `treatment{t}` promoted to a decision node and utility nodes
scoring daily cost and health outcome.

Decision/utility construction reuses cross_impact.influence_diagram's
build_influence_diagram rather than hand-rolling pyAgrum node-adding code
(see bn_to_built_network, which adapts an already-trained/unrolled
gum.BayesNet into the BuiltNetwork shape that function expects).

See train_dbn.py for the underlying chance network and label vocabularies.
"""

import itertools

import numpy as np
import pyagrum as gum
import pyagrum.lib.dynamicBN as gdyn
from elder_care_dbn.train_dbn import DIAGNOSIS_LABELS, LIVING_LABELS, TREATMENT_LABELS

from cross_impact.influence_diagram import (
    InfluenceDiagramSpec,
    Utility,
    build_influence_diagram,
)
from cross_impact.model import BuiltNetwork, CrossImpactNetwork, UncertaintyFactor

# Bare labels (no "Unit."/"Sickness." prefix -- see dbn_data_prep._clean_*),
# split per node since living/treatment/diagnosis are different variables
# with different (and, for living/treatment, overlapping but not identical)
# domains.
LIVING_COST_PER_DAY = {
    "INDEPENDENT": 0,
    "HOME_CARE": 20,
    "FAMILY_CARE": 5,
    "HEALTH_CENTRE": 20,
}
TREATMENT_COST_PER_DAY = {
    "NONE": 0,
    "HEALTH_CENTRE": 20,
    "ED": 50,
    "HOSPITAL": 150,
}
# Per-day utility penalty for being in a given diagnosis state, so that a
# cost-only objective can't trivially "optimize" by never treating anyone
# (withholding treatment is free but has no modeled downside without this).
# DEAD is charged per day like any other state (deliberately, not a one-time
# terminal penalty), capped at HOSPITAL's daily cost rather than something
# larger.
HEALTH_PENALTY_PER_DAY = {
    "NONE": 0,
    "MILD": 20,
    "MEDIUM": 60,
    "DIFFICULT": 120,
    "DEAD": 150,
}

OMALAAKARI_COST_PER_DAY = {
    "0": 0,
    "1": 5
}


def _check_cost_table(cost_per_day: dict, labels: list, node_name: str) -> None:
    """Raise ValueError if `cost_per_day`'s keys don't exactly match `labels`.

    Args:
        cost_per_day: {label: daily cost} table to validate.
        labels: the node's actual domain, e.g. LIVING_LABELS.
        node_name: node name, used only for the error message.
    """
    missing = set(labels) - cost_per_day.keys()
    extra = cost_per_day.keys() - set(labels)
    if missing or extra:
        raise ValueError(
            f"{node_name} cost table doesn't match its domain {labels}: "
            f"missing={missing or None}, unexpected={extra or None}"
        )


def expected_total_objective(
    bn2_path: str,
    T: int,
    living_cost_per_day: dict = LIVING_COST_PER_DAY,
    treatment_cost_per_day: dict = TREATMENT_COST_PER_DAY,
    health_penalty_per_day: dict = HEALTH_PENALTY_PER_DAY,
    omalaakari_cost_per_day: dict = OMALAAKARI_COST_PER_DAY
) -> tuple[float, list[float], list[float]]:
    """Baseline expected cost and health penalty over T timesteps under the
    model's own (status quo) transition dynamics, with no decisions involved
    -- sum_t E[living_cost_per_day[living_t] + treatment_cost_per_day[treatment_t]]
    and sum_t E[health_penalty_per_day[diagnosis_t]], computed directly on
    the plain unrolled BayesNet via LazyPropagation. This is the same
    objective build_treatment_decision_diagram's utilities score, so its
    total is directly comparable to solve_treatment_policy's MEU.

    Args:
        bn2_path: path to the trained 2TBN .bif file, e.g. "data/elder_twodbn.bif".
        T: number of timesteps to unroll and score.
        living_cost_per_day: see HEALTH_PENALTY_PER_DAY-style module constants.
        treatment_cost_per_day: as above.
        health_penalty_per_day: as above.

    Returns:
        (total_cost_plus_penalty, per_timestep_cost, per_timestep_health):
        total is cost + health penalty summed over all t; the other two are
        each term's breakdown by t.
    """
    _check_cost_table(living_cost_per_day, LIVING_LABELS, "living")
    _check_cost_table(treatment_cost_per_day, TREATMENT_LABELS, "treatment")
    _check_cost_table(health_penalty_per_day, DIAGNOSIS_LABELS, "diagnosis")
    _check_cost_table(omalaakari_cost_per_day, ["0", "1"], "omalaakari")

    bn2 = gum.loadBN(bn2_path)
    dbn2 = gdyn.unroll2TBN(bn2, T)

    ie = gum.LazyPropagation(dbn2)
    for t in range(T):
        # joint (not just marginal) posteriors have to be declared as
        # targets up front, or jointPosterior raises after makeInference.
        ie.addJointTarget(
            {
                dbn2.idFromName(f"living{t}"),
                dbn2.idFromName(f"treatment{t}"),
                dbn2.idFromName(f"omalaakari{t}")
            }
        )
    ie.makeInference()

    per_timestep_cost = []
    per_timestep_health = []
    for t in range(T):
        living_var = dbn2.variable(f"living{t}")
        treatment_var = dbn2.variable(f"treatment{t}")
        omalaakari_var = dbn2.variable(f"omalaakari{t}")
        jp = ie.jointPosterior(
            {
                living_var.name(),
                treatment_var.name(),
                omalaakari_var.name()
            }
        )

        expected_cost = 0.0
        inst = gum.Instantiation(jp)
        inst.setFirst()
        while not inst.end():
            lliving = living_var.label(inst.val(living_var))
            ltreat = treatment_var.label(inst.val(treatment_var))
            lomal = omalaakari_var.label(inst.val(omalaakari_var))
            expected_cost += jp.get(inst) * (
                living_cost_per_day[lliving] + 
                treatment_cost_per_day[ltreat] +
                omalaakari_cost_per_day[lomal]
            )
            inst.inc()
        per_timestep_cost.append(expected_cost)

        diagnosis_var = dbn2.variable(f"diagnosis{t}")
        diagnosis_post = ie.posterior(f"diagnosis{t}")
        expected_health = sum(
            diagnosis_post[i] * health_penalty_per_day[diagnosis_var.label(i)]
            for i in range(diagnosis_var.domainSize())
        )
        per_timestep_health.append(expected_health)

    total = sum(per_timestep_cost) + sum(per_timestep_health)
    return total, per_timestep_cost, per_timestep_health


# Kept as a thin wrapper: expected_total_objective now covers cost, but this
# name is already used elsewhere (e.g. the notebook), so it stays as the
# cost-only entry point.
def expected_total_cost(
    bn2_path: str,
    T: int,
    living_cost_per_day: dict = LIVING_COST_PER_DAY,
    treatment_cost_per_day: dict = TREATMENT_COST_PER_DAY,
) -> tuple[float, list[float]]:
    """Baseline expected total daily cost over T timesteps under the model's
    own transition dynamics (status quo, no decisions, no health penalty).
    See expected_total_objective for the cost+health-penalty version used to
    compare against the optimized policy.

    Args:
        bn2_path: path to the trained 2TBN .bif file, e.g. "data/elder_twodbn.bif".
        T: number of timesteps to unroll and cost.
        living_cost_per_day: see module constants.
        treatment_cost_per_day: see module constants.

    Returns:
        (total, per_timestep): total expected cost, and its breakdown by t.
    """
    total, per_timestep_cost, _ = expected_total_objective(
        bn2_path,
        T,
        living_cost_per_day,
        treatment_cost_per_day,
        {l: 0 for l in DIAGNOSIS_LABELS},
    )
    return total, per_timestep_cost


def bn_to_built_network(bn: gum.BayesNet) -> tuple[BuiltNetwork, dict[int, list[str]]]:
    """Adapt an already-trained/unrolled gum.BayesNet into the BuiltNetwork
    shape cross_impact.influence_diagram.build_influence_diagram expects --
    without invoking any of cross_impact's own fitting machinery, since
    `bn`'s CPTs are already known (learned via BNLearner + unroll2TBN), not
    something to fit from elicited cross-impact estimates.

    CrossImpactNetwork.marginals is left as zero-filled placeholders:
    build_influence_diagram never reads it (only .factors and built.cpts),
    and computing it for real would require exact joint inference over the
    whole network for no benefit here.

    Args:
        bn: the (typically already unrolled) chance network to adapt.

    Returns:
        (built, state_labels): built.network.factors are in bn's topological
        order; state_labels is {factor_index: [labels]}, suitable as
        build_influence_diagram's state_labels argument.
    """
    order = bn.topologicalOrder()
    names = [bn.variable(nid).name() for nid in order]
    index_of = {name: i for i, name in enumerate(names)}

    factors: list[UncertaintyFactor] = []
    cpts: list[np.ndarray] = []
    state_labels: dict[int, list[str]] = {}
    for i, nid in enumerate(order):
        var = bn.variable(nid)
        n_states = var.domainSize()
        state_labels[i] = list(var.labels())

        parent_names = sorted(
            (bn.variable(p).name() for p in bn.parents(nid)), key=lambda n: index_of[n]
        )
        parent_indices = tuple(index_of[n] for n in parent_names)
        factors.append(
            UncertaintyFactor(
                name=var.name(), n_states=n_states, parents=parent_indices
            )
        )

        parent_dims = [bn.variable(bn.idFromName(n)).domainSize() for n in parent_names]
        cpt = np.zeros((n_states, *parent_dims))
        pot = bn.cpt(nid)
        if not parent_names:
            cpt[:] = pot[:]
        else:
            # dict-based partial instantiation (parents only, own var left
            # free) sidesteps pyAgrum's internal (not insertion-order) axis
            # ordering -- same convention train_dbn.py's set_prior/tie_cpt use.
            for config in itertools.product(*(range(d) for d in parent_dims)):
                cpt[(slice(None), *config)] = pot[
                    {n: s for n, s in zip(parent_names, config)}
                ]
        cpts.append(cpt)

    network = CrossImpactNetwork(
        factors=factors, marginals=[np.zeros(f.n_states) for f in factors]
    )
    return BuiltNetwork(network=network, cpts=cpts, joint=None), state_labels


def build_treatment_decision_diagram(
    bn2_path: str,
    T: int,
    living_cost_per_day: dict = LIVING_COST_PER_DAY,
    treatment_cost_per_day: dict = TREATMENT_COST_PER_DAY,
    health_penalty_per_day: dict = HEALTH_PENALTY_PER_DAY,
) -> gum.InfluenceDiagram:
    """Load the trained 2TBN, unroll it over T timesteps, promote every
    treatment{t} to a decision node, and add per-timestep utility nodes for
    (negative) cost and health penalty -- utility = -(cost + health penalty),
    so maximizing utility minimizes cost+penalty.

    build_influence_diagram's `promote` drops a promoted node's own incoming
    arcs (by design, for its original "strategic lever" use case -- see
    cross_impact/influence_diagram.py's docstring). That's wrong here: we
    want each decision informed by that day's diagnosis, so
    diagnosis{t} -> treatment{t} is manually restored afterward. This also
    happens to be what makes the diagram solvable in plain LIMID mode (see
    solve_treatment_policy): it's the only structural link connecting
    treatment{t} to treatment{t+1} (via sickness{t+1} -> diagnosis{t+1}),
    without which pyAgrum can't establish any decision order at all.

    Promoted decision nodes lose their original labels (they become "0".."3"
    rather than NONE/HEALTH_CENTRE/ED/HOSPITAL) -- use TREATMENT_LABELS
    (train_dbn.py) to map indices back; see decision_label in
    solve_treatment_policy.

    Args:
        bn2_path: path to the trained 2TBN .bif file, e.g. "data/elder_twodbn.bif".
        T: number of timesteps to unroll.
        living_cost_per_day: see module constants.
        treatment_cost_per_day: see module constants.
        health_penalty_per_day: see module constants.

    Returns:
        The constructed gum.InfluenceDiagram, with treatment0..treatment{T-1}
        as decisions and cost0..cost{T-1}/health0..health{T-1} as utilities.
    """
    _check_cost_table(living_cost_per_day, LIVING_LABELS, "living")
    _check_cost_table(treatment_cost_per_day, TREATMENT_LABELS, "treatment")
    _check_cost_table(health_penalty_per_day, DIAGNOSIS_LABELS, "diagnosis")

    bn2 = gum.loadBN(bn2_path)
    dbn2 = gdyn.unroll2TBN(bn2, T)
    built, state_labels = bn_to_built_network(dbn2)

    cost_table = -np.array(
        [
            [
                living_cost_per_day[l] + treatment_cost_per_day[tr]
                for tr in TREATMENT_LABELS
            ]
            for l in LIVING_LABELS
        ]
    )
    health_table = -np.array([health_penalty_per_day[d] for d in DIAGNOSIS_LABELS])

    decision_names = tuple(f"treatment{t}" for t in range(T))
    utilities = tuple(
        Utility(f"cost{t}", (f"living{t}", f"treatment{t}"), cost_table)
        for t in range(T)
    ) + tuple(Utility(f"health{t}", (f"diagnosis{t}",), health_table) for t in range(T))

    spec = InfluenceDiagramSpec(
        built=built, promote=decision_names, utilities=utilities
    )
    id_ = build_influence_diagram(spec, state_labels=state_labels)
    for t in range(T):
        id_.addArc(f"diagnosis{t}", f"treatment{t}")
    return id_


def decision_label(index: int) -> str:
    """Map a promoted treatment decision's chosen index (0..3, since
    build_influence_diagram's promote drops the original NONE/HEALTH_CENTRE/
    ED/HOSPITAL labels -- see build_treatment_decision_diagram) back to its
    TREATMENT_LABELS name."""
    return TREATMENT_LABELS[index]


def solve_treatment_policy(
    id_: gum.InfluenceDiagram, T: int, sample_timesteps: list[int] | None = None
):
    """Solve build_treatment_decision_diagram's diagram for the optimal
    treatment policy via gum.ShaferShenoyLIMIDInference in plain LIMID mode
    (no addNoForgettingAssumption).

    LIMID mode uses only each decision's own declared parents (here:
    diagnosis{t} alone) as its information set, which is exactly right for
    this model -- treatment{t}'s true information need is the current
    diagnosis, not the full history, since diagnosis{t} is already a Markov
    sufficient statistic. addNoForgettingAssumption instead forces literal
    full-history tracking, which was measured to blow up (junction-tree
    clique size growing ~linearly with T, T=10 already taking ~12s) --
    whereas plain LIMID mode stays at clique size 5 regardless of T (checked
    up to T=1000).

    Args:
        id_: as returned by build_treatment_decision_diagram.
        T: number of timesteps (must match what id_ was built with).
        sample_timesteps: which decisions' policies to report (finite-horizon
            optimal policies are generally non-stationary -- more/less
            willingness to treat depending on how much horizon remains -- so
            printing every one of T decisions isn't useful). Defaults to
            first, quarter, half, three-quarter, and last.

    Returns:
        (expected_utility, policy): expected_utility is ie.MEU()'s mean
        (utility = -(cost + health penalty), so -expected_utility is the
        expected total cost+penalty, comparable to expected_total_objective);
        policy is {t: {diagnosis_label: chosen_treatment_label}} for the
        sampled timesteps.
    """
    if sample_timesteps is None:
        sample_timesteps = sorted({0, T // 4, T // 2, (3 * T) // 4, T - 1})

    ie = gum.ShaferShenoyLIMIDInference(id_)
    ie.makeInference()

    policy = {}
    for t in sample_timesteps:
        diagnosis_var = id_.variable(f"diagnosis{t}")
        treatment_var = id_.variable(f"treatment{t}")
        # optimalDecision returns a near-deterministic distribution over the
        # decision's alternatives per diagnosis value (1.0 for the chosen
        # one); read it back via Instantiation over both variables.
        opt = ie.optimalDecision(f"treatment{t}")
        by_diagnosis = {}
        inst = gum.Instantiation(opt)
        inst.setFirst()
        while not inst.end():
            if opt.get(inst) > 0.5:
                dlabel = diagnosis_var.label(inst.val(diagnosis_var))
                by_diagnosis[dlabel] = decision_label(inst.val(treatment_var))
            inst.inc()
        policy[t] = by_diagnosis

    return ie.MEU()["mean"], policy


if __name__ == "__main__":
    T = 100
    total, per_timestep_cost, per_timestep_health = expected_total_objective(
        "data/elder_twodbn.bif", T
    )
    print(f"status quo: expected total cost+penalty over {T} timesteps: {total:.2f}")

    id_ = build_treatment_decision_diagram("data/elder_twodbn.bif", T)
    expected_utility, policy = solve_treatment_policy(id_, T)
    print(f"optimized policy: expected total cost+penalty: {-expected_utility:.2f}")
    for t, by_diagnosis in policy.items():
        print(f"  t={t}: {by_diagnosis}")
