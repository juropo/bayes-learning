"""Learn the elder-care 2TBN's parameters from the visit log. `sickness` is
set equal to `diagnosis` (see build_training_csv) rather than left hidden:
with sickness almost entirely unanchored, EM only ever assigns its 5 labels
arbitrarily and independently between the sickness0 and sicknesst nodes,
which silently corrupts tie_stationary_relations (it renames sickness0's
label space onto sicknesst's, assuming they mean the same thing -- they
don't, unless sickness is fully observed). `anchor` remains the only
genuinely hidden variable and is fit via EM.
"""

import itertools

import pyagrum as gum

from dbn_data_prep import build_transition_table

DIAGNOSIS_LABELS = ["NONE", "MILD", "MEDIUM", "DIFFICULT", "DEAD"]
# No "DEAD" here: death is fully captured by `diagnosis`, and living never
# gains a redundant state of its own for it (see dbn_data_prep._endpoint_events).
LIVING_LABELS = ["INDEPENDENT", "FAMILY_CARE", "HOME_CARE", "HEALTH_CENTRE"]
TREATMENT_LABELS = ["NONE", "HEALTH_CENTRE", "ED", "HOSPITAL"]
SICKNESS_LABELS = DIAGNOSIS_LABELS  # sickness is set equal to diagnosis (same vocabulary)
ANCHOR_LABELS = ["0", "1"]

# sickness0's fixed prior, replacing what the learner would otherwise estimate
# from a mix of every tick, not just true starting states. Elders' first
# recorded diagnosis, corrected for selection bias: the simulator generates
# 5000 people but only 4547 ever appear in the visit log, so 5000-4547=453
# never got sick enough to need care and are added as NONE. DEAD is excluded
# and the rest renormalized (nobody starts out already dead).
#   NONE 2827+453=3280, MILD 1328, MEDIUM 26, DIFFICULT 228, DEAD 138 (excluded)
#   total (excl. DEAD) = 4862
SICKNESS0_PRIOR = {
    "NONE": 3280 / 4862,
    "MILD": 1328 / 4862,
    "MEDIUM": 26 / 4862,
    "DIFFICULT": 228 / 4862,
    "DEAD": 0.0,
}

TOPOLOGY = (
    "diagnosis0[5]<-sickness0[5]->sicknesst[5]<-treatment0[3]<-diagnosis0[5]<-anchor[2];"
    "sicknesst[5]<-living0[4]->livingt[4]<-diagnosis0[5];"
    "diagnosist[5]<-anchor[2];"
    "treatmentt[3]<-diagnosist[5]<-sicknesst[5]"
)
# diagnosist has no arc from diagnosis0, and treatmentt has none from treatment0:
# the sickness->diagnosis and diagnosis->treatment relationships are meant to be
# identical at every timestep (see tie_cpt below), which only holds if diagnosist
# and treatmentt have the same parent shape as diagnosis0 and treatment0.

LABELS = {
    "diagnosis0": DIAGNOSIS_LABELS,
    "diagnosist": DIAGNOSIS_LABELS,
    "living0": LIVING_LABELS,
    "livingt": LIVING_LABELS,
    "treatment0": TREATMENT_LABELS,
    "treatmentt": TREATMENT_LABELS,
    "sickness0": SICKNESS_LABELS,
    "sicknesst": SICKNESS_LABELS,
    "anchor": ANCHOR_LABELS,
}


def build_skeleton() -> gum.BayesNet:
    """Same topology as gum.fastBN(TOPOLOGY), but with real category labels
    instead of fastBN's default "0".."n-1" labels, so BNLearner can match the
    training CSV's string values to the right modalities.

    Takes no parameters.
    """
    shape = gum.fastBN(TOPOLOGY)
    bn = gum.BayesNet()
    for nid in shape.nodes():
        name = shape.variable(nid).name()
        bn.add(gum.LabelizedVariable(name, name, LABELS[name]))
    for tail, head in shape.arcs():
        bn.addArc(shape.variable(tail).name(), shape.variable(head).name())
    bn.generateCPTs()
    return bn


def force_deterministic(bn: gum.BayesNet, node_name: str, fixed: dict, target_label: str) -> None:
    """Overwrite bn.cpt(node_name) so that whenever `fixed` holds (a subset of
    the node's parents), the node takes `target_label` with probability 1,
    regardless of its other parents' values.

    Args:
        bn: network to modify in place.
        node_name: name of the node whose CPT is overwritten.
        fixed: {parent_name: label} values that trigger the override; may be
            a subset of node_name's parents, with the rest left free.
        target_label: the label to pin to probability 1 whenever `fixed` holds.
    """
    var_id = bn.idFromName(node_name)
    cpt = bn.cpt(var_id)
    labels = bn.variable(var_id).labels()
    dist = [1.0 if lbl == target_label else 0.0 for lbl in labels]

    other_parents = [p for p in bn.parents(var_id) if bn.variable(p).name() not in fixed]
    other_names = [bn.variable(p).name() for p in other_parents]
    other_labels = [bn.variable(p).labels() for p in other_parents]

    for combo in itertools.product(*other_labels) if other_names else [()]:
        inst = dict(fixed)
        inst.update(zip(other_names, combo))
        cpt[inst] = dist


def tie_cpt(bn: gum.BayesNet, target_name: str, source_name: str, rename: dict) -> None:
    """Overwrite bn.cpt(target_name) with bn.cpt(source_name)'s values,
    renaming source's parent names to target's via `rename`. Requires both
    nodes to have the same parent shape (same domains, up to the rename).

    Args:
        bn: network to modify in place.
        target_name: name of the node whose CPT is overwritten.
        source_name: name of the node whose CPT values are copied.
        rename: {source_parent_name: target_parent_name} mapping used to
            translate a source parent instantiation into a target one; a
            source parent absent from `rename` is assumed to have the same
            name on the target.
    """
    src_id = bn.idFromName(source_name)
    src_cpt = bn.cpt(src_id)
    tgt_cpt = bn.cpt(bn.idFromName(target_name))

    src_parents = [bn.variable(p).name() for p in bn.parents(src_id)]
    parent_labels = [bn.variable(bn.idFromName(p)).labels() for p in src_parents]

    for combo in itertools.product(*parent_labels) if src_parents else [()]:
        src_inst = dict(zip(src_parents, combo))
        tgt_inst = {rename.get(k, k): v for k, v in src_inst.items()}
        tgt_cpt[tgt_inst] = src_cpt[src_inst]


def set_prior(bn: gum.BayesNet, node_name: str, distribution: dict) -> None:
    """Overwrite a parent-less node's marginal CPT with a fixed distribution.

    Args:
        bn: network to modify in place.
        node_name: name of the parent-less node whose marginal is overwritten.
        distribution: {label: probability} covering every one of the node's
            labels; must sum to 1 (not renormalized here).
    """
    var_id = bn.idFromName(node_name)
    labels = bn.variable(var_id).labels()
    bn.cpt(var_id)[:] = [distribution[lbl] for lbl in labels]


def tie_stationary_relations(bn: gum.BayesNet) -> gum.BayesNet:
    """sickness->diagnosis and diagnosis->treatment are assumed to be the same
    relationship at t0 and tn, so share the t0 CPT instead of trusting EM to
    have learned two independent (and possibly drifting) copies of it. Safe
    now that sickness0/sicknesst are fully observed (equal to diagnosis0/
    diagnosist) and therefore share the same label semantics.

    Args:
        bn: network to modify in place.

    Returns:
        The same `bn`, for chaining.
    """
    tie_cpt(bn, "diagnosist", "diagnosis0", {"sickness0": "sicknesst"})
    tie_cpt(bn, "treatmentt", "treatment0", {"diagnosis0": "diagnosist"})
    return bn


def apply_domain_knowledge(bn: gum.BayesNet) -> gum.BayesNet:
    """Bake in facts we're certain of rather than trusting the learned CPTs on
    parent combinations that have little or no real data behind them:
    - "DEAD" sickness is absorbing;
    - once diagnosed dead, treatment stops;
    - sickness0's own prior is fixed from data instead of left to the learner
      (see SICKNESS0_PRIOR): nobody starts out already dead, and most people
      aren't very sick to begin with.
    (living has no "DEAD" state of its own to force -- see LIVING_LABELS.)

    Args:
        bn: freshly learned network to correct in place.

    Returns:
        The same `bn`, for chaining.
    """
    tie_stationary_relations(bn)

    force_deterministic(bn, "sicknesst", {"sickness0": "DEAD"}, "DEAD")
    force_deterministic(bn, "treatmentt", {"diagnosist": "DEAD"}, "NONE")
    set_prior(bn, "sickness0", SICKNESS0_PRIOR)
    return bn


def build_training_csv(visits_csv: str, out_csv: str) -> None:
    """Reshape the raw visit log into (t, t+1) transition pairs (via
    dbn_data_prep.build_transition_table) and add the columns BNLearner needs
    but the raw log doesn't have: sickness0/sicknesst (set equal to
    diagnosis0/diagnosist) and anchor (fully hidden, marked "?").

    Args:
        visits_csv: path to the raw visit log CSV.
        out_csv: path to write the reshaped training CSV to.
    """
    table = build_transition_table(visits_csv).drop(columns=["Elder ID"])
    # sickness = diagnosis exactly -- see module docstring for why.
    table["sickness0"] = table["diagnosis0"]
    table["sicknesst"] = table["diagnosist"]
    table["anchor"] = "?"
    table.to_csv(out_csv, index=False)


def train(
    visits_csv: str,
    training_csv: str,
    output_bn: str,
    em_epsilon: float = 1e-2,
    smoothing_weight: float = 1e-2,
) -> gum.BayesNet:
    """Build the training CSV, learn CPTs via EM, apply the domain-knowledge
    corrections, and save the result to a .bif file.

    Args:
        visits_csv: path to the raw visit log CSV.
        training_csv: path to write the intermediate reshaped training CSV to
            (also the file BNLearner reads from).
        output_bn: path to save the trained network to (.bif).
        em_epsilon: log-likelihood convergence threshold for EM (only
            `anchor` is actually latent, so this mostly affects how tightly
            its prior is fit).
        smoothing_weight: Laplace-style smoothing prior weight, needed to
            avoid a hard error on parent combinations the data never observes.

    Returns:
        The trained gum.BayesNet (also written to `output_bn`).
    """
    build_training_csv(visits_csv, training_csv)
    skeleton = build_skeleton()

    learner = gum.BNLearner(training_csv, skeleton)
    # Some (parent) configurations are never jointly observed in the data;
    # EM needs a prior to avoid a hard DatabaseError on those zero counts.
    learner.useSmoothingPrior(smoothing_weight)
    learner.useEM(em_epsilon)
    learned = learner.learnParameters(skeleton)
    apply_domain_knowledge(learned)

    gum.saveBN(learned, output_bn)
    return learned


if __name__ == "__main__":
    bn = train(
        visits_csv="data/visits_hc_450_ed_50_hosp_25.csv",
        training_csv="data/dbn_training_data.csv",
        output_bn="data/elder_twodbn.bif",
    )
    print(bn)
