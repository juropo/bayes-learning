# bayes-learning

This repo holds two unrelated Bayesian-network projects, each in its own folder:

- [elder_care_dbn/](elder_care_dbn/), a 2TBN learned from simulated visit logs with pyAgrum — see the section below.
- [cross_impact/](cross_impact/), a Python port of Roponen & Salo (2024), ["A probabilistic cross-impact methodology for explorative scenario analysis"](https://doi.org/10.1002/ffo2.165) — see [that section](#cross_impact-cross-impact-scenario-analysis) below.

Both share this repo's single `pyproject.toml`/`uv.lock` (`uv sync` once at the root installs everything for both); each project's own commands should be run with that project's folder as the working directory (see each section's Usage).

## elder_care_dbn: elder care 2TBN

A dynamic Bayesian network (2-time-slice / 2TBN) modeling elder care trajectories — health diagnosis, living situation, and treatment over time — learned from simulated healthcare visit logs using [pyAgrum](https://agrum.gitlab.io/).

### What this is

The data is a simulated log of healthcare visits for a population of elders: when they're treated, where they're living, what they're diagnosed with, and how that evolves over time. This project turns that visit log into a trained 2TBN that can be unrolled and queried to answer questions like "given someone's current diagnosis and living situation, how is their trajectory likely to evolve?"

The model has five variables per time slice:

- `diagnosis` — health diagnosis (`NONE`, `MILD`, `MEDIUM`, `DIFFICULT`, `DEAD`)
- `living` — living situation (`INDEPENDENT`, `FAMILY_CARE`, `HOME_CARE`, `HEALTH_CENTRE`, `DEAD`)
- `treatment` — treatment received that tick (`NONE`, `HEALTH_CENTRE`, `ED`)
- `sickness` — set equal to `diagnosis` (see [elder_care_dbn/train_dbn.py](elder_care_dbn/train_dbn.py) module docstring for why it isn't left hidden)
- `anchor` — a static, binary, genuinely latent per-individual trait, constant across all time slices

`DEAD` is absorbing for diagnosis, living, and treatment: once reached, none of them change again.

### Project structure

| File | Purpose |
|---|---|
| [elder_care_dbn/dbn_data_prep.py](elder_care_dbn/dbn_data_prep.py) | Reshapes the raw visit log into `(t, t+1)` transition pairs, filling in the gaps between logged visits (no visit = no treatment, diagnosis/living carry over) |
| [elder_care_dbn/train_dbn.py](elder_care_dbn/train_dbn.py) | Builds the network structure and learns its parameters from the transition pairs |
| [elder_care_dbn/test.ipynb](elder_care_dbn/test.ipynb) | Exploration notebook: building/visualizing 2TBNs, unrolling, running inference |
| `elder_care_dbn/data/visits_added_waiting_times_*.csv` | Raw simulated visit logs (different `hc_*_ed_*` files are separate simulation runs with different healthcare-capacity parameters — not combinable) |
| `elder_care_dbn/data/elder_twodbn.bif` | The trained model |

### Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

### Usage

All commands below are run from inside `elder_care_dbn/` (the scripts use paths relative to that folder, e.g. `data/...`, not the repo root):

```bash
cd elder_care_dbn
```

Train the model (rebuilds `data/dbn_training_data.csv` and `data/elder_twodbn.bif`):

```bash
uv run python train_dbn.py
```

Load and query the trained model:

```python
import pyagrum as gum
import pyagrum.lib.dynamicBN as gdyn

bn = gum.loadBN("data/elder_twodbn.bif")

# unroll into a T-timestep simulation
T = 20
dbn = gdyn.unroll2TBN(bn, T)

# run inference given a starting condition
ie = gum.LazyPropagation(dbn)
ie.setEvidence({"diagnosis0": "MILD", "living0": "FAMILY_CARE", "treatment0": "NONE"})
ie.makeInference()
print(ie.posterior("diagnosis10"))
```

### Notes

- `sickness0`/`sicknesst` are set equal to `diagnosis0`/`diagnosist` rather than left as free hidden variables. With almost no data anchoring what "sickness" means, EM assigns its labels arbitrarily and independently between the two nodes — which silently corrupts any attempt to tie their conditional distributions together across time slices. Making sickness fully observed sidesteps that; `anchor` is the only variable actually fit via EM.
- `sickness0`'s own prior is fixed from data (see `SICKNESS0_PRIOR` in [elder_care_dbn/train_dbn.py](elder_care_dbn/train_dbn.py)) rather than learned, since the raw data mixes every timestep together rather than isolating true starting states.

## cross_impact: cross-impact scenario analysis

A Python port of the MATLAB code accompanying Roponen & Salo (2024), ["A probabilistic cross-impact methodology for explorative scenario analysis"](https://doi.org/10.1002/ffo2.165) (original MATLAB: a separate `cross-impact` repo). The method turns expert judgments — marginal probabilities per uncertainty factor plus pairwise "cross-impact multipliers" `C^ij_kl = P(X^i=k | X^j=l) / P(X^i=k)`, which need not be mutually consistent — into a Bayesian network: a constrained fit (solved with cvxpy, using its bundled open-source solvers) produces each node's conditional probability table in topological order, minimizing deviation from the cross-impact estimates subject to matching every marginal exactly.

- [cross_impact/model.py](cross_impact/model.py) — `UncertaintyFactor`/`CrossImpactNetwork` data structures, joint-distribution bookkeeping (`build_network`, `extend_joint`, `parent_joint_distribution`), and `scenario_probability` (chain-rule lookup from the fitted CPTs alone).
- [cross_impact/fitting.py](cross_impact/fitting.py) — `fit_conditional_distribution` (the per-node fit `build_network` calls once per factor) and `fit_joint_distribution` (the direct whole-network fit, used only to sanity-check the iterative approach on small networks since its variable count is the full product of all factor state counts). Both take a `loss` argument — `"least_squares"` (default, matches the paper) or `"kl"` (generalized KL divergence from the cross-impact-implied targets, penalizing relative rather than absolute deviation) — and an `engine` argument selecting how it's solved.
- [cross_impact/ipf.py](cross_impact/ipf.py) — `engine="ipf"`: Iterative Proportional Fitting, a dependency-free multiplicative-update algorithm and the classical way to solve a KL/table-fitting problem like this one, faster than routing through cvxpy's exponential-cone solvers and with no solver-robustness edge cases at scale.
- [cross_impact/io.py](cross_impact/io.py) — loading cross-impact matrices from the block-triangular CSV format the original MATLAB spreadsheets use.
- [cross_impact/examples.py](cross_impact/examples.py) — the paper's two worked examples (`energy_network`, `printing_network`) as reusable network builders.
- [cross_impact/pyagrum_export.py](cross_impact/pyagrum_export.py) — `to_pyagrum`, exporting a fitted network into a `gum.BayesNet` for inference/visualization (the pyAgrum equivalent of what `genie_parser.m` did for GeNIe in the original MATLAB code); `graft_onto_pyagrum`, adding newly-fitted cross-impact nodes onto an *existing* `gum.BayesNet` (e.g. one learned from data) without touching its existing nodes.
- [cross_impact/pyagrum_import.py](cross_impact/pyagrum_import.py) — `marginals_and_joint_from_pyagrum`, extracting exact marginals/joints out of an already-fitted `gum.BayesNet` via inference, for use as `build_network`'s `initial_joint`.

```python
from cross_impact import build_network, energy_network, to_pyagrum

built = build_network(energy_network(), mode="exact", loss="least_squares")
built.joint          # full joint distribution over all scenarios
built.cpts[i]         # factor i's conditional probability table, shape (n_states_i, *parent_state_dims)

bn = to_pyagrum(built)  # gum.BayesNet, ready for gum.LazyPropagation etc.
```

`build_network`'s `mode` trades off exactness against scalability: `"exact"` tracks the true running joint over already-processed factors (fine up to a few dozen factors); `"independent"` instead approximates each node's parent distribution as the outer product of its parents' own marginals, which is far cheaper and is what makes fitting a 121-factor network take a fraction of a second (see `tests/test_tree_network.py`) but loses accuracy whenever parents are correlated.

`loss`/`engine` combinations: `loss="least_squares"` (the default, matching the paper) only runs on `engine="cvxpy"`. `loss="kl"` runs on either `engine="cvxpy"` (a proper convex solve, but needs an exponential-cone-capable solver) or `engine="ipf"` (raking/Sinkhorn-style, no cvxpy dependency at solve time). Both `loss="kl"` engines minimize the same objective in spirit but aren't guaranteed to reach the identical optimum when the cross-impact estimates are mutually inconsistent (their fitting problem is then underdetermined/non-convex-in-practice for cyclic projection) — `tests/test_ipf.py` checks they stay in the same neighborhood rather than requiring bit-for-bit agreement.

#### Combining with a network learned from data (e.g. via `elder_care_dbn/train_dbn.py`-style `BNLearner`)

`pyAgrum`'s `BNLearner` has no way to hold a subset of a network's parameters fixed while learning the rest (confirmed empirically, not just from the docs: pre-setting a node's CPT on the skeleton and marking it as never-observed in the training data does *not* survive `learnParameters` -- it's only used as an EM starting point, then overwritten). So rather than fight that, cross-impact nodes can be added *after* learning, downstream of the learned nodes, without needing `BNLearner` to know about them at all:

```python
from cross_impact import CrossImpactNetwork, UncertaintyFactor, build_network
from cross_impact.pyagrum_export import graft_onto_pyagrum
from cross_impact.pyagrum_import import marginals_and_joint_from_pyagrum

# `learned_bn` came from BNLearner (e.g. elder_care_dbn/train_dbn.py's train()).
marginals, joint = marginals_and_joint_from_pyagrum(learned_bn, ["diagnosis0", "living0"])

network = CrossImpactNetwork(
    factors=[
        UncertaintyFactor("diagnosis0", 5, ()),
        UncertaintyFactor("living0", 5, (0,)),
        UncertaintyFactor("new_factor", 3, (0, 1)),  # cross-impact node, depends on both learned nodes
    ],
    marginals=[*marginals, my_new_factor_marginal],
    cross_impacts={(0, 2): ..., (1, 2): ...},  # only for pairs involving new_factor
)
built = build_network(network, mode="exact", initial_joint=joint)
merged = graft_onto_pyagrum(learned_bn, built, n_given=2)
```

`marginals_and_joint_from_pyagrum` pulls the *exact* joint of however many learned nodes will act as parents (not just their marginals treated independently -- correlation between them is preserved via `gum.LazyPropagation.jointPosterior`, which needs `addJointTarget` called before inference for node sets that aren't already a clique, handled internally). `build_network`'s `initial_joint` then seeds the running joint with that exact prefix instead of fitting it, so the learned nodes are never touched by the (approximate) cross-impact fitting machinery at all -- only the genuinely new nodes get fit, correctly conditioned on the learned nodes' true correlation structure. `graft_onto_pyagrum` adds those new nodes onto a *copy* of the original `learned_bn` (which is left untouched) rather than rebuilding it from scratch.

### Tests

`tests/test_*.py` port the original MATLAB example scripts as validation:

| Test file | Original script | What it checks |
|---|---|---|
| `test_energy_network.py` | `energy_network.m` | Fitted CPTs match the paper's own worked numbers (Table 4) almost exactly |
| `test_printing_network.py` | `printing_network.m` | The 11-factor 3D-printing/FDF case study; reproduces the paper's "8.62%" most-likely-scenario result |
| `test_tree_network.py` | `tree_network.m` / Appendix B | 121-factor network builds correctly and quickly under `mode="independent"` |
| `test_randomized_consistency.py` | `randomized_data.m` | Fitted result is only mildly sensitive to processing order, and the iterative and direct-fit (`fit_joint_distribution`) approaches agree |
| `test_kl_loss.py` | — (new) | `loss="kl"` (via cvxpy) produces valid, marginal-matching distributions and is a genuinely different fit from `loss="least_squares"` |
| `test_ipf.py` | — (new) | `engine="ipf"` produces valid, marginal-matching distributions and broadly agrees with `engine="cvxpy"` under `loss="kl"` |
| `test_pyagrum_export.py` | — (new) | `to_pyagrum`'s structure/CPTs match the fitted network, and `gum.LazyPropagation` inference over the exported net reproduces the fitted joint's marginals |
| `test_pyagrum_import.py` | — (new) | `marginals_and_joint_from_pyagrum`'s extracted marginals/joints (adjacent, non-adjacent, 3-way, with evidence) match hand-computed values on a known chain network |
| `test_initial_joint.py` | — (new) | `build_network(initial_joint=...)`'s seeded prefix is derived exactly (not approximated), and a new factor's fit doesn't distort it |
| `test_pyagrum_roundtrip.py` | — (new) | Full learn-then-graft workflow: pyAgrum's own inference on the *merged* network reproduces `build_network`'s computed joint exactly, and the learned nodes' distribution is provably unchanged after grafting |

Run with `uv run pytest tests`.
