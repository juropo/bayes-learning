# bayes-learning

A dynamic Bayesian network (2-time-slice / 2TBN) modeling elder care trajectories — health diagnosis, living situation, and treatment over time — learned from simulated healthcare visit logs using [pyAgrum](https://agrum.gitlab.io/).

## What this is

The data is a simulated log of healthcare visits for a population of elders: when they're treated, where they're living, what they're diagnosed with, and how that evolves over time. This project turns that visit log into a trained 2TBN that can be unrolled and queried to answer questions like "given someone's current diagnosis and living situation, how is their trajectory likely to evolve?"

The model has five variables per time slice:

- `diagnosis` — health diagnosis (`NONE`, `MILD`, `MEDIUM`, `DIFFICULT`, `DEAD`)
- `living` — living situation (`INDEPENDENT`, `FAMILY_CARE`, `HOME_CARE`, `HEALTH_CENTRE`, `DEAD`)
- `treatment` — treatment received that tick (`NONE`, `HEALTH_CENTRE`, `ED`)
- `sickness` — set equal to `diagnosis` (see [train_dbn.py](train_dbn.py) module docstring for why it isn't left hidden)
- `anchor` — a static, binary, genuinely latent per-individual trait, constant across all time slices

`DEAD` is absorbing for diagnosis, living, and treatment: once reached, none of them change again.

## Project structure

| File | Purpose |
|---|---|
| [dbn_data_prep.py](dbn_data_prep.py) | Reshapes the raw visit log into `(t, t+1)` transition pairs, filling in the gaps between logged visits (no visit = no treatment, diagnosis/living carry over) |
| [train_dbn.py](train_dbn.py) | Builds the network structure and learns its parameters from the transition pairs |
| [test.ipynb](test.ipynb) | Exploration notebook: building/visualizing 2TBNs, unrolling, running inference |
| `data/visits_added_waiting_times_*.csv` | Raw simulated visit logs (different `hc_*_ed_*` files are separate simulation runs with different healthcare-capacity parameters — not combinable) |
| `data/elder_twodbn.bif` | The trained model |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

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

## Notes

- `sickness0`/`sicknesst` are set equal to `diagnosis0`/`diagnosist` rather than left as free hidden variables. With almost no data anchoring what "sickness" means, EM assigns its labels arbitrarily and independently between the two nodes — which silently corrupts any attempt to tie their conditional distributions together across time slices. Making sickness fully observed sidesteps that; `anchor` is the only variable actually fit via EM.
- `sickness0`'s own prior is fixed from data (see `SICKNESS0_PRIOR` in [train_dbn.py](train_dbn.py)) rather than learned, since the raw data mixes every timestep together rather than isolating true starting states.
