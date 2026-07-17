"""The worked examples from Roponen & Salo (2024), ported from the original
MATLAB scripts (energy_network.m, printing_network.m) as reusable
CrossImpactNetwork builders -- both for tests and as usage examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cross_impact.io import load_cross_impact_matrix
from cross_impact.model import CrossImpactNetwork, UncertaintyFactor

DATA_DIR = Path(__file__).parent / "data"


def energy_network() -> CrossImpactNetwork:
    """The Soisalo energy-production example (Section 3, Tables 1-4):
    3 factors, nuclear power -> renewables -> energy storage."""
    factors = [
        UncertaintyFactor("Nuclear power", 3, ()),
        UncertaintyFactor("Renewables", 4, (0,)),
        UncertaintyFactor("Energy storage", 3, (0, 1)),
    ]
    marginals = [
        np.array([0.5, 0.3, 0.2]),
        np.array([0.3, 0.2, 0.3, 0.2]),
        np.array([0.5, 0.4, 0.1]),
    ]
    cross_impacts = load_cross_impact_matrix(
        DATA_DIR / "energy_example.csv",
        states_by_ci_index=[3, 4, 3],
        ci_index_of_factor=[0, 1, 2],
        scale="multiplier",
    )
    return CrossImpactNetwork(factors=factors, marginals=marginals, cross_impacts=cross_impacts)


PRINTING_FACTOR_NAMES = [
    "Global industry growth",
    "Printing speed compared to present",
    "Printing cost compared to present",
    "Finnish industry growth",
    "Graduates with 3D-printing expertise",
    "Legal regulation of 3D-printing in Finland",
    "Standardization of processes and models",
    "Use of 3D-printed objects in FDF",
    "FDF access to 3D-printing model files",
    "FDF 3D-printing spare parts in peacetime",
    "FDF 3D-printing spare parts in crisis times",
]

# 1-indexed, matching printing_network.m's `nodes(i).parents` exactly.
_PRINTING_PARENTS_1IDX = [
    [],
    [1],
    [1, 2],
    [1, 2, 3],
    [1, 4],
    [1, 4],
    [1, 4],
    [3, 4, 6],
    [3, 4, 5, 6, 7, 8],
    [3, 4, 6, 8, 9],
    [2, 4, 7, 8, 9, 10],
]

# nodes(i).CIindex in printing_network.m: the block position of factor i in
# cross_impacts_3d.csv, which differs from processing order for factors 2-4.
_PRINTING_CI_INDEX_1IDX = [1, 4, 3, 2, 5, 6, 7, 8, 9, 10, 11]

PRINTING_MARGINALS = [
    np.array([0.3, 0.5, 0.2]),
    np.array([0.4, 0.5, 0.1]),
    np.array([0.5, 0.4, 0.1]),
    np.array([0.3, 0.5, 0.2]),
    np.array([0.2, 0.6, 0.2]),
    np.array([0.05, 0.9, 0.05]),
    np.array([0.35, 0.45, 0.2]),
    np.array([0.1, 0.5, 0.4]),
    np.array([0.2, 0.7, 0.1]),
    np.array([0.7, 0.29, 0.01]),
    np.array([0.45, 0.45, 0.1]),
]


def printing_network() -> CrossImpactNetwork:
    """The 3D-printing / Finnish Defence Forces case study (Section 4,
    Tables 6-8): 11 factors, each with 3 outcomes. The largest node (9,
    "FDF access to 3D-printing model files") has 6 parents, so fitting its
    conditional distribution needs ~2200 QP variables -- by far the most
    expensive node to fit in this network."""
    factors = [
        UncertaintyFactor(name, 3, tuple(p - 1 for p in parents))
        for name, parents in zip(PRINTING_FACTOR_NAMES, _PRINTING_PARENTS_1IDX)
    ]
    cross_impacts = load_cross_impact_matrix(
        DATA_DIR / "cross_impacts_3d.csv",
        states_by_ci_index=[3] * 11,
        ci_index_of_factor=[c - 1 for c in _PRINTING_CI_INDEX_1IDX],
        scale="ordinal",
    )
    return CrossImpactNetwork(factors=factors, marginals=list(PRINTING_MARGINALS), cross_impacts=cross_impacts)
