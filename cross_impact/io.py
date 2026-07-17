"""Loading cross-impact matrices from the block-triangular CSV format used by
the original MATLAB elicitation spreadsheets (energy_example.csv,
cross_impacts_3d.csv): a single (sum(states) x sum(states)) matrix with one
contiguous row/column block per uncertainty factor, only the strict upper
triangle of blocks populated, and the lower triangle filled in by symmetry
(eq. 5 in the paper: the cross-impact multiplier is the same in either
direction).

Two value scales are used across the source examples:
- "multiplier": cells already hold the cross-impact multiplier C^ij_kl.
- "ordinal": cells hold the -3..3 elicitation scale V^ij_kl, converted via
  C^ij_kl = sqrt(2)**V^ij_kl (eq. 32).

A network's factors are usually processed in a topological order that
differs from the order its cross-impacts were tabulated in (a factor's
"CIindex" in the MATLAB code); ci_index_of_factor bridges the two.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def multiplier_from_ordinal_scale(v: np.ndarray | float) -> np.ndarray | float:
    """C = sqrt(2)**V (eq. 32): converts the -3..3 elicitation scale into a
    cross-impact multiplier."""
    return np.sqrt(2.0) ** np.asarray(v, dtype=float)


def _block_bounds(states: list[int]) -> list[tuple[int, int]]:
    starts = np.cumsum([0, *states])
    return list(zip(starts[:-1], starts[1:]))


def cross_impacts_from_block_matrix(
    raw: np.ndarray, states_by_ci_index: list[int], scale: str = "multiplier"
) -> dict[tuple[int, int], np.ndarray]:
    """Split a raw block matrix into per-pair multiplier blocks, keyed by
    (ci_index_i, ci_index_j) with ci_index_i < ci_index_j -- i.e. still in
    the CSV's own block order, not yet remapped to a network's factor order."""
    raw = np.asarray(raw, dtype=float)
    symmetric = raw + raw.T
    bounds = _block_bounds(states_by_ci_index)

    if scale == "ordinal":
        symmetric = multiplier_from_ordinal_scale(symmetric)
    elif scale != "multiplier":
        raise ValueError(f"unknown scale {scale!r}")

    cross_impacts = {}
    for i in range(len(states_by_ci_index)):
        for j in range(i + 1, len(states_by_ci_index)):
            r0, r1 = bounds[i]
            c0, c1 = bounds[j]
            cross_impacts[(i, j)] = symmetric[r0:r1, c0:c1]
    return cross_impacts


def remap_cross_impacts(
    cross_impacts_by_ci: dict[tuple[int, int], np.ndarray], ci_index_of_factor: list[int]
) -> dict[tuple[int, int], np.ndarray]:
    """Relabel a CI-index-keyed cross-impact dict (as produced by
    cross_impacts_from_block_matrix) onto a network's own factor indices,
    re-orienting each block's rows/columns as needed."""
    remapped = {}
    n = len(ci_index_of_factor)
    for i in range(n):
        for j in range(i + 1, n):
            ci_i, ci_j = ci_index_of_factor[i], ci_index_of_factor[j]
            lo, hi = (ci_i, ci_j) if ci_i < ci_j else (ci_j, ci_i)
            block = cross_impacts_by_ci.get((lo, hi))
            if block is None:
                continue
            remapped[(i, j)] = block if ci_i < ci_j else block.T
    return remapped


def load_cross_impact_matrix(
    path: str | Path, states_by_ci_index: list[int], ci_index_of_factor: list[int], scale: str = "multiplier"
) -> dict[tuple[int, int], np.ndarray]:
    """Load a cross-impact CSV and remap it directly onto a network's factor
    indices. `states_by_ci_index[c]` is the state count of whichever factor
    occupies CSV block `c`; `ci_index_of_factor[i]` is the CSV block index
    for the network's i-th factor."""
    raw = np.loadtxt(path, delimiter=";", encoding="utf-8-sig")
    by_ci = cross_impacts_from_block_matrix(raw, states_by_ci_index, scale=scale)
    return remap_cross_impacts(by_ci, ci_index_of_factor)
