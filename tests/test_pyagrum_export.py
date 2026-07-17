"""to_pyagrum: exporting a fitted network into a gum.BayesNet for inference
and visualization -- the pyAgrum equivalent of genie_parser.m's GeNIe export
in the original MATLAB code.
"""

import itertools

import numpy as np
import pyagrum as gum
import pytest

from cross_impact import CrossImpactNetwork, UncertaintyFactor, build_network, energy_network, printing_network, to_pyagrum


def test_structure_matches_network():
    network = energy_network()
    built = build_network(network, mode="exact")
    bn = to_pyagrum(built)

    node_names = {bn.variable(i).name() for i in bn.nodes()}
    assert node_names == {f.name for f in network.factors}

    arcs = {(bn.variable(a).name(), bn.variable(b).name()) for a, b in bn.arcs()}
    expected_arcs = {
        (network.factors[p].name, factor.name) for factor in network.factors for p in factor.parents
    }
    assert arcs == expected_arcs


def test_cpts_match_fitted_cpts():
    network = energy_network()
    built = build_network(network, mode="exact")
    bn = to_pyagrum(built)

    for i, factor in enumerate(network.factors):
        parents = sorted(factor.parents)
        parent_dims = [network.factors[p].n_states for p in parents]
        cpt = built.cpts[i]
        for config in itertools.product(*(range(d) for d in parent_dims)) or [()]:
            parent_inst = {network.factors[p].name: s for p, s in zip(parents, config)}
            expected = cpt[(slice(None), *config)]
            actual = np.array([bn.cpt(factor.name)[{**parent_inst, factor.name: k}] for k in range(factor.n_states)])
            assert np.allclose(expected, actual)


def test_inference_reproduces_fitted_marginals():
    network = energy_network()
    built = build_network(network, mode="exact")
    bn = to_pyagrum(built)

    ie = gum.LazyPropagation(bn)
    ie.makeInference()
    for i, factor in enumerate(network.factors):
        posterior = ie.posterior(factor.name).toarray()
        other_axes = tuple(a for a in range(built.joint.ndim) if a != i)
        expected = built.joint.sum(axis=other_axes)
        assert np.allclose(posterior, expected, atol=1e-5)


def test_exports_printing_network():
    built = build_network(printing_network(), mode="exact")
    bn = to_pyagrum(built)
    assert bn.size() == 11
    ie = gum.LazyPropagation(bn)
    ie.makeInference()  # just needs to not raise


def test_custom_state_labels():
    network = energy_network()
    built = build_network(network, mode="exact")
    labels = {0: ["low", "medium", "high"]}
    bn = to_pyagrum(built, state_labels=labels)
    assert list(bn.variable("Nuclear power").labels()) == ["low", "medium", "high"]
    # untouched factor still gets the default numeric labels
    assert list(bn.variable("Renewables").labels()) == ["0", "1", "2", "3"]


def test_wrong_number_of_state_labels_raises():
    built = build_network(energy_network(), mode="exact")
    with pytest.raises(ValueError, match="expected 3"):
        to_pyagrum(built, state_labels={0: ["only_one_label"]})


def test_duplicate_names_raise():
    network = CrossImpactNetwork(
        factors=[UncertaintyFactor("same", 2, ()), UncertaintyFactor("same", 2, ())],
        marginals=[np.array([0.5, 0.5]), np.array([0.5, 0.5])],
        cross_impacts={},
    )
    built = build_network(network, mode="exact")
    with pytest.raises(ValueError, match="unique"):
        to_pyagrum(built)
