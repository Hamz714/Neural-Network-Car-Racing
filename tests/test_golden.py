"""Pin the network's numerical behaviour to a snapshot of the original code.

golden/network_golden.json was generated against the pre-refactor implementation.
Every assertion here is exact: the point of the file is that no restructure,
optimisation or "tidy-up" may change a single bit of the forward pass, the RNG
stream or the mutation operator without someone noticing.
"""

import json
import os
import random

import pytest

from nncar import neural_network as nn

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "network_golden.json")


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


def _seeded_network(seed=20240828):
    random.seed(seed)
    return nn.Network()


def test_architecture_unchanged(golden):
    net = _seeded_network()
    got = [[len(layer.weights[0]), len(layer.weights)] for layer in net.layers]
    assert got == [list(pair) for pair in golden["architecture"]]


def test_parameter_count_is_320(golden):
    """6->12->10->8->2 fully connected with biases."""
    net = _seeded_network()
    count = sum(len(l.weights) * len(l.weights[0]) + len(l.bias) for l in net.layers)
    assert count == 320
    assert count == golden["parameter_count"]


def test_seeded_initialisation_is_bit_identical(golden):
    net = _seeded_network()
    for layer, (weights, bias) in zip(net.layers, golden["seeded_weights"]):
        assert layer.weights == weights
        assert layer.bias == bias


def test_forward_pass_is_bit_identical(golden):
    net = _seeded_network()

    class FakeCar:
        pass

    for case in golden["forward_cases"]:
        car = FakeCar()
        car.network = net
        car.inputs = [row[:] for row in case["inputs"]]
        assert list(nn.forward_propagation(car)) == case["output"]


def test_mutation_stream_is_bit_identical(golden):
    random.seed(99)
    net = nn.Network()
    for expected in golden["mutation_trace"]:
        net.mutate()
        got = net.layers[0].weights[0][:3] + net.layers[-1].bias[0][:1]
        assert got == expected


def test_random_normal_stream_is_bit_identical(golden):
    """Box-Muller consumes exactly two random() draws per call.

    Pinning this stream is what makes every seeded test in the suite
    reproducible, and it is why the wasted sin() term is deliberately left
    unfixed - reclaiming it would shift every downstream value.
    """
    random.seed(7)
    assert [nn.random_normal() for _ in range(10)] == golden["random_normal_stream"]


def test_tanh_probe_is_bit_identical(golden):
    got = [nn.Layer.tanh(None, x) for x in (-3.0, -0.5, 0.0, 0.5, 3.0, 19.9)]
    assert got == golden["tanh_probe"]
