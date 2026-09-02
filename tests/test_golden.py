"""Pin the network's numerical behaviour to a snapshot of the original code.

golden/network_golden.json was generated against the pre-refactor implementation.
The point of the file is that no restructure, optimisation or "tidy-up" may
change the forward pass, the RNG stream or the mutation operator without
someone noticing.

Shapes, counts and the order in which the RNG is consumed are asserted
exactly. The floats are asserted to TOLERANCE instead, because they cannot be
bit-identical everywhere: every transcendental the network uses - math.log and
math.cos inside Box-Muller, math.exp inside tanh - is evaluated by the
platform's libm, and neither IEEE-754 nor CPython promises two libm
implementations agree on the last bit. They do not. Draw 77 of the seeded
stream is one ulp lower on glibc than on the MSVC runtime this file was
recorded against, and the forward pass carries that through four layers into a
relative difference of a few parts in 1e15.

TOLERANCE is roughly two hundred times wider than the largest divergence
observed and still many orders of magnitude tighter than any change of
algorithm: re-seeding, reordering an accumulation or reclaiming Box-Muller's
discarded sin() term all move these numbers in the leading digits, not the
thirteenth.
"""

import json
import os
import random

import pytest

from nncar import neural_network as nn

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "network_golden.json")

#: Wide enough for libm-to-libm noise, narrow enough that a behaviour change
#: cannot hide underneath it. See the module docstring.
REL_TOLERANCE = 1e-12
ABS_TOLERANCE = 1e-12


def assert_matches(got, want, what):
    """Compare nested lists elementwise: structure exactly, floats to TOLERANCE."""
    if isinstance(want, list):
        assert isinstance(got, list), "%s: expected a list, got %r" % (what, got)
        assert len(got) == len(want), (
            "%s: length changed - golden has %d entries, got %d"
            % (what, len(want), len(got))
        )
        for index, (g, w) in enumerate(zip(got, want)):
            assert_matches(g, w, "%s[%d]" % (what, index))
        return

    assert not isinstance(got, list), (
        "%s: shape changed - golden has the single value %r, got the list %r"
        % (what, want, got)
    )
    assert got == pytest.approx(want, rel=REL_TOLERANCE, abs=ABS_TOLERANCE), (
        "%s: %r differs from golden %r by %.3g" % (what, got, want, abs(got - want))
    )


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


def test_seeded_initialisation_matches_golden(golden):
    net = _seeded_network()
    for index, (layer, (weights, bias)) in enumerate(zip(net.layers, golden["seeded_weights"])):
        assert_matches(layer.weights, weights, "layer %d weights" % index)
        assert_matches(layer.bias, bias, "layer %d bias" % index)


def test_forward_pass_matches_golden(golden):
    net = _seeded_network()

    class FakeCar:
        pass

    for index, case in enumerate(golden["forward_cases"]):
        car = FakeCar()
        car.network = net
        car.inputs = [row[:] for row in case["inputs"]]
        assert_matches(list(nn.forward_propagation(car)), case["output"],
                       "forward case %d" % index)


def test_mutation_stream_matches_golden(golden):
    random.seed(99)
    net = nn.Network()
    for index, expected in enumerate(golden["mutation_trace"]):
        net.mutate()
        got = net.layers[0].weights[0][:3] + net.layers[-1].bias[0][:1]
        assert_matches(got, expected, "mutation %d" % index)


def test_random_normal_stream_matches_golden(golden):
    """Box-Muller consumes exactly two random() draws per call.

    Pinning this stream is what makes every seeded test in the suite
    reproducible, and it is why the wasted sin() term is deliberately left
    unfixed - reclaiming it would shift every downstream value.
    """
    random.seed(7)
    assert_matches([nn.random_normal() for _ in range(10)],
                   golden["random_normal_stream"], "random_normal stream")


def test_random_normal_consumes_two_uniforms_per_call():
    """The half of the stream contract that *is* bit-exact everywhere.

    Mersenne Twister is integer arithmetic, so the uniforms underneath
    Box-Muller are identical on every platform. Consuming a different number
    of them - or consuming them in a different order - would reshuffle every
    seeded value in the project, and this catches it exactly, where the
    tolerance above could not.
    """
    random.seed(7)
    raw = [random.random() for _ in range(21)]

    random.seed(7)
    [nn.random_normal() for _ in range(10)]

    assert random.random() == raw[20], (
        "random_normal no longer consumes exactly two uniforms per call")


def test_tanh_probe_matches_golden(golden):
    got = [nn.Layer.tanh(None, x) for x in (-3.0, -0.5, 0.0, 0.5, 3.0, 19.9)]
    assert_matches(got, golden["tanh_probe"], "tanh probe")
