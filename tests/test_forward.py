"""End-to-end forward propagation."""

import random

import numpy as np

from nncar import neural_network as nn


class FakeCar:
    """forward_propagation only touches .network and .inputs."""

    def __init__(self, network, inputs):
        self.network = network
        self.inputs = inputs


def _inputs(rng):
    return [[rng.uniform(-1, 1)] for _ in range(6)]


def test_output_is_two_values_in_range():
    random.seed(3)
    net = nn.Network()
    accelerate, turn = nn.forward_propagation(FakeCar(net, _inputs(random.Random(4))))
    assert -1.0 <= accelerate <= 1.0
    assert -1.0 <= turn <= 1.0


def test_same_inputs_give_bit_identical_outputs():
    random.seed(5)
    net = nn.Network()
    rng = random.Random(6)
    x = _inputs(rng)
    first = nn.forward_propagation(FakeCar(net, [row[:] for row in x]))
    second = nn.forward_propagation(FakeCar(net, [row[:] for row in x]))
    assert first == second


def test_matches_a_numpy_reference():
    random.seed(8)
    net = nn.Network()
    rng = random.Random(9)
    x = _inputs(rng)

    a = np.array(x)
    for layer in net.layers:
        a = np.matmul(np.array(layer.weights), a) + np.array(layer.bias)
        a = np.where(a < -20, -1.0, np.where(a > 20, 1.0, np.tanh(a)))

    got = nn.forward_propagation(FakeCar(net, [row[:] for row in x]))
    assert np.allclose(got, a.ravel(), atol=1e-12)


def test_raw_pixel_inputs_saturate_the_first_layer():
    """Why input normalisation is mandatory, pinned as a measurement.

    Sensor distances reach 700 while weights are drawn from N(0,1), so the
    first-layer pre-activations average |z| ~ 1000 and all but a fraction of a
    percent of hidden units clamp to exactly +/-1. Layer one degenerates into
    sign(z): piecewise constant, and therefore nearly invisible to a search
    that only ever perturbs weights slightly.

    Averaged over 200 random networks, so it measures the architecture rather
    than one lucky seed.
    """
    raw = [[500.0], [600.0], [700.0], [600.0], [500.0], [5.0]]
    normalised = [[1.0], [1.0], [1.0], [1.0], [1.0], [5 / 12]]

    def profile(inputs):
        magnitudes, saturated = [], []
        for seed in range(200):
            random.seed(seed)
            layer = nn.Network().layers[0]
            z = layer.forward([row[:] for row in inputs])
            magnitudes.append(sum(abs(v[0]) for v in z) / len(z))
            a = layer.activation([row[:] for row in z])
            saturated.append(sum(1 for v in a if abs(v[0]) > 0.999) / len(a))
        return sum(magnitudes) / len(magnitudes), sum(saturated) / len(saturated)

    raw_z, raw_saturated = profile(raw)
    norm_z, norm_saturated = profile(normalised)

    assert raw_z > 500, "expected pre-activations around 10^3, got %.1f" % raw_z
    assert raw_saturated > 0.98, "expected near-total saturation, got %.3f" % raw_saturated

    assert norm_z < 5, "normalised pre-activations should sit near unity, got %.2f" % norm_z
    assert norm_saturated < 0.35, "normalised layer should stay responsive, got %.3f" % norm_saturated


def _reference(net, inputs):
    """What forward_propagation computes, expressed with the matrix helpers.

    This is the readable definition; forward_propagation is the flattened one
    that actually runs. They must agree exactly.
    """
    values = [row[:] for row in inputs]
    for layer in net.layers:
        values = layer.activation(layer.forward(values))
    return values[0][0], values[1][0]


def test_fused_forward_pass_is_bit_exact():
    """The hot path is hand-fused; it must not drift from the reference.

    Checked over 500 random networks rather than a handful, because the
    failure this guards against - a slightly different summation order - shows
    up in the last bit or two and only on some inputs.
    """
    mismatches = 0
    for seed in range(500):
        random.seed(seed)
        net = nn.Network()
        rng = random.Random(seed + 10_000)
        x = [[rng.uniform(-3, 3)] for _ in range(6)]
        if nn.forward_propagation(FakeCar(net, [row[:] for row in x])) != _reference(net, x):
            mismatches += 1
    assert mismatches == 0, "%d of 500 networks disagreed with the reference" % mismatches


def test_fused_forward_pass_is_bit_exact_on_saturating_inputs():
    """The clamp branch is where a rewrite is most likely to diverge."""
    extremes = [-1e6, -700.0, -21.0, -20.0, 0.0, 20.0, 21.0, 700.0, 1e6]
    for seed in range(50):
        random.seed(seed)
        net = nn.Network()
        rng = random.Random(seed)
        x = [[rng.choice(extremes)] for _ in range(6)]
        assert nn.forward_propagation(FakeCar(net, [row[:] for row in x])) == _reference(net, x)


def test_matrix_helpers_are_still_exercised():
    """The reference path is kept deliberately; it should not rot unused."""
    random.seed(1)
    layer = nn.Layer(6, 12)
    x = [[0.5]] * 6
    assert len(layer.activation(layer.forward(x))) == 12
