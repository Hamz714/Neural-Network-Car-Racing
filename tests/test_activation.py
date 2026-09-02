"""tanh and the saturation clamp."""

import math

import pytest

from nncar import neural_network as nn


@pytest.mark.parametrize("x", [-19.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 19.0])
def test_tanh_matches_stdlib(x):
    assert nn.Layer.tanh(None, x) == pytest.approx(math.tanh(x), abs=1e-12)


def test_clamp_saturates_beyond_twenty():
    layer = nn.Layer(1, 1)
    assert layer.activation([[1000.0]]) == [[1]]
    assert layer.activation([[-1000.0]]) == [[-1]]


def test_clamp_prevents_overflow():
    """The naive (e^x - e^-x)/(e^x + e^-x) form overflows around x=710.

    The +/-20 clamp is what keeps math.exp reachable-only in its safe range, so
    this is load-bearing rather than cosmetic.
    """
    layer = nn.Layer(1, 1)
    for extreme in (1e3, 1e6, -1e6):
        layer.activation([[extreme]])  # must not raise OverflowError

    with pytest.raises(OverflowError):
        nn.Layer.tanh(None, 1e6)


def test_activation_mutates_in_place():
    """Documents an aliasing hazard relied on by the forward pass."""
    layer = nn.Layer(1, 1)
    data = [[0.5]]
    out = layer.activation(data)
    assert out is data
