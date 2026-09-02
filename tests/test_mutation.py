"""The mutation operator - the search itself.

Gaussian perturbation of every weight and bias. These tests pin the statistical
properties the genetic algorithm relies on.
"""

import math
import random
import statistics

import pytest

import pytest

from nncar import neural_network as nn


def test_random_normal_is_standard_normal():
    random.seed(1)
    draws = [nn.random_normal() for _ in range(200_000)]
    # Standard error of the mean is 1/sqrt(n) ~ 0.0022; 4 sigma is comfortable.
    assert abs(statistics.mean(draws)) < 0.01
    assert abs(statistics.pstdev(draws) - 1.0) < 0.01


def test_random_normal_is_reproducible_from_a_seed():
    random.seed(42)
    first = [nn.random_normal() for _ in range(20)]
    random.seed(42)
    assert [nn.random_normal() for _ in range(20)] == first


def test_mutation_perturbs_every_parameter():
    random.seed(2)
    net = nn.Network()
    before = [[row[:] for row in layer.weights] for layer in net.layers]
    net.mutate()
    for layer, original in zip(net.layers, before):
        for new_row, old_row in zip(layer.weights, original):
            assert all(n != o for n, o in zip(new_row, old_row))


def test_mutation_step_size_matches_the_mutation_rate():
    """Each parameter moves by mutation_rate * N(0,1), so the empirical
    standard deviation of the deltas should recover mutation_rate."""
    random.seed(3)
    net = nn.Network()
    before = [v for layer in net.layers for row in layer.weights for v in row]
    net.mutate()
    after = [v for layer in net.layers for row in layer.weights for v in row]

    deltas = [a - b for a, b in zip(after, before)]
    assert len(deltas) == 288  # 12*6 + 10*12 + 8*10 + 2*8; the 32 biases are separate
    assert statistics.pstdev(deltas) == pytest.approx(nn.mutation_rate, rel=0.15)


def test_mutation_is_deterministic_under_a_seed():
    def mutated(seed):
        random.seed(seed)
        net = nn.Network()
        random.seed(seed + 1000)
        net.mutate()
        return [v for layer in net.layers for row in layer.weights for v in row]

    assert mutated(11) == mutated(11)
    assert mutated(11) != mutated(12)


def test_mutate_returns_self_for_chaining():
    random.seed(4)
    net = nn.Network()
    assert net.mutate() is net


def test_box_muller_never_takes_log_of_zero():
    """random.random() can in principle return exactly 0.0, which would make
    math.log raise. The probability is ~2^-53 per draw, so this documents the
    hazard rather than exercising it."""
    assert math.isfinite(nn.random_normal())
