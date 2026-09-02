"""The genetic algorithm, driven by a stand-in evaluator.

Scoring networks by a cheap deterministic proxy rather than by simulating them
keeps these tests fast and makes the properties being checked - elitism,
constant population, reproducibility - properties of the search rather than of
the track.
"""

import pickle
import random

import pytest

from nncar.ga import population as ga
from nncar.ga.evaluate import derive_seed
from nncar.neural_network import Network


def proxy_fitness(network):
    """A deterministic score standing in for a rollout."""
    return sum(network.layers[0].weights[0])


def scored_population(networks):
    return [(proxy_fitness(net), index, net) for index, net in enumerate(networks)]


def evolve(generations=6, size=12, seed=7, **kwargs):
    config = ga.GAConfig(population=size, elite=2, parents=5, **kwargs)
    rng = random.Random(seed)
    networks = ga.initial_population(size, rng)
    history = []
    for generation in range(generations):
        scored = scored_population(networks)
        history.append(max(score for score, _, _ in scored))
        sigma = ga.sigma_for(generation, generations, config)
        networks = ga.next_generation(scored, config, sigma, rng)
    return history, networks


def test_elitism_makes_the_best_score_monotone():
    history, _ = evolve()
    assert history == sorted(history), "best fitness fell: %s" % history


def test_population_size_is_constant():
    _, networks = evolve(size=12)
    assert len(networks) == 12


def test_a_seeded_run_reproduces_exactly():
    first, networks_a = evolve(seed=3)
    second, networks_b = evolve(seed=3)
    assert first == second
    for a, b in zip(networks_a, networks_b):
        assert a.layers[0].weights == b.layers[0].weights


def test_different_seeds_diverge():
    first, _ = evolve(seed=3)
    second, _ = evolve(seed=4)
    assert first != second


@pytest.mark.parametrize("size", [1, 2, 5, 9])
def test_at_least_one_elite_at_any_population_size(size):
    """The original computed survivors as population // 10.

    For any population under ten that is zero, so selection returned nothing
    and there was no parent for the next generation to descend from.
    """
    config = ga.GAConfig(population=size, elite=5)
    assert ga.elite_count(config) >= 1
    assert ga.elite_count(config) <= size


def test_breeding_from_a_tiny_population_still_works():
    config = ga.GAConfig(population=5, elite=5, parents=20)
    rng = random.Random(1)
    networks = ga.initial_population(5, rng)
    children = ga.next_generation(scored_population(networks), config, 0.1, rng)
    assert len(children) == 5


def test_elites_are_copies_not_aliases():
    """A mutated child must not be able to reach back into the champion."""
    config = ga.GAConfig(population=6, elite=2, parents=3, random_inject=0.0)
    rng = random.Random(11)
    networks = ga.initial_population(6, rng)
    scored = scored_population(networks)
    best = ga.rank(scored)[0][2]
    snapshot = [row[:] for row in best.layers[0].weights]

    children = ga.next_generation(scored, config, 0.5, rng)
    for child in children:
        assert child is not best
    assert best.layers[0].weights == snapshot


def test_ranking_is_best_first_and_breaks_ties_deterministically():
    nets = ga.initial_population(4, random.Random(0))
    scored = [(1.0, 3, nets[0]), (5.0, 1, nets[1]), (1.0, 0, nets[2]), (5.0, 2, nets[3])]
    ranked = ga.rank(scored)
    assert [entry[0] for entry in ranked] == [5.0, 5.0, 1.0, 1.0]
    # Equal scores order by individual id, never by input order.
    assert [entry[1] for entry in ranked] == [1, 2, 0, 3]


def test_sigma_decays_from_the_start_value_to_the_final_one():
    config = ga.GAConfig(sigma0=0.15, sigma1=0.02, sigma_schedule="exp")
    assert ga.sigma_for(0, 100, config) == pytest.approx(0.15)
    assert ga.sigma_for(99, 100, config) == pytest.approx(0.02)
    values = [ga.sigma_for(g, 100, config) for g in range(100)]
    assert values == sorted(values, reverse=True)


def test_constant_schedule_does_not_decay():
    config = ga.GAConfig(sigma0=0.1, sigma_schedule="constant")
    assert {ga.sigma_for(g, 50, config) for g in range(50)} == {0.1}


def test_mutation_respects_the_step_size():
    rng = random.Random(5)
    parent = Network(rng)
    before = [v for layer in parent.layers for row in layer.weights for v in row]

    small = parent.copy()
    small.mutate(0.001, random.Random(1))
    large = parent.copy()
    large.mutate(0.5, random.Random(1))

    def spread(net):
        after = [v for layer in net.layers for row in layer.weights for v in row]
        return max(abs(a - b) for a, b in zip(after, before))

    assert spread(small) < spread(large)


def test_zero_sigma_leaves_a_network_unchanged():
    rng = random.Random(2)
    parent = Network(rng)
    before = [row[:] for row in parent.layers[0].weights]
    parent.copy().mutate(0.0, random.Random(9))
    child = parent.copy()
    child.mutate(0.0, random.Random(9))
    assert child.layers[0].weights == before


def test_crossover_takes_from_both_parents():
    rng = random.Random(4)
    first = Network(rng)
    second = Network(rng)
    for layer in first.layers:
        for row in range(len(layer.weights)):
            for column in range(len(layer.weights[row])):
                layer.weights[row][column] = 0.0
    for layer in second.layers:
        for row in range(len(layer.weights)):
            for column in range(len(layer.weights[row])):
                layer.weights[row][column] = 1.0

    child = ga.uniform_crossover(first, second, random.Random(6))
    values = {v for layer in child.layers for row in layer.weights for v in row}
    assert values == {0.0, 1.0}


def test_crossover_is_off_by_default():
    """It is implemented so it can be measured, not assumed to help."""
    assert ga.GAConfig().crossover_rate == 0.0


def test_derived_seeds_do_not_depend_on_scheduling():
    assert derive_seed(1234, 3, 7, 1) == derive_seed(1234, 3, 7, 1)
    assert derive_seed(1234, 3, 7, 1) != derive_seed(1234, 3, 7, 0)
    assert derive_seed(1234, 3, 7, 1) != derive_seed(1234, 4, 7, 1)
    assert derive_seed(1234, 3, 7, 1) != derive_seed(9999, 3, 7, 1)

    seeds = [derive_seed(1, g, i, s)
             for g in range(20) for i in range(20) for s in range(2)]
    assert len(set(seeds)) == len(seeds), "seed collision"


def test_a_task_payload_stays_small():
    """The occupancy grid is 11 MB and must never travel with a task.

    If it ever leaks into the payload, training does not break - it just gets
    mysteriously slow, which is far harder to notice.
    """
    from nncar.ga.evaluate import Evaluator
    from nncar.sim.rollout import RolloutConfig

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.config = RolloutConfig()
    evaluator.start_indices = (0, 1)
    evaluator._pool = None

    networks = ga.initial_population(2, random.Random(0))
    payload = next(iter(evaluator.payloads(networks, 0, 1234)))
    assert len(pickle.dumps(payload, protocol=4)) < 64_000


def test_a_network_survives_a_pickle_round_trip():
    net = Network(random.Random(8))
    restored = pickle.loads(pickle.dumps(net, protocol=4))
    assert restored.layers[0].weights == net.layers[0].weights
    assert len(restored.layers) == len(net.layers)
