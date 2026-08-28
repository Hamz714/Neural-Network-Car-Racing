"""Turning one generation of networks into the next.

The search is mutation-driven hill climbing with elitism: score everybody, keep
the best few untouched, and fill the rest of the population with mutated copies
of the best twenty. The network has only 320 parameters, so this is a small
enough space for a simple algorithm to cross.

Three choices here are deliberate and worth being able to defend.

**Truncation selection, not fitness-proportional.** Roulette-wheel selection
needs non-negative fitness, which forces an arbitrary rescaling every
generation whenever a car scores below zero - and cars that crash immediately
routinely do. Ranking and taking the top slice needs no such fudge, and it is
invariant to how the scores happen to be scaled.

**Elitism guarantees monotone progress.** Carrying the best few through
unchanged means the best score can never fall, which is what makes a training
curve legible rather than a cloud of noise.

**Crossover is implemented but off by default.** Two networks that both drive
well may do so with entirely different internal arrangements - hidden unit
three in one plays the role of unit seven in the other. Averaging or splicing
their weights usually produces something worse than either, a failure known as
the competing conventions problem. Keeping it behind a flag makes it a
measurable ablation instead of an unexamined assumption.
"""

import random as _random

from nncar.neural_network import Network


class GAConfig:
    """Search hyper-parameters."""

    __slots__ = ("population", "elite", "parents", "sigma0", "sigma1",
                 "sigma_schedule", "crossover_rate", "random_inject", "tournament_k")

    def __init__(self, population=100, elite=5, parents=20, sigma0=0.15, sigma1=0.02,
                 sigma_schedule="exp", crossover_rate=0.0, random_inject=0.05,
                 tournament_k=3):
        self.population = population
        self.elite = elite
        self.parents = parents
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.sigma_schedule = sigma_schedule
        self.crossover_rate = crossover_rate
        self.random_inject = random_inject
        self.tournament_k = tournament_k

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    def __repr__(self):
        return "GAConfig(%s)" % ", ".join("%s=%r" % kv for kv in self.as_dict().items())


def elite_count(config):
    """At least one elite, whatever the population size.

    The original code computed the survivor count as ``population // 10``,
    which is zero for any population below ten - so selection returned an empty
    list and the next generation had nothing to descend from.
    """
    return max(1, min(config.elite, config.population))


def parent_count(config):
    return max(1, min(config.parents, config.population))


def initial_population(size, rng=None):
    return [Network(rng) for _ in range(size)]


def sigma_for(generation, generations, config):
    """Mutation step size for a generation.

    Decaying it turns a coarse early search into a fine late one: large steps
    while the population is bad and any direction might help, small steps once
    it is good and most large changes break something.
    """
    if config.sigma_schedule == "constant" or generations <= 1:
        return config.sigma0
    progress = generation / float(generations - 1)
    if config.sigma_schedule == "linear":
        return config.sigma0 + (config.sigma1 - config.sigma0) * progress
    # Exponential decay, the default: proportional steps rather than absolute.
    if config.sigma0 <= 0:
        return config.sigma0
    return config.sigma0 * (config.sigma1 / config.sigma0) ** progress


def rank(scored):
    """Sort (fitness, individual_id, network) triples, best first.

    Uses the project's own merge sort rather than sorted(): it is already
    written, already tested against sorted() including stability, and keeping
    it in the path that matters means it stays exercised. Ties break on
    individual_id so the order never depends on dictionary or scheduling
    order - which is what makes a run reproducible across worker counts.
    """
    from nncar.game import merge_sort

    by_id = merge_sort(list(scored), key=lambda entry: entry[1])
    return merge_sort(by_id, key=lambda entry: entry[0], reverse=True)


def select(scored, count):
    """The `count` best networks, best first."""
    return [network for _, _, network in rank(scored)[:count]]


def tournament(parents, rng, k=3):
    """Pick the best of k random parents.

    Softens truncation: every parent can be chosen, but better ones are chosen
    more often, so the population does not collapse onto a single lineage.
    """
    best = None
    best_index = None
    for _ in range(max(1, k)):
        index = rng.randrange(len(parents))
        if best_index is None or index < best_index:
            best_index = index
            best = parents[index]
    return best


def uniform_crossover(first, second, rng):
    """Take each parameter from one parent or the other, at random."""
    child = first.copy()
    for layer, other in zip(child.layers, second.layers):
        for row in range(len(layer.weights)):
            for column in range(len(layer.weights[row])):
                if rng.random() < 0.5:
                    layer.weights[row][column] = other.weights[row][column]
        for row in range(len(layer.bias)):
            if rng.random() < 0.5:
                layer.bias[row][0] = other.bias[row][0]
    return child


def next_generation(scored, config, sigma, rng):
    """Build the next population from the scored current one.

    All mutation happens here, in one process, drawing from one seeded stream -
    so a run with eight workers produces exactly the same networks as a run
    with one. Distributing it would be faster by a few milliseconds a
    generation and would cost that guarantee.
    """
    ranked = rank(scored)
    if not ranked:
        raise ValueError("cannot breed from an empty population")

    n_elite = elite_count(config)
    n_parents = parent_count(config)

    # Elites are copied, never carried by reference: a later mutation of a
    # child must not be able to reach back and alter the champion.
    children = [network.copy() for _, _, network in ranked[:n_elite]]
    parents = [network for _, _, network in ranked[:n_parents]]

    while len(children) < config.population:
        draw = rng.random()
        if draw < config.random_inject:
            child = Network(rng)
        elif draw < config.random_inject + config.crossover_rate:
            child = uniform_crossover(tournament(parents, rng, config.tournament_k),
                                      tournament(parents, rng, config.tournament_k),
                                      rng)
            child.mutate(sigma, rng)
        else:
            child = tournament(parents, rng, config.tournament_k).copy()
            child.mutate(sigma, rng)
        children.append(child)

    return children
