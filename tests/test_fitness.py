"""The fitness function, and the two ways it could quietly go wrong.

Everything here is a pure function of a RolloutResult, so none of it needs
pygame or a track.
"""

import pytest

from nncar.sim.fitness import DEFAULT_WEIGHTS, FitnessWeights, check_weights, fitness
from nncar.sim.rollout import RolloutResult


def result(**kwargs):
    """Build a RolloutResult.

    `total_gates` is the sum across circuits and is descriptive only;
    `best_circuit_gates` is what fitness scores. Tests that set one and mean
    both get both, unless they are specifically exercising the difference.
    """
    defaults = dict(ticks=500, total_gates=0, valid_laps=0, lap_gates=(),
                    collisions=0, terminated="stall")
    defaults.update(kwargs)
    defaults.setdefault("best_circuit_gates", defaults["total_gates"])
    return RolloutResult(**defaults)


def test_more_checkpoints_always_scores_higher():
    scores = [fitness(result(total_gates=n)) for n in range(11)]
    assert scores == sorted(scores)
    assert all(b > a for a, b in zip(scores, scores[1:]))


def test_equal_progress_prefers_the_faster_car():
    quick = fitness(result(total_gates=5, ticks=400))
    slow = fitness(result(total_gates=5, ticks=1200))
    assert quick > slow


def test_progress_beats_speed_when_they_conflict():
    """A car that gets further must win even if it took much longer.

    This is why speed enters as a rate rather than as a time penalty: a
    subtracted time term would eventually outweigh a checkpoint and start
    rewarding cars for stopping sooner.
    """
    further_but_slower = fitness(result(total_gates=6, ticks=2900))
    nearer_but_quicker = fitness(result(total_gates=5, ticks=200))
    assert further_but_slower > nearer_but_quicker


def test_standing_still_is_never_optimal():
    """The failure mode the collision weight is sized against.

    A car that never moves takes no penalty at all. If crashing cost more than
    a checkpoint pays, doing nothing would be the global optimum and the
    population would converge on it.
    """
    idle = fitness(result(total_gates=0, collisions=0, ticks=3000, terminated="stall"))
    tried_and_crashed = fitness(result(total_gates=1, collisions=5, ticks=300,
                                       terminated="crash"))
    assert tried_and_crashed > idle


def test_the_weight_invariant_is_checked():
    check_weights(DEFAULT_WEIGHTS, collision_limit=5)

    reckless = FitnessWeights(progress=100.0, collision=25.0)
    with pytest.raises(ValueError, match="stand still"):
        check_weights(reckless, collision_limit=5)


def test_the_free_lap_exploit_scores_nothing():
    """The regression test for the reward hack this design exists to close.

    Crossing the finish line northbound increments the lap counter, and every
    spawn point is north of it - so reversing a short distance and driving
    forward again banks laps indefinitely without passing a single checkpoint.
    Because progress is counted in checkpoints and each lap records how many it
    cleared, that behaviour is worth exactly zero.
    """
    exploiter = result(total_gates=0, best_circuit_gates=0, valid_laps=0,
                       lap_gates=(0, 0, 0, 0, 0), ticks=200, collisions=0)
    assert fitness(exploiter) == 0.0

    honest = result(total_gates=3, valid_laps=0, lap_gates=(), ticks=800)
    assert fitness(honest) > fitness(exploiter)


def test_a_genuine_lap_is_rewarded():
    lap = result(total_gates=10, valid_laps=1, lap_gates=(10,), ticks=1500,
                 terminated="finished")
    partial = result(total_gates=9, valid_laps=0, ticks=1500)
    assert fitness(lap) > fitness(partial)


def test_finishing_sooner_pays_more():
    quick = result(total_gates=10, valid_laps=1, lap_gates=(10,), ticks=900,
                   terminated="finished")
    slow = result(total_gates=10, valid_laps=1, lap_gates=(10,), ticks=2900,
                  terminated="finished")
    assert fitness(quick) > fitness(slow)


def test_collisions_are_penalised_but_only_mildly():
    clean = fitness(result(total_gates=4, collisions=0))
    scraped = fitness(result(total_gates=4, collisions=4))
    assert clean > scraped
    # Still worth far more than having gone nowhere cleanly.
    assert scraped > fitness(result(total_gates=0, collisions=0))


def test_negative_scores_are_allowed():
    """Truncation selection ranks, so nothing needs shifting positive."""
    assert fitness(result(total_gates=0, collisions=3)) < 0


def test_is_a_pure_function_of_the_result():
    sample = result(total_gates=4, collisions=1)
    assert fitness(sample) == fitness(sample)
    snapshot = sample.as_dict()
    fitness(sample)
    assert sample.as_dict() == snapshot


def test_zero_ticks_does_not_divide_by_zero():
    assert isinstance(fitness(result(ticks=0, total_gates=1)), float)


def test_two_sloppy_laps_lose_to_one_clean_one():
    """The second reward hack, and the reason progress is a single circuit.

    Summing checkpoints across circuits pays a car that clips two corners on
    each of two laps sixteen, against ten for a car that drives one lap
    properly - so the search learns to take a scenic route and skip gates. The
    first trained champion came back with lap_gates of (7, 8) and never once
    cleared all ten.
    """
    sloppy = result(total_gates=15, best_circuit_gates=8, valid_laps=0,
                    lap_gates=(7, 8), ticks=1600)
    clean = result(total_gates=10, best_circuit_gates=10, valid_laps=1,
                   lap_gates=(10,), ticks=900, terminated="finished")
    assert fitness(clean) > fitness(sloppy)


def test_repeating_a_partial_circuit_never_helps():
    once = result(total_gates=8, best_circuit_gates=8, lap_gates=(8,), ticks=800)
    twice = result(total_gates=16, best_circuit_gates=8, lap_gates=(8, 8), ticks=1600)
    # More laps of the same quality must not outscore one - it is only slower.
    assert fitness(twice) < fitness(once)


def test_a_lap_requires_every_checkpoint():
    from nncar.sim.rollout import RolloutConfig

    assert RolloutConfig().min_gates_per_lap == 10
