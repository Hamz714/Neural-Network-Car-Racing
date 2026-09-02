"""The physics is deterministic, and stays bit-for-bit what it was.

The simulation is a chaotic system: a car's next position depends on rays cast
from its current one, so a difference of one ulp anywhere compounds into a
visibly different trajectory within a few hundred ticks. That makes a recorded
trajectory an unusually sharp instrument - it catches refactors that unit tests
would wave through.

golden/trajectory_golden.json was verified by running the same seeded scenario
against the pre-refactor code and confirming both produced identical numbers
across 400 ticks, 5 cars and 7 fields.

When a change to driving behaviour is intended, regenerate the file with
tests/golden/_generate_trajectory.py and say so in the commit message. Silent
regeneration defeats the purpose.
"""

import json
import os

import pytest

from conftest import requires_pygame

pytestmark = requires_pygame

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden", "trajectory_golden.json")


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def replayed(golden):
    from golden._generate_trajectory import run

    return run(golden["ticks"])


def test_trajectory_is_bit_identical(golden, replayed):
    expected = golden["rows"]
    assert set(replayed) == set(expected)

    for tick in sorted(expected, key=int):
        for index, (got, want) in enumerate(zip(replayed[tick], expected[tick])):
            assert got == want, (
                "car %d diverged at tick %s\n  expected %s\n  got      %s\n  fields   %s"
                % (index, tick, want, got, golden["fields"])
            )


def test_cars_actually_move(golden, replayed):
    """Guards against the trajectory passing because nothing happened.

    The networks are random, so some of them stall against a wall within a few
    ticks and genuinely do not move - that is what an untrained population
    looks like. The check is that the simulation is running, not that every car
    drives well.
    """
    first = replayed["0"]
    last = replayed[max(replayed, key=int)]
    moved = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(first, last)]

    assert sum(moved) > 500, "the whole population barely moved: %s" % moved
    assert sum(1 for d in moved if d > 50) >= 3,         "expected most cars to get somewhere: %s" % moved


def test_each_car_has_its_own_network(golden):
    """The original loader aliased one Network across all five opponents."""
    from copy import deepcopy

    from nncar import entities as v
    from nncar import game as f

    v.NPC.start_positions = deepcopy(v.NPC_START_POS)
    cars = f.load("easy")
    assert len({id(car.network) for car in cars}) == len(cars)


def test_the_golden_run_does_not_depend_on_the_shipped_models(golden):
    """The tripwire pins the simulation, not whichever opponents ship.

    Generating the networks from a seed keeps it valid across retraining, so a
    failure here always means the physics moved.
    """
    import inspect

    from golden import _generate_trajectory

    source = inspect.getsource(_generate_trajectory.run)
    assert "nn.Network()" in source, "networks should be generated from a seed"
    assert 'f.load("' not in source, "must not load a shipped model"
    assert 'load_model' not in source
