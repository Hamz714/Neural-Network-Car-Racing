"""Regenerate tests/golden/trajectory_golden.json.

Runs five networked cars for 400 fully seeded ticks and records the state of
each one every 20 ticks. Because the physics is deterministic, any change to
the simulation shows up here - which is the point: the file is a tripwire, and
regenerating it should always be a deliberate act described in a commit message.

    python tests/golden/_generate_trajectory.py
"""

import json
import os
import random
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["NNCAR_HEADLESS"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "src"))

SEED = 20240828
TICKS = 400
SAMPLE_EVERY = 20
FIELDS = ["world_x", "world_y", "angle", "velocity", "collisions",
          "checkpoints_passed", "laps"]
OUT = os.path.join(HERE, "trajectory_golden.json")


def state(car):
    return [round(car.world_x, 6), round(car.world_y, 6), round(car.angle, 6),
            round(car.velocity, 6), car.collisions, car.checkpoints_passed, car.laps]


def run(ticks=TICKS):
    """Five seeded networks driving for `ticks` frames.

    The networks are generated from SEED rather than loaded from models/, so
    this pins the *simulation* and stays valid when the shipped opponents are
    retrained. Anything that changes physics, sensing or collision still shows
    up here immediately.
    """
    from copy import deepcopy

    import pygame

    from nncar import entities as v
    from nncar import game as f
    from nncar import neural_network as nn

    pygame.init()
    random.seed(SEED)

    v.reset_race_timer()
    v.player = v.PlayerCar()
    v.track = v.Track(1, load_visuals=False)
    v.NPC.start_positions = deepcopy(v.NPC_START_POS)
    v.NPC_cars = [v.NPC(v.CARS[index][0], 0, nn.Network()) for index in range(5)]
    for car in v.NPC_cars:
        car.normalise_inputs = True
    v.track.leaderboard = v.NPC_cars + [v.player]

    rows = {}
    for tick in range(ticks):
        f.move()
        f.checkpoints()
        f.NPC_collision()
        f.update_leaderboard()
        if tick % SAMPLE_EVERY == 0:
            rows[str(tick)] = [state(car) for car in v.NPC_cars]
    return rows


if __name__ == "__main__":
    payload = {
        "_comment": ("Seeded NPC trajectory, sampled every %d ticks. Regenerate with "
                     "tests/golden/_generate_trajectory.py, and only when a behaviour "
                     "change is intended - say which in the commit message." % SAMPLE_EVERY),
        "seed": SEED,
        "ticks": TICKS,
        "sample_every": SAMPLE_EVERY,
        "fields": FIELDS,
        "rows": run(),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote", OUT, "-", len(payload["rows"]), "sampled ticks")
