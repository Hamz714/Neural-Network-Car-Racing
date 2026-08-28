"""Measure the simulation's hot paths and write the results to JSON.

Both implementations of each hot path are timed in the same process, on the
same machine, moments apart - so the comparison is not two numbers from two
different days. Where an old and a new implementation are compared, the
benchmark first asserts they agree; a speedup between two functions that
compute different things is not a speedup.

Method, applied throughout:
  * time.perf_counter, garbage collector disabled during timing
  * one discarded warm-up pass
  * REPEATS independent repeats, reporting the minimum, with median and
    standard deviation recorded alongside
  * no display: SDL runs on its dummy driver

Raycasting is reported per regime rather than as a single aggregate. Its old
cost depended entirely on how far the ray travelled before hitting something,
so one averaged number would hide the fact that the benefit is concentrated in
exactly the open-space case an untrained network spends all its time in.

    python scripts/benchmark.py --out docs/optimised.json --label optimised
"""

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("NNCAR_HEADLESS", "1")

import _bootstrap  # noqa: F401,E402

REPEATS = 7


def timed(fn, iterations, repeats=REPEATS):
    """Per-iteration microseconds: minimum, median, standard deviation."""
    fn()
    samples = []
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iterations):
                fn()
            samples.append((time.perf_counter() - start) / iterations * 1e6)
    finally:
        gc.enable()
    return {
        "us_min": round(min(samples), 3),
        "us_median": round(statistics.median(samples), 3),
        "us_stdev": round(statistics.stdev(samples), 3) if len(samples) > 1 else 0.0,
        "iterations": iterations,
        "repeats": repeats,
    }


def bench_forward():
    """One forward pass through the 6-12-10-8-2 network."""
    import random

    from nncar import neural_network as nn

    random.seed(0)
    net = nn.Network()

    class Car:
        pass

    car = Car()
    car.network = net
    template = [[0.9], [0.7], [1.0], [0.6], [0.5], [0.4]]

    def once():
        car.inputs = [row[:] for row in template]
        nn.forward_propagation(car)

    return timed(once, 2000)


def _sample_poses(grid, occ, track, count, seed=1234):
    """Sensor origins drawn from open track, in world coordinates."""
    import random

    rng = random.Random(seed)
    width, height = occ.grid_shape(grid)
    poses = []
    while len(poses) < count:
        gx = rng.uniform(0, width)
        gy = rng.uniform(0, height)
        if not (grid[int(gx) + 1, int(gy) + 1] & occ.RAY_BIT):
            poses.append((gx + track.x, gy + track.y, rng.uniform(0, 360)))
    return poses


def bench_raycast(poses_per_regime=400):
    """The 5-px pygame march against the vectorised grid cast, per regime."""
    import numpy as np

    from nncar import entities as v
    from nncar.sim import occupancy as occ
    from nncar.sim import raycast as rc

    grid = occ.load_grid()
    track = v.Track(1, load_visuals=False, backend="mask")
    v.track = track
    batch = rc.RayBatch()
    poses = _sample_poses(grid, occ, track, poses_per_regime)

    def old(pose):
        world_x, world_y, heading = pose
        return [v.Sensor(world_x, world_y, heading + offset, length).distance()
                for offset, length in zip(batch.angles, batch.lengths)]

    def new(pose):
        world_x, world_y, heading = pose
        return batch.cast(grid, world_x - track.x, world_y - track.y, heading)

    # Agreement first: a speedup only counts if both answer the same question.
    agreed = total = 0
    oversized = 0
    for pose in poses:
        fast = new(pose)
        for value, reference in zip(fast, old(pose)):
            total += 1
            difference = abs(float(value) - reference)
            if difference < 1e-9:
                agreed += 1
            elif difference > rc.STEP + 1e-9:
                oversized += 1

    equivalence = {
        "rays_compared": total,
        "agree_within_1e-9": agreed,
        "agree_fraction": round(agreed / total, 5),
        "disagree_by_more_than_one_step": oversized,
        "note": ("residual disagreement is one 5-px step, where a sample lands "
                 "within a rounding error of a pixel boundary"),
    }
    if oversized:
        raise AssertionError("raycast implementations disagree by more than one step")

    regimes = {"short (<150px)": [], "medium (150-450px)": [], "open (>450px)": []}
    for pose in poses:
        mean = float(np.mean(new(pose)))
        name = ("short (<150px)" if mean < 150
                else "medium (150-450px)" if mean < 450 else "open (>450px)")
        regimes[name].append(pose)

    results = {"equivalence": equivalence, "regimes": {}}
    for name, group in regimes.items():
        if len(group) < 8:
            results["regimes"][name] = {"skipped": "only %d poses sampled" % len(group)}
            continue
        counter = {"i": 0}

        def step_old():
            counter["i"] += 1
            old(group[counter["i"] % len(group)])

        def step_new():
            counter["i"] += 1
            new(group[counter["i"] % len(group)])

        iterations = min(len(group), 150)
        before = timed(step_old, iterations)
        after = timed(step_new, iterations)
        results["regimes"][name] = {
            "poses": len(group),
            "march_5px": before,
            "vectorised": after,
            "speedup": round(before["us_min"] / after["us_min"], 2),
        }
    return results


def bench_rollout(episodes=12):
    """End-to-end: how many simulated ticks per second, single process."""
    import random

    from nncar import neural_network as nn
    from nncar.sim.rollout import RolloutConfig, build_track, simulate

    cfg = RolloutConfig(normalise_inputs=True)
    track = build_track(cfg)

    networks = []
    for seed in range(episodes):
        random.seed(seed)
        networks.append(nn.Network())

    def run():
        ticks = 0
        for index, network in enumerate(networks):
            result = simulate(network, start_index=index % 5, cfg=cfg,
                              seed=index, individual_id=index, track=track)
            ticks += result.ticks
        return ticks

    ticks = run()
    samples = []
    gc.disable()
    try:
        for _ in range(3):
            start = time.perf_counter()
            run()
            samples.append(time.perf_counter() - start)
    finally:
        gc.enable()

    best = min(samples)
    return {
        "episodes": episodes,
        "ticks_simulated": ticks,
        "seconds_min": round(best, 4),
        "ticks_per_second": round(ticks / best),
        "us_per_tick": round(best / ticks * 1e6, 2),
        "mean_ticks_per_episode": round(ticks / episodes, 1),
    }


def environment():
    import numpy

    info = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    try:
        import pygame

        info["pygame"] = pygame.version.ver
    except ImportError:
        pass
    return info


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/optimised.json")
    parser.add_argument("--label", default="optimised")
    parser.add_argument("--only", choices=["forward", "raycast", "rollout"],
                        help="run a single benchmark")
    args = parser.parse_args()

    report = {
        "label": args.label,
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": ("perf_counter, gc disabled, one warm-up, minimum of %d repeats; "
                   "equivalence asserted before timing" % REPEATS),
        "environment": environment(),
        "results": {},
    }

    if args.only in (None, "forward"):
        print("forward pass ...", flush=True)
        report["results"]["forward_pass"] = bench_forward()
        print("  %.1f us" % report["results"]["forward_pass"]["us_min"])

    if args.only in (None, "raycast"):
        print("raycast, 5 rays per car ...", flush=True)
        data = bench_raycast()
        report["results"]["raycast_5_rays"] = data
        eq = data["equivalence"]
        print("  equivalence: %d/%d rays agree (%.3f%%), none off by more than one step"
              % (eq["agree_within_1e-9"], eq["rays_compared"], 100 * eq["agree_fraction"]))
        print("  %-20s %12s %12s %9s" % ("regime", "5-px march", "vectorised", "speedup"))
        for name, entry in data["regimes"].items():
            if "skipped" in entry:
                print("  %-20s %s" % (name, entry["skipped"]))
            else:
                print("  %-20s %9.1f us %9.1f us %8.1fx"
                      % (name, entry["march_5px"]["us_min"],
                         entry["vectorised"]["us_min"], entry["speedup"]))

    if args.only in (None, "rollout"):
        print("full rollout ...", flush=True)
        data = bench_rollout()
        report["results"]["rollout"] = data
        print("  %d ticks/sec (%.1f us/tick), mean episode %.0f ticks"
              % (data["ticks_per_second"], data["us_per_tick"],
                 data["mean_ticks_per_episode"]))

    out = os.path.join(_bootstrap.ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
