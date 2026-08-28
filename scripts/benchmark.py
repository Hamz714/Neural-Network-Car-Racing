"""Measure the simulation's hot paths and write the results to JSON.

Methodology, applied to every measurement here:
  * time.perf_counter, with the garbage collector disabled during timing
  * one discarded warm-up pass
  * REPEATS independent repeats, reporting the minimum
    (minimum rejects OS scheduling noise; median and stdev are recorded too)
  * no display: SDL runs on the dummy driver

Raycasting is reported per regime rather than as one aggregate, because its
cost depends entirely on how far the ray travels before it hits something -
a single number would hide the fact that the win is concentrated in exactly
the open-space case that untrained networks spend all their time in.

    python scripts/benchmark.py --out docs/baseline.json --label baseline
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
    """Return per-iteration microseconds: min, median, stdev."""
    fn()  # warm-up
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


def _on_track_poses(count=240):
    """Sample sensor origins from the interiors of the ten checkpoint gates.

    Checkpoint coordinates are known-good on-track positions, so this samples
    where cars actually drive rather than uniformly over a mostly-empty map.
    """
    import random

    rng = random.Random(1234)
    poses = []
    checkpoints = [
        (450, -355, 1030, -355), (1810, -690, 1810, -120), (1765, 375, 2360, 375),
        (3395, 400, 3395, 960), (2840, -540, 2840, -265), (3370, -100, 3930, -100),
        (3365, 1440, 3935, 1440), (2325, 1455, 2325, 2005), (1050, 1205, 1050, 1755),
        (465, 665, 1025, 665),
    ]
    while len(poses) < count:
        x1, y1, x2, y2 = checkpoints[len(poses) % len(checkpoints)]
        t = rng.uniform(0.25, 0.75)
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        poses.append((x, y, rng.uniform(0, 360)))
    return poses


def bench_raycast_march():
    """The original 5-px march, bucketed by how far the rays got."""
    from nncar import entities as v

    v.track = v.Track(1)
    poses = _on_track_poses()
    lengths = (500, 600, 700, 600, 500)
    offsets = (-90, -45, 0, 45, 90)

    def cast(x, y, angle):
        total = 0.0
        for offset, length in zip(offsets, lengths):
            sensor = v.Sensor(x, y, angle + offset, length)
            total += sensor.distance()
        return total

    # Classify each pose by its mean ray length, then time the buckets separately.
    buckets = {"short": [], "medium": [], "open": []}
    for x, y, angle in poses:
        mean = cast(x, y, angle) / len(offsets)
        name = "short" if mean < 150 else ("medium" if mean < 450 else "open")
        buckets[name].append((x, y, angle))

    results = {}
    for name, group in buckets.items():
        if len(group) < 8:
            results[name] = {"skipped": "only %d poses" % len(group)}
            continue
        index = {"i": 0}

        def once():
            x, y, angle = group[index["i"] % len(group)]
            index["i"] += 1
            cast(x, y, angle)

        results[name] = timed(once, min(len(group), 120))
        results[name]["poses"] = len(group)
    return results


def environment():
    import numpy

    info = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import pygame

        info["pygame"] = pygame.version.ver
    except ImportError:
        pass
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/baseline.json")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--skip-raycast", action="store_true",
                        help="skip the track-dependent benchmarks (fast smoke test)")
    args = parser.parse_args()

    report = {
        "label": args.label,
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "perf_counter, gc disabled, 1 warm-up, min of %d repeats" % REPEATS,
        "environment": environment(),
        "results": {},
    }

    print("forward pass ...", flush=True)
    report["results"]["forward_pass"] = bench_forward()
    print("  %.1f us" % report["results"]["forward_pass"]["us_min"])

    if not args.skip_raycast:
        print("raycast, 5 rays per car (loading track) ...", flush=True)
        report["results"]["raycast_5_rays"] = bench_raycast_march()
        for name, data in report["results"]["raycast_5_rays"].items():
            if "us_min" in data:
                print("  %-7s %8.1f us  (%d poses)" % (name, data["us_min"], data["poses"]))

    out = os.path.join(_bootstrap.ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
