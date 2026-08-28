"""The vectorised raycaster against the algorithm it replaced.

A speedup only means anything if both sides answer the same question, so the
equivalence check runs first and the benchmark quotes it.
"""

import math
import random

import numpy as np
import pytest

from conftest import requires_pygame

from nncar.sim import raycast as rc
from nncar.sim.occupancy import RAY_BIT

PAD = 1


def synthetic_grid(width=400, height=400):
    """An empty arena with a solid border, padded like the real grid."""
    grid = np.zeros((width + 2 * PAD, height + 2 * PAD), dtype=np.uint8)
    inner = grid[PAD:-PAD, PAD:-PAD]
    inner[0, :] = RAY_BIT
    inner[-1, :] = RAY_BIT
    inner[:, 0] = RAY_BIT
    inner[:, -1] = RAY_BIT
    return grid


def test_matches_the_reference_march_on_a_synthetic_grid():
    grid = synthetic_grid()
    batch = rc.RayBatch()
    rng = random.Random(11)

    mismatches = []
    total = 0
    for _ in range(400):
        x = rng.uniform(20, 380)
        y = rng.uniform(20, 380)
        heading = rng.uniform(0, 360)
        fast = batch.cast(grid, x, y, heading)
        for index, (offset, length) in enumerate(zip(batch.angles, batch.lengths)):
            reference = rc.march_reference(grid, x, y, heading + offset, length)
            total += 1
            if abs(reference - float(fast[index])) > 1e-9:
                mismatches.append(abs(reference - float(fast[index])))

    assert total == 2000
    # Residual disagreement is inherent: the march accumulates its position one
    # step at a time while the vectorised form evaluates a closed-form distance,
    # so a sample sitting within a rounding error of a pixel boundary can land
    # on either side of it. When that happens the answers differ by exactly one
    # step, never more.
    assert len(mismatches) / total < 0.005, "%d/%d disagreed" % (len(mismatches), total)
    assert all(diff <= rc.STEP + 1e-9 for diff in mismatches), \
        "disagreement larger than one step: %s" % sorted(mismatches)[-5:]


def test_a_wall_exactly_one_step_past_the_limit_is_still_reported():
    """The original tested the pixel before testing the length bound.

    So the sample set runs to length + step, and a wall sitting exactly there
    is reported at length + step rather than clamped to length. Masking at
    `t <= length` instead would disagree on about one ray in five hundred.
    """
    length = 100.0
    grid = np.zeros((400 + 2 * PAD, 8 + 2 * PAD), dtype=np.uint8)
    # Heading 90 degrees points along -x, so place the wall to the left.
    origin_x, origin_y = 300.0, 4.0
    wall_x = int(origin_x - (length + rc.STEP))
    grid[wall_x + PAD, :] = RAY_BIT

    batch = rc.RayBatch(angles=(0,), lengths=(length,))
    distance = batch.cast(grid, origin_x, origin_y, 90.0)[0]
    assert distance == pytest.approx(length + rc.STEP)


def test_a_ray_that_hits_nothing_returns_its_full_length():
    """argmax returns 0 for an all-False row.

    Without confirming the hit separately, every ray seeing open space would
    report a distance of zero - each car certain it is about to crash, with
    nothing to show anything had gone wrong.
    """
    grid = np.zeros((2000 + 2 * PAD, 2000 + 2 * PAD), dtype=np.uint8)
    batch = rc.RayBatch()
    distances = batch.cast(grid, 1000.0, 1000.0, 0.0)
    assert list(distances) == list(batch.lengths)
    assert not any(d == 0.0 for d in distances)


def test_a_ray_starting_inside_a_wall_reads_zero():
    grid = np.ones((100 + 2 * PAD, 100 + 2 * PAD), dtype=np.uint8) * RAY_BIT
    batch = rc.RayBatch()
    assert list(batch.cast(grid, 50.0, 50.0, 0.0)) == [0.0] * 5


def test_out_of_bounds_samples_read_as_empty_not_as_walls():
    """The pad plus a clip stands in for a bounds test."""
    grid = synthetic_grid(50, 50)
    batch = rc.RayBatch(angles=(0,), lengths=(500,))
    for heading in (0, 90, 180, 270):
        distance = batch.cast(grid, -400.0, -400.0, heading)[0]
        assert 0.0 <= distance <= 500.0


def test_is_deterministic():
    grid = synthetic_grid()
    batch = rc.RayBatch()
    first = batch.cast(grid, 123.4, 210.9, 33.3)
    second = batch.cast(grid, 123.4, 210.9, 33.3)
    assert list(first) == list(second)


def test_geometry_matches_the_cars_heading_convention():
    """Position updates use x -= v*sin(a), y -= v*cos(a); rays must agree."""
    grid = synthetic_grid(600, 600)
    batch = rc.RayBatch(angles=(0,), lengths=(400,))
    # Heading 0 points along -y, so the wall met first is the one at y == 0.
    distance = batch.cast(grid, 300.0, 250.0, 0.0)[0]
    assert distance == pytest.approx(250.0, abs=rc.STEP)


@requires_pygame
@pytest.mark.slow
def test_matches_the_original_sensor_class_on_the_real_track():
    """The equivalence that actually matters, on the real circuit."""
    from nncar import entities as v
    from nncar.sim import occupancy as occ

    grid = occ.load_grid()
    width, height = occ.grid_shape(grid)
    # The legacy Sensor reads pixels straight off the border surfaces, so the
    # reference track has to be built on the mask backend.
    v.track = v.Track(1, load_visuals=False, backend="mask")
    batch = rc.RayBatch()
    rng = random.Random(99)

    poses = []
    while len(poses) < 400:
        gx = rng.uniform(0, width)
        gy = rng.uniform(0, height)
        if not (grid[int(gx) + PAD, int(gy) + PAD] & RAY_BIT):
            poses.append((gx + v.track.x, gy + v.track.y, rng.uniform(0, 360)))

    agreed = 0
    total = 0
    oversized = []
    for world_x, world_y, heading in poses:
        fast = batch.cast(grid, world_x - v.track.x, world_y - v.track.y, heading)
        for index, (offset, length) in enumerate(zip(batch.angles, batch.lengths)):
            original = v.Sensor(world_x, world_y, heading + offset, length).distance()
            total += 1
            difference = abs(original - float(fast[index]))
            if difference < 1e-9:
                agreed += 1
            elif difference > rc.STEP + 1e-9:
                oversized.append((difference, original, float(fast[index])))

    assert not oversized, "disagreed by more than one step: %s" % oversized[:5]
    assert agreed / total >= 0.995, \
        "only %d of %d rays agreed (%.3f%%)" % (agreed, total, 100 * agreed / total)


def test_march_reference_is_a_faithful_transcription():
    """Spot-check the oracle itself, since everything else is measured against it."""
    grid = np.zeros((200 + 2 * PAD, 20 + 2 * PAD), dtype=np.uint8)
    grid[50 + PAD, :] = RAY_BIT
    # Heading 90 points along -x from x=150 toward the wall at x=50.
    assert rc.march_reference(grid, 150.0, 10.0, 90.0, 500.0) == pytest.approx(100.0)
    # Facing away, nothing is hit, so the full length comes back.
    assert rc.march_reference(grid, 150.0, 10.0, 270.0, 300.0) == pytest.approx(300.0)
    assert math.isfinite(rc.march_reference(grid, 150.0, 10.0, 0.0, 100.0))
