"""The grid collision test against pygame's masks, on the real track.

Replacing pygame.mask.overlap is only safe if the two agree everywhere, so this
compares them across thousands of poses rather than spot-checking a few. The
subtle part is the threshold: masks treat alpha > 127 as solid while the
sensors treat any non-zero alpha as solid, and the grid has to carry both.
"""

import random

import numpy as np
import pytest

from conftest import requires_pygame

pytestmark = requires_pygame

POSES = 5000


@pytest.fixture(scope="module")
def setup():
    import pygame

    from nncar import entities as v
    from nncar.sim import occupancy as occ

    grid = occ.load_grid()
    mask_track = v.Track(1, load_visuals=False, backend="mask")
    image = pygame.transform.scale(v.CARS[0][0], (80, 140))
    return {
        "grid": grid,
        "shape": occ.grid_shape(grid),
        "masks": mask_track.mask,
        "mask": pygame.mask.from_surface(image),
        "silhouette": occ.car_silhouette(image),
    }


def test_silhouette_and_mask_describe_the_same_shape(setup):
    assert int(setup["silhouette"].sum()) == setup["mask"].count()
    assert setup["silhouette"].shape == setup["mask"].get_size()


def test_the_two_alpha_thresholds_genuinely_differ(setup):
    """If they did not, one bitplane would do and this whole design is overkill.

    They do: the anti-aliased fringe of the borders is non-zero but below 127,
    so a single-plane grid would have been quietly wrong for one of the two
    callers.
    """
    from nncar.sim import occupancy as occ

    grid = setup["grid"]
    ray_solid = int((grid & occ.RAY_BIT).astype(bool).sum())
    mask_solid = int((grid & occ.MASK_BIT).astype(bool).sum())
    assert ray_solid > mask_solid
    assert ray_solid - mask_solid > 1000, "expected a real anti-aliased fringe"


def test_grid_and_mask_agree_on_every_pose(setup):
    from nncar.sim.collision import overlaps

    grid = setup["grid"]
    width, height = setup["shape"]
    silhouette = setup["silhouette"]
    masks = setup["masks"]
    mask = setup["mask"]

    rng = random.Random(2024)
    disagreements = []
    solid_seen = 0

    for _ in range(POSES):
        x = rng.randint(-100, width + 20)
        y = rng.randint(-100, height + 20)

        by_grid = overlaps(grid, silhouette, x, y)
        by_mask = any(m.overlap(mask, (x, y)) is not None for m in masks)

        if by_mask:
            solid_seen += 1
        if by_grid != by_mask:
            disagreements.append((x, y, by_grid, by_mask))

    assert not disagreements, \
        "%d of %d poses disagreed, e.g. %s" % (len(disagreements), POSES, disagreements[:5])
    # Guard against the test passing because every pose happened to miss.
    assert solid_seen > POSES * 0.05, \
        "only %d of %d poses touched a wall; the sample is not exercising much" \
        % (solid_seen, POSES)


def test_offsets_outside_the_track_do_not_raise(setup):
    from nncar.sim.collision import overlaps

    grid = setup["grid"]
    width, height = setup["shape"]
    silhouette = setup["silhouette"]

    for x, y in [(-10_000, -10_000), (width + 5000, height + 5000),
                 (-80, -140), (width, height), (-1, -1)]:
        assert overlaps(grid, silhouette, x, y) is False


def test_a_car_fully_inside_a_wall_collides(setup):
    from nncar.sim import occupancy as occ
    from nncar.sim.collision import overlaps

    grid = setup["grid"]
    solid = np.argwhere((grid & occ.MASK_BIT).astype(bool))
    assert len(solid) > 0
    x, y = solid[len(solid) // 2]
    ones = np.ones((80, 140), dtype=bool)
    assert overlaps(grid, ones, int(x) - 40 - 1, int(y) - 70 - 1) is True


def test_both_backends_agree_through_the_track_api(setup):
    """The same check one level up, through Track.collides and a real car."""
    from copy import deepcopy

    from nncar import entities as v

    grid_track = v.Track(1, load_visuals=False, backend="grid")
    mask_track = v.Track(1, load_visuals=False, backend="mask")

    v.track = grid_track
    v.NPC.start_positions = deepcopy(v.NPC_START_POS)
    car = v.NPC(v.CARS[0][0], 0, None, start_position=[610, 490])

    rng = random.Random(5)
    for _ in range(600):
        car.x = rng.uniform(-500, 4000)
        car.y = rng.uniform(-1500, 2500)
        v.track = grid_track
        by_grid = car.collide()
        v.track = mask_track
        by_mask = car.collide()
        assert bool(by_grid) == bool(by_mask), \
            "backends disagree at (%.1f, %.1f)" % (car.x, car.y)
