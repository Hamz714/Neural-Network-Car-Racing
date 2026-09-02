"""A bitmap of where the track's walls are.

The three border images are 3608x3081 RGBA surfaces - about 133 MB once
decoded. Every sensor reading and every collision test used to interrogate them
through pygame, one pixel at a time. Collapsing them into a single array of
flags costs a couple of seconds once, then answers the same questions with an
array lookup, and lets the training workers avoid decoding the PNGs at all.

**Two thresholds, one grid.** The sensors and the collision test do not agree on
what counts as solid: ``Track.get_pixel_alpha`` treats any non-zero alpha as a
wall, while ``pygame.mask.from_surface`` defaults to alpha > 127. On the
anti-aliased edge of a border those differ by a pixel or two. A single plane
could only reproduce one of them, so the grid carries both as separate bits and
each caller asks for the one it has always used.

Indexing is [x][y], matching ``Surface.get_at((x, y))`` and
``surfarray.array_alpha`` - not the [row][column] convention numpy usually
implies. The array is padded by one pixel on every side and reads are clipped
into the padding, so an out-of-bounds sample returns "no wall" without needing a
bounds test.
"""

import hashlib
import os

import numpy as np

from nncar import assets

#: Alpha != 0. What the sensors have always treated as a wall.
RAY_BIT = 1
#: Alpha > 127. What pygame.mask.from_surface treats as solid.
MASK_BIT = 2

MASK_THRESHOLD = 127

#: Bumped when the grid's meaning changes, to invalidate stale caches.
FORMAT_VERSION = 1

_PAD = 1


def source_hash(paths):
    """A digest of the border images, so a stale cache is never silently used."""
    digest = hashlib.sha1()
    digest.update(("v%d" % FORMAT_VERSION).encode())
    for path in paths:
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def build_grid(paths=None):
    """Decode the border images into one padded uint8 grid of flags.

    Uses ``array_alpha``, which copies, rather than ``pixels_alpha``, which
    locks the surface - so each border can be released as soon as it has been
    folded in and peak memory stays near one surface rather than three.
    """
    import pygame

    paths = list(paths or assets.border_paths())

    grid = None
    for path in paths:
        surface = pygame.image.load(path)
        alpha = pygame.surfarray.array_alpha(surface)
        if grid is None:
            width, height = alpha.shape
            grid = np.zeros((width + 2 * _PAD, height + 2 * _PAD), dtype=np.uint8)
        elif alpha.shape != (grid.shape[0] - 2 * _PAD, grid.shape[1] - 2 * _PAD):
            raise ValueError("border %s is %s, expected %s"
                             % (path, alpha.shape,
                                (grid.shape[0] - 2 * _PAD, grid.shape[1] - 2 * _PAD)))

        inner = grid[_PAD:-_PAD, _PAD:-_PAD]
        inner |= (alpha != 0).astype(np.uint8) * RAY_BIT
        inner |= (alpha > MASK_THRESHOLD).astype(np.uint8) * MASK_BIT

        del alpha, surface

    return grid


def cache_path():
    return os.path.join(assets.CACHE_DIR, "track_occupancy.npz")


def save_grid(grid, paths=None, path=None):
    paths = list(paths or assets.border_paths())
    path = path or cache_path()
    assets.ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, grid=grid, source=source_hash(paths),
                        version=FORMAT_VERSION)
    return path


def load_grid(rebuild=False, paths=None, allow_build=True):
    """Return the occupancy grid, building and caching it if necessary.

    Loading the cache takes about 30 ms against roughly two seconds to decode
    the PNGs, which matters because every training worker does this at startup.
    Pass allow_build=False in a worker to make a cache miss an error rather than
    letting eight processes each decode 133 MB of images at once.
    """
    paths = list(paths or assets.border_paths())
    path = cache_path()

    if not rebuild and os.path.exists(path):
        with np.load(path, allow_pickle=False) as data:
            if (int(data["version"]) == FORMAT_VERSION
                    and str(data["source"]) == source_hash(paths)):
                return data["grid"]

    if not allow_build:
        raise RuntimeError(
            "no valid occupancy cache at %s; build it in the parent process "
            "before starting workers" % path)

    grid = build_grid(paths)
    save_grid(grid, paths, path)
    return grid


def grid_shape(grid):
    """The track's dimensions, excluding the one-pixel pad."""
    return grid.shape[0] - 2 * _PAD, grid.shape[1] - 2 * _PAD


def car_silhouette(surface, threshold=MASK_THRESHOLD):
    """A car's solid pixels as a boolean array, indexed [x][y] to match the grid."""
    import pygame

    return pygame.surfarray.array_alpha(surface) > threshold
