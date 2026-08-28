"""Car-against-wall tests, answered from the occupancy grid.

``pygame.mask.Mask.overlap`` returns the first overlapping pixel, but every
caller in this project only ever asks whether the result was None - so the
question is really "do these two bitmaps intersect at this offset", which is an
array slice and a bitwise and.

This is equivalent rather than approximate, provided both sides agree on what
counts as solid. ``pygame.mask.from_surface`` thresholds alpha at 127, so the
grid's MASK_BIT plane is built with the same threshold and the car silhouettes
are taken the same way. ``tests/test_collision_grid.py`` checks the two agree
on every one of several thousand poses across the real track.

Dropping the masks lets the game release the three decoded border surfaces -
around 133 MB - which it otherwise holds for the entire session.
"""

import numpy as np

from nncar.sim.occupancy import MASK_BIT

_PAD = 1


def overlaps(grid, silhouette, offset_x, offset_y):
    """True if any solid pixel of the car covers a solid pixel of the track.

    offset_x/offset_y are the car's top-left corner in grid coordinates, and
    may be negative or beyond the track - the overlap region is clipped, and an
    empty region simply means no collision.
    """
    car_w, car_h = silhouette.shape
    grid_w = grid.shape[0] - 2 * _PAD
    grid_h = grid.shape[1] - 2 * _PAD

    x0 = int(offset_x)
    y0 = int(offset_y)

    # Clip the car's rectangle against the track.
    gx0 = max(x0, 0)
    gy0 = max(y0, 0)
    gx1 = min(x0 + car_w, grid_w)
    gy1 = min(y0 + car_h, grid_h)
    if gx0 >= gx1 or gy0 >= gy1:
        return False

    window = grid[gx0 + _PAD:gx1 + _PAD, gy0 + _PAD:gy1 + _PAD]
    car = silhouette[gx0 - x0:gx1 - x0, gy0 - y0:gy1 - y0]

    return bool(np.any((window & MASK_BIT).astype(bool) & car))
