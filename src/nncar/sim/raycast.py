"""Sensor rays, cast against the occupancy grid.

The original sensor walked outwards five pixels at a time, asking pygame for a
pixel each step, and stopped when it found a wall - up to 140 Python iterations
and 420 surface reads per ray, five rays per car, per frame. The cost therefore
scaled with how far the ray travelled, which is exactly backwards: the
expensive case is a ray that sees nothing, and untrained networks drive into
open space constantly.

Sampling every point along a ray at once and letting numpy find the first hit
makes the cost flat instead. That flatness, rather than the raw factor, is what
makes training viable.

Two details decide whether this agrees with the original:

* **The sample set runs one step past the ray's length.** The old loop tested
  the pixel *before* testing whether it had gone too far, so a wall sitting
  exactly at ``length + 5`` was still reported. Masking at ``t <= length``
  instead of ``t <= length + step`` disagrees on roughly one ray in five
  hundred.
* **``argmax`` returns 0 when nothing matches.** Taken at face value, every ray
  that sees no wall would report a distance of zero - every car permanently
  convinced it is about to crash, with nothing to indicate anything is wrong.
  The hit has to be confirmed separately.
"""

import numpy as np

from nncar.sim.occupancy import RAY_BIT

#: Pixels advanced per sample. Matches the original sensor's stride.
STEP = 5.0

_PAD = 1


def march_reference(grid, origin_x, origin_y, angle_degrees, length, step=STEP):
    """The original algorithm, kept as the oracle for the vectorised one.

    Deliberately a transcription rather than an improvement: its job is to be
    obviously equivalent to the code this replaced, so the equivalence test
    means something.
    """
    import math

    radians = math.radians(angle_degrees)
    dx = -math.sin(radians) * step
    dy = -math.cos(radians) * step

    x, y = origin_x, origin_y
    travelled = 0.0
    while True:
        xi, yi = int(x) + _PAD, int(y) + _PAD
        if 0 <= xi < grid.shape[0] and 0 <= yi < grid.shape[1]:
            if grid[xi, yi] & RAY_BIT:
                return travelled
        if travelled > length:
            return length
        x += dx
        y += dy
        travelled += step


class RayBatch:
    """All of one car's sensors, cast together.

    The sample offsets are precomputed once and shared by every car, so a cast
    allocates only the small per-call arrays.
    """

    def __init__(self, angles=(-90, -45, 0, 45, 90), lengths=(500, 600, 700, 600, 500),
                 step=STEP):
        if len(angles) != len(lengths):
            raise ValueError("need one length per ray")

        self.angles = np.asarray(angles, dtype=np.float64)
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.step = float(step)
        self.count = len(angles)

        samples = int(max(lengths) / step) + 2
        self.t = np.arange(samples, dtype=np.float64) * step
        self.t_grid = np.broadcast_to(self.t, (self.count, samples))

        # One step past the limit, matching the original loop's ordering.
        self.in_range = self.t_grid <= (self.lengths[:, None] + self.step)

        self._radians = np.empty(self.count, dtype=np.float64)

    def cast(self, grid, origin_x, origin_y, car_angle):
        """Distances to the first wall along each ray, in grid coordinates."""
        np.radians(self.angles + car_angle, out=self._radians)

        # The car's heading convention: x decreases with sin, y with cos.
        xs = origin_x - np.sin(self._radians)[:, None] * self.t_grid
        ys = origin_y - np.cos(self._radians)[:, None] * self.t_grid

        # int() truncates toward zero; np.floor would not, and would disagree
        # with the original on negative coordinates.
        xi = xs.astype(np.int32) + _PAD
        yi = ys.astype(np.int32) + _PAD

        # The grid is padded, so clipping into the pad reads a guaranteed zero
        # and stands in for an explicit bounds test.
        np.clip(xi, 0, grid.shape[0] - 1, out=xi)
        np.clip(yi, 0, grid.shape[1] - 1, out=yi)

        hit = (grid[xi, yi] & RAY_BIT).astype(bool) & self.in_range

        first = hit.argmax(axis=1)
        found = hit[np.arange(self.count), first]

        return np.where(found, self.t[first], self.lengths)
