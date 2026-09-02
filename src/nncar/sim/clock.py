"""Time, counted in simulation ticks rather than read off the wall clock.

The original timers called time.time() directly, which ties behaviour to how
fast the machine happens to be running. That is wrong in two directions: a
training run that simulates faster than real time would see every timer fire
immediately, and the game's own drift timer would mean different things at
different frame rates.

Counting ticks instead makes the simulation deterministic and frame-rate
independent. The game advances the clock once per frame at its fixed 50 fps, so
a 1.5 second timer becomes exactly 75 ticks - the same duration it always was,
now guaranteed rather than assumed.
"""

import time


class TickClock:
    """A clock driven by an explicit tick counter."""

    def __init__(self, fps=50):
        self.fps = fps
        self.tick = 0

    def now(self):
        return self.tick / self.fps

    def advance(self, ticks=1):
        self.tick += ticks

    def reset(self):
        self.tick = 0


class RealClock:
    """Wall-clock time, as the original code used."""

    def __init__(self, fps=50):
        self.fps = fps

    def now(self):
        return time.time()

    def advance(self, ticks=1):
        pass

    def reset(self):
        pass


#: The clock every Event consults unless given one of its own.
ambient = TickClock()


def set_clock(clock):
    """Install the process-wide clock and return the one it replaced."""
    global ambient
    previous = ambient
    ambient = clock
    return previous


def now():
    return ambient.now()


def advance(ticks=1):
    ambient.advance(ticks)


def reset():
    ambient.reset()
