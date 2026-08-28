"""Scoring a single drive.

Designing this is the hard part of neuroevolution: the search optimises exactly
what you write down, including the parts you did not mean. Two failure modes
drove the shape below, and both are pinned by tests.

**Standing still must never be optimal.** If crashing costs more than making
progress earns, the highest-scoring behaviour is to sit on the start line
forever - zero progress, zero collisions, zero penalty. The invariant
``collision * collision_limit < progress`` is what rules that out, and
``check_weights`` asserts it.

**Laps must be earned.** The lap counter increments whenever a car crosses the
finish line northbound, and every spawn point sits north of that line, so a car
that reverses a little over a hundred pixels and drives forward again collects a
lap in about thirty ticks having passed no checkpoints at all. Rewarding
``laps`` directly would make that the whole game. Progress is therefore measured
in *checkpoints*, and a lap only counts once it has cleared most of them; the
rollout records the gate count at each lap boundary so this stays checkable
after the fact. (See ``RolloutConfig.min_gates_per_lap`` for why the threshold
is eight of ten and not all ten - the track is a double loop, and two gates sit
on a section the outer circuit does not touch.)

**Progress is the best single circuit, never the total.** An earlier version
summed checkpoints across every circuit a car attempted, which sounds like the
same thing and is not: it paid a car for going round twice badly rather than
once well. A driver that clears eight checkpoints on each of two laps banked
sixteen, beating one that cleared ten in a single clean lap, so the search had
every reason to keep circling instead of improving. Scoring the best single
circuit removes it - eight is worth less than ten however many times it is
repeated, and lapping again only costs time.
"""


class FitnessWeights:
    """Weights for the scoring terms. Immutable by convention."""

    __slots__ = ("progress", "speed", "lap", "collision", "finish")

    def __init__(self, progress=100.0, speed=25.0, lap=250.0, collision=5.0, finish=500.0):
        self.progress = progress
        self.speed = speed
        self.lap = lap
        self.collision = collision
        self.finish = finish

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    def __repr__(self):
        return "FitnessWeights(%s)" % ", ".join(
            "%s=%g" % (k, v) for k, v in self.as_dict().items())


DEFAULT_WEIGHTS = FitnessWeights()


def check_weights(weights, collision_limit):
    """Raise if the weights make giving up more attractive than driving.

    A car is retired after ``collision_limit`` collisions, so the worst penalty
    it can accumulate is ``collision * collision_limit``. If that exceeds what a
    single checkpoint pays, the search learns not to move.
    """
    worst_penalty = weights.collision * collision_limit
    if worst_penalty >= weights.progress:
        raise ValueError(
            "collision penalty %g x limit %d = %g would exceed the %g paid per "
            "checkpoint, making it optimal to stand still"
            % (weights.collision, collision_limit, worst_penalty, weights.progress))


def fitness(result, weights=DEFAULT_WEIGHTS, fps=50, max_ticks=3000):
    """Score one rollout. Higher is better; negative scores are fine.

    Selection is by truncation rather than roulette, so there is no need to
    keep scores positive and no need for the usual rescaling hack.
    """
    gates = result.best_circuit_gates
    seconds = max(result.ticks, 1) / float(fps)

    score = weights.progress * gates

    # Speed enters as a rate, not as a time penalty. Subtracting elapsed ticks
    # would punish a car for getting further before it stopped, which is the
    # opposite of the intended pressure; a ratio only rewards covering the same
    # ground faster.
    score += weights.speed * (gates / seconds)

    score += weights.lap * result.valid_laps
    score -= weights.collision * result.collisions

    if result.terminated == "finished":
        score += weights.finish * (1.0 - result.ticks / float(max_ticks))

    return score
