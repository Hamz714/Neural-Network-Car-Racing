"""Drive one network around the track and report what happened.

The rollout runs the same Car/NPC code the interactive game runs - there is no
second copy of the physics that could quietly drift out of step with it. What
differs is only the surroundings: no window, no player, no other traffic, and a
clock advanced by the loop rather than read off the wall.

Early termination is what makes training practical. An untrained network
typically spins, reverses into a wall, or drives into open grass, and there is
nothing to learn from watching it do so for the full sixty seconds. Two
independent rules retire a car:

* **no checkpoint progress for `stall_ticks`** - a generous bound, because the
  widest gap between gates is around two thousand pixels and a timid driver
  crossing it at four pixels a tick legitimately needs five hundred ticks;
* **barely moved over `displacement_window` ticks** - a tight bound that catches
  spinning, wall-hugging and wedged cars immediately, without penalising a car
  that is slow but genuinely getting somewhere.

Either alone would be wrong. Together they cut a generation-zero rollout from
three thousand ticks to a couple of hundred.

`min_gates_per_lap` is 8 of the 10 checkpoints rather than all of them, and the
reason is a property of the track rather than a tolerance for sloppiness. The
circuit is a double loop: two of the checkpoints sit on an inner section, and
the other eight lie on the outer ring. Driving the outer ring is a complete,
closed lap that returns across the finish line - it is simply not the route the
checkpoint list describes. Requiring all ten would mean requiring the inner
detour, which is a harder navigation problem than "drive round the track" and
not the one being posed here. A network that does find the inner section scores
higher for it, because ten checkpoints beat eight.
"""

import math
from collections import deque


class RolloutConfig:
    """Everything that defines an evaluation episode."""

    __slots__ = ("fps", "laps", "max_ticks", "stall_ticks", "collision_limit",
                 "displacement_window", "displacement_min", "min_gates_per_lap",
                 "exploration_noise", "normalise_inputs", "difficulty_image")

    def __init__(self, fps=50, laps=1, max_ticks=3000, stall_ticks=400,
                 collision_limit=5, displacement_window=100, displacement_min=50.0,
                 min_gates_per_lap=8, exploration_noise=0.0, normalise_inputs=True,
                 difficulty_image="red.png"):
        self.fps = fps
        self.laps = laps
        self.max_ticks = max_ticks
        self.stall_ticks = stall_ticks
        self.collision_limit = collision_limit
        self.displacement_window = displacement_window
        self.displacement_min = displacement_min
        self.min_gates_per_lap = min_gates_per_lap
        self.exploration_noise = exploration_noise
        self.normalise_inputs = normalise_inputs
        self.difficulty_image = difficulty_image

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    def __repr__(self):
        return "RolloutConfig(%s)" % ", ".join(
            "%s=%r" % (k, v) for k, v in self.as_dict().items())


class RolloutResult:
    """The outcome of one episode. Small, picklable, and free of pygame objects."""

    __slots__ = ("individual_id", "start_index", "seed", "ticks", "total_gates",
                 "best_circuit_gates", "valid_laps", "lap_gates", "lap_ticks",
                 "collisions", "distance_travelled", "terminated")

    def __init__(self, individual_id=0, start_index=0, seed=0, ticks=0, total_gates=0,
                 best_circuit_gates=0, valid_laps=0, lap_gates=(), lap_ticks=(),
                 collisions=0, distance_travelled=0.0, terminated="timeout"):
        self.individual_id = individual_id
        self.start_index = start_index
        self.seed = seed
        self.ticks = ticks
        #: Checkpoints cleared across every circuit attempted. Descriptive only.
        self.total_gates = total_gates
        #: Checkpoints cleared on the single best circuit. This is what fitness
        #: scores - see nncar.sim.fitness.
        self.best_circuit_gates = best_circuit_gates
        self.valid_laps = valid_laps
        self.lap_gates = tuple(lap_gates)
        self.lap_ticks = tuple(lap_ticks)
        self.collisions = collisions
        self.distance_travelled = distance_travelled
        self.terminated = terminated

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}

    # __slots__ classes need these two to survive pickling to a worker.
    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def __repr__(self):
        return ("RolloutResult(best_circuit=%d, total_gates=%d, laps=%d, ticks=%d, "
                "collisions=%d, %s)"
                % (self.best_circuit_gates, self.total_gates, self.valid_laps,
                   self.ticks, self.collisions, self.terminated))


def build_track(cfg):
    """A track with no backdrop - the trainer never draws anything."""
    from nncar import entities as v

    return v.Track(cfg.laps, load_visuals=False)


def make_car(network, start_position, cfg, rng=None):
    from nncar import entities as v

    car = v.NPC(v.CARS[0][0], 0, network,
                start_position=start_position,
                rng=rng,
                exploration_noise=cfg.exploration_noise)
    car.normalise_inputs = cfg.normalise_inputs
    return car


def simulate(network, start_index=0, cfg=None, seed=0, individual_id=0, track=None,
             on_tick=None):
    """Run one episode and return a RolloutResult.

    `track` may be supplied to reuse a loaded track across many rollouts, which
    is what the parallel evaluator does. `on_tick` is called with the car after
    every step, and exists so the same function can drive an on-screen replay.
    """
    import random

    from nncar import entities as v
    from nncar.sim import clock as sim_clock

    cfg = cfg or RolloutConfig()

    clock = sim_clock.TickClock(cfg.fps)
    previous_clock = sim_clock.set_clock(clock)

    if track is None:
        track = build_track(cfg)
    previous_track = getattr(v, "track", None)
    v.track = track

    try:
        rng = random.Random(seed)
        start = v.NPC_START_POS[start_index % len(v.NPC_START_POS)]
        car = make_car(network, start, cfg, rng)

        lap_gates = []
        lap_ticks = []
        collisions = 0
        previous_collisions = 0
        best_gates = 0
        best_circuit = 0
        last_progress_tick = 0
        distance = 0.0

        # Sampled sparsely: only the endpoints of the window matter.
        stride = max(1, cfg.displacement_window // 4)
        history = deque(maxlen=cfg.displacement_window // stride + 1)

        previous_x, previous_y = car.world_x, car.world_y
        terminated = "timeout"
        tick = 0

        for tick in range(cfg.max_ticks):
            car.update_sensors()
            car.move()

            # A lap resets checkpoints_passed, so the count has to be banked
            # before reset_checkpoints runs, not after.
            gates_this_lap = car.checkpoints_passed
            laps_before = car.laps
            car.reset_checkpoints()
            if car.laps > laps_before:
                lap_gates.append(gates_this_lap)
                lap_ticks.append(tick)
            car.check_checkpoints()

            # collisions is also zeroed by a lap, so accumulate the increments.
            collisions += max(0, car.collisions - previous_collisions)
            previous_collisions = car.collisions

            step = math.hypot(car.world_x - previous_x, car.world_y - previous_y)
            distance += step
            previous_x, previous_y = car.world_x, car.world_y

            # Progress for the anti-stall rule is cumulative: a car that
            # crosses the line and starts a fresh circuit has not stalled.
            gates = sum(lap_gates) + car.checkpoints_passed
            if gates > best_gates:
                best_gates = gates
                last_progress_tick = tick

            # Progress for scoring is the best single circuit. Summing across
            # circuits would pay a car more for two sloppy laps than for one
            # clean one, which is exactly the wrong lesson.
            circuit = max(lap_gates) if lap_gates else 0
            if car.checkpoints_passed > circuit:
                circuit = car.checkpoints_passed
            if circuit > best_circuit:
                best_circuit = circuit

            valid_laps = sum(1 for count in lap_gates if count >= cfg.min_gates_per_lap)

            if on_tick is not None:
                on_tick(car, tick)

            clock.advance()

            if valid_laps >= cfg.laps:
                terminated = "finished"
                break
            if collisions >= cfg.collision_limit:
                terminated = "crash"
                break
            if tick - last_progress_tick > cfg.stall_ticks:
                terminated = "stall"
                break
            if tick % stride == 0:
                history.append((car.world_x, car.world_y))
                if len(history) == history.maxlen:
                    old_x, old_y = history[0]
                    if math.hypot(car.world_x - old_x, car.world_y - old_y) < cfg.displacement_min:
                        terminated = "stall"
                        break

        return RolloutResult(
            individual_id=individual_id,
            start_index=start_index,
            seed=seed,
            ticks=tick + 1,
            total_gates=sum(lap_gates) + car.checkpoints_passed,
            best_circuit_gates=best_circuit,
            valid_laps=sum(1 for count in lap_gates if count >= cfg.min_gates_per_lap),
            lap_gates=lap_gates,
            lap_ticks=lap_ticks,
            collisions=collisions,
            distance_travelled=distance,
            terminated=terminated,
        )
    finally:
        sim_clock.set_clock(previous_clock)
        if previous_track is not None:
            v.track = previous_track
