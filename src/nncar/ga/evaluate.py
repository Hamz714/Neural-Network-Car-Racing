"""Scoring a population, across every core.

Evaluations are independent, so this is close to ideal parallel work - but
getting it right on Windows takes some care.

**SDL must be told to be headless before pygame is imported.** Windows spawns
fresh processes rather than forking, and a spawned child imports the module
holding the task function *before* it runs the pool initialiser. Setting the
environment in the initialiser would therefore be too late, so it is set in the
parent and inherited.

**The occupancy grid must never travel in a task payload.** It is 11 MB. Sent
with each of a hundred tasks per generation it would dominate everything else,
so it is loaded once per worker into a module global. `test_ga.py` asserts the
payload stays under 64 KB, because this is the sort of regression that shows up
as "training got slow" and nothing else.

**Seeds are derived, not counted out.** Each seed is a pure function of the run
seed, the generation, the individual and the start position, so it does not
depend on how work was scheduled, how many workers there were, or what order
results came back in.
"""

import os
import zlib

# Must precede any pygame import in a spawned worker.
from nncar.sim import headless  # noqa: F401  (imported for its side effect)

from nncar.sim.rollout import RolloutConfig, simulate

#: Per-worker state. Populated by init_worker, never pickled.
_WORKER = {}


def derive_seed(base_seed, generation, individual, start_index):
    """A reproducible seed for one rollout.

    crc32 over the identifying tuple: cheap, and it scatters neighbouring
    inputs to unrelated seeds, so individual 3 and individual 4 do not get
    correlated noise.
    """
    key = "%d:%d:%d:%d" % (base_seed, generation, individual, start_index)
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF


def init_worker(config_dict):
    """Load the track once per worker process."""
    headless.enable()

    from nncar import entities as v
    from nncar.sim import occupancy

    config = RolloutConfig(**config_dict)

    # allow_build=False: eight processes each decoding 133 MB of PNGs at once
    # is not a cache miss worth tolerating. The parent builds it first.
    grid = occupancy.load_grid(allow_build=False)

    track = v.Track(config.laps, load_visuals=False)
    track.grid = grid

    _WORKER["config"] = config
    _WORKER["track"] = track


def _task(payload):
    """Run one rollout. Must stay small and picklable."""
    individual_id, start_index, seed, network = payload
    result = simulate(network,
                      start_index=start_index,
                      cfg=_WORKER["config"],
                      seed=seed,
                      individual_id=individual_id,
                      track=_WORKER["track"])
    return individual_id, start_index, result


class Evaluator:
    """Scores populations, in parallel or in this process.

    workers=1 runs inline with no pool at all, which keeps profiling and
    debugging straightforward and makes the single-worker path a genuine
    reference for the parallel one rather than a different code path.
    """

    def __init__(self, config, workers=1, start_indices=(0, 1)):
        self.config = config
        self.workers = max(1, int(workers))
        self.start_indices = tuple(start_indices)
        self._pool = None

        if self.workers > 1:
            import multiprocessing as mp

            # Explicit spawn: Windows' default, and stating it keeps Linux
            # behaving identically rather than forking and inheriting state.
            context = mp.get_context("spawn")
            self._pool = context.Pool(
                processes=self.workers,
                initializer=init_worker,
                initargs=(config.as_dict(),),
            )
        else:
            init_worker(config.as_dict())

    def payloads(self, networks, generation, base_seed):
        for individual_id, network in enumerate(networks):
            for start_index in self.start_indices:
                yield (individual_id, start_index,
                       derive_seed(base_seed, generation, individual_id, start_index),
                       network)

    def evaluate(self, networks, generation, base_seed):
        """Return a list of RolloutResults per individual, in population order."""
        results = [[] for _ in networks]
        tasks = list(self.payloads(networks, generation, base_seed))

        if self._pool is None:
            for payload in tasks:
                individual_id, _, result = _task(payload)
                results[individual_id].append(result)
        else:
            # chunksize=1: rollout lengths vary more than tenfold, since a
            # crasher stops in a few dozen ticks and a lap-completer runs for
            # thousands. Static chunking would leave workers idle.
            for individual_id, _, result in self._pool.imap_unordered(_task, tasks,
                                                                     chunksize=1):
                results[individual_id].append(result)

        # imap_unordered returns out of order; sort each individual's rollouts
        # by start position so downstream aggregation is deterministic.
        for group in results:
            group.sort(key=lambda result: result.start_index)
        return results

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def resolve_workers(requested):
    """Clamp a worker count to what the machine has.

    Note that cpu_count - 1 sometimes beats cpu_count: the parent process is
    doing the mutation and logging, and competes for a core.
    """
    available = os.cpu_count() or 1
    if requested in (None, 0):
        return max(1, available - 1)
    return max(1, min(int(requested), available))
