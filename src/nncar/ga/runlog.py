"""Recording a training run.

A number is only quotable if someone else could get it again, so a run writes
down enough to be repeated: every argument, the seed, the commit it ran from
and whether the tree was clean, and the library versions. The per-generation
CSV is flushed as it goes, so a run that is interrupted still leaves usable
data behind.

Champions are hashed. That makes it possible to say which generation of which
run produced the model that ships, rather than hoping the filename is right.
"""

import csv
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time

MODEL_VERSION = 2

COLUMNS = [
    "generation", "wall_seconds", "cum_wall_seconds", "sigma",
    "evaluations", "ticks_simulated", "evals_per_sec", "ticks_per_sec",
    "fitness_best", "fitness_mean", "fitness_median", "fitness_std",
    "fitness_p25", "fitness_p75", "fitness_worst",
    "gates_best", "gates_mean", "gates_median",
    "laps_best", "laps_mean", "lap_completion_rate",
    "best_lap_ticks", "best_lap_seconds", "collisions_mean",
    "term_crash", "term_stall", "term_timeout", "term_finished",
    "champion_fitness", "champion_sha1",
]


def git_state():
    def run(*args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL,
                                           cwd=os.path.dirname(os.path.abspath(__file__))
                                           ).decode().strip()
        except Exception:
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _version(package):
    """Report a dependency's version without importing it.

    The ga package is covered by tests/test_purity.py, which forbids importing
    a numerical library here - so the version is read from installed metadata
    rather than from the module.
    """
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def environment():
    return {
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "git": git_state(),
    }


def percentile(values, fraction):
    """Nearest-rank percentile. No numpy: this module stays pure Python."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[index]


def mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return (sum((v - average) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def median(values):
    return percentile(values, 0.5)


class RunLog:
    """Writes generations.csv, config.json and the champion checkpoints."""

    def __init__(self, directory, config):
        self.directory = directory
        self.champion_dir = os.path.join(directory, "champions")
        os.makedirs(self.champion_dir, exist_ok=True)

        self.csv_path = os.path.join(directory, "generations.csv")
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._file.flush()

        self.started = time.time()
        self.best_fitness = None

        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as fh:
            json.dump({"config": config, "environment": environment(),
                       "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=2)

    def write(self, row):
        self._writer.writerow({name: row.get(name, "") for name in COLUMNS})
        self._file.flush()

    def save_champion(self, network, generation, meta):
        path = os.path.join(self.champion_dir, "gen%04d.pkl" % generation)
        digest = export_network(path, [network], meta)
        export_network(os.path.join(self.directory, "champion.pkl"), [network], meta)
        return digest

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def export_network(path, networks, meta=None):
    """Write the versioned model format and return its sha1."""
    payload = {
        "version": MODEL_VERSION,
        "networks": list(networks),
        "normalise_inputs": True,
        "meta": dict(meta or {}),
    }
    blob = pickle.dumps(payload, protocol=4)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    return hashlib.sha1(blob).hexdigest()


def summarise(generation, sigma, scores, results, elapsed, cumulative, fps, laps_target):
    """Collapse a generation's rollouts into one CSV row."""
    flat = [result for group in results for result in group]

    gates = [max(r.total_gates for r in group) for group in results]
    laps = [max(r.valid_laps for r in group) for group in results]
    collisions = [mean([r.collisions for r in group]) for group in results]
    ticks = sum(r.ticks for r in flat)

    finished = [r for r in flat if r.terminated == "finished"]
    best_lap_ticks = min((r.ticks for r in finished), default="")

    counts = {"crash": 0, "stall": 0, "timeout": 0, "finished": 0}
    for result in flat:
        counts[result.terminated] = counts.get(result.terminated, 0) + 1

    return {
        "generation": generation,
        "wall_seconds": round(elapsed, 3),
        "cum_wall_seconds": round(cumulative, 3),
        "sigma": round(sigma, 6),
        "evaluations": len(flat),
        "ticks_simulated": ticks,
        "evals_per_sec": round(len(flat) / elapsed, 2) if elapsed > 0 else "",
        "ticks_per_sec": round(ticks / elapsed) if elapsed > 0 else "",
        "fitness_best": round(max(scores), 4),
        "fitness_mean": round(mean(scores), 4),
        "fitness_median": round(median(scores), 4),
        "fitness_std": round(stdev(scores), 4),
        "fitness_p25": round(percentile(scores, 0.25), 4),
        "fitness_p75": round(percentile(scores, 0.75), 4),
        "fitness_worst": round(min(scores), 4),
        "gates_best": max(gates),
        "gates_mean": round(mean(gates), 3),
        "gates_median": median(gates),
        "laps_best": max(laps),
        "laps_mean": round(mean(laps), 3),
        # The headline: what fraction of the population can get round at all.
        "lap_completion_rate": round(sum(1 for value in laps if value >= laps_target)
                                     / float(len(laps)), 4),
        "best_lap_ticks": best_lap_ticks,
        "best_lap_seconds": round(best_lap_ticks / float(fps), 3) if finished else "",
        "collisions_mean": round(mean(collisions), 3),
        "term_crash": counts["crash"],
        "term_stall": counts["stall"],
        "term_timeout": counts["timeout"],
        "term_finished": counts["finished"],
    }
