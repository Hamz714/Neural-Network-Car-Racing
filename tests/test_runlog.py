"""Run logging: the statistics, the CSV, and the model format.

Everything a training run reports comes through here, so a quiet error in this
module would not break a run - it would just make its numbers wrong, which is
worse.
"""

import csv
import json
import os
import pickle
import random

import pytest

from nncar.ga import runlog
from nncar.neural_network import Network
from nncar.sim.rollout import RolloutResult


class TestStatistics:
    def test_mean(self):
        assert runlog.mean([1, 2, 3, 4]) == 2.5
        assert runlog.mean([]) == 0.0

    def test_stdev_matches_the_sample_definition(self):
        import statistics

        values = [3.0, 1.5, 4.25, 9.0, 2.0]
        assert runlog.stdev(values) == pytest.approx(statistics.stdev(values))
        assert runlog.stdev([7.0]) == 0.0
        assert runlog.stdev([]) == 0.0

    def test_percentiles_are_ordered_and_bounded(self):
        values = list(range(101))
        assert runlog.percentile(values, 0.0) == 0
        assert runlog.percentile(values, 1.0) == 100
        assert runlog.percentile(values, 0.5) == 50
        assert runlog.percentile(values, 0.25) <= runlog.percentile(values, 0.75)

    def test_percentile_of_nothing_is_zero_not_an_error(self):
        assert runlog.percentile([], 0.5) == 0.0

    def test_median_of_an_even_count_uses_nearest_rank(self):
        assert runlog.median([1, 2, 3, 4]) in (2, 3)


def rollout(**kwargs):
    defaults = dict(ticks=500, total_gates=0, valid_laps=0, lap_gates=(),
                    collisions=0, terminated="stall")
    defaults.update(kwargs)
    # Summaries report the scored quantity - the best single circuit.
    defaults.setdefault("best_circuit_gates", defaults["total_gates"])
    return RolloutResult(**defaults)


class TestSummarise:
    @staticmethod
    def build():
        results = [
            [rollout(total_gates=10, valid_laps=1, lap_gates=(10,), ticks=900,
                     terminated="finished"),
             rollout(total_gates=10, valid_laps=1, lap_gates=(10,), ticks=1000,
                     terminated="finished")],
            [rollout(total_gates=4, ticks=600, collisions=2),
             rollout(total_gates=3, ticks=500, collisions=1)],
            [rollout(total_gates=0, ticks=120, collisions=5, terminated="crash"),
             rollout(total_gates=0, ticks=140, collisions=5, terminated="crash")],
        ]
        scores = [2000.0, 400.0, -25.0]
        return runlog.summarise(7, 0.1, scores, results, elapsed=2.0,
                                cumulative=14.0, fps=50, laps_target=1)

    def test_reports_the_generation_and_its_cost(self):
        row = self.build()
        assert row["generation"] == 7
        assert row["wall_seconds"] == 2.0
        assert row["cum_wall_seconds"] == 14.0
        assert row["evaluations"] == 6
        assert row["evals_per_sec"] == 3.0

    def test_fitness_statistics(self):
        row = self.build()
        assert row["fitness_best"] == 2000.0
        assert row["fitness_worst"] == -25.0
        assert row["fitness_mean"] == pytest.approx((2000 - 25 + 400) / 3, abs=1e-3)

    def test_lap_completion_counts_individuals_not_rollouts(self):
        """One of three individuals got round, so the rate is a third.

        Counting rollouts instead would report two of six - the same number
        here by coincidence, but not in general, and the question being asked
        is what fraction of the population can drive.
        """
        row = self.build()
        assert row["lap_completion_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert row["laps_best"] == 1

    def test_best_lap_time_comes_from_finished_rollouts_only(self):
        row = self.build()
        assert row["best_lap_ticks"] == 900
        assert row["best_lap_seconds"] == pytest.approx(18.0)

    def test_a_generation_with_no_finisher_leaves_lap_time_blank(self):
        results = [[rollout(total_gates=2, ticks=300)]]
        row = runlog.summarise(0, 0.1, [200.0], results, 1.0, 1.0, 50, 1)
        assert row["best_lap_seconds"] == ""
        assert row["lap_completion_rate"] == 0.0

    def test_termination_reasons_add_up(self):
        row = self.build()
        total = (row["term_crash"] + row["term_stall"]
                 + row["term_timeout"] + row["term_finished"])
        assert total == row["evaluations"]
        assert row["term_finished"] == 2
        assert row["term_crash"] == 2

    def test_an_individual_is_judged_on_its_best_rollout_for_progress(self):
        results = [[rollout(total_gates=1), rollout(total_gates=9)]]
        row = runlog.summarise(0, 0.1, [100.0], results, 1.0, 1.0, 50, 1)
        assert row["gates_best"] == 9

    def test_progress_is_the_best_circuit_not_the_sum_across_circuits(self):
        """Two sloppy laps must not report as more progress than one clean one."""
        sloppy = [[rollout(total_gates=15, best_circuit_gates=8, lap_gates=(7, 8))]]
        assert runlog.summarise(0, 0.1, [1.0], sloppy, 1.0, 1.0, 50, 1)["gates_best"] == 8

    def test_every_column_is_produced(self):
        row = self.build()
        missing = [name for name in runlog.COLUMNS
                   if name not in row and name not in ("champion_fitness", "champion_sha1")]
        assert not missing


class TestRunLog:
    def test_writes_a_readable_csv_and_a_config(self, tmp_path):
        directory = str(tmp_path / "run")
        os.makedirs(directory)
        with runlog.RunLog(directory, {"seed": 7}) as log:
            log.write({"generation": 0, "fitness_best": 1.0})
            log.write({"generation": 1, "fitness_best": 2.0})

        with open(os.path.join(directory, "generations.csv"), newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert [row["generation"] for row in rows] == ["0", "1"]
        assert set(rows[0]) == set(runlog.COLUMNS)

        with open(os.path.join(directory, "config.json")) as fh:
            config = json.load(fh)
        assert config["config"]["seed"] == 7
        assert "python" in config["environment"]

    def test_rows_are_flushed_as_they_are_written(self, tmp_path):
        """An interrupted run must still leave usable data behind."""
        directory = str(tmp_path / "run")
        os.makedirs(directory)
        log = runlog.RunLog(directory, {})
        log.write({"generation": 0, "fitness_best": 5.0})

        with open(os.path.join(directory, "generations.csv"), newline="") as fh:
            assert len(list(csv.DictReader(fh))) == 1
        log.close()

    def test_saving_a_champion_writes_both_the_checkpoint_and_the_latest(self, tmp_path):
        directory = str(tmp_path / "run")
        os.makedirs(directory)
        with runlog.RunLog(directory, {}) as log:
            digest = log.save_champion(Network(random.Random(1)), 12, {"fitness": 9.0})

        assert os.path.exists(os.path.join(directory, "champions", "gen0012.pkl"))
        assert os.path.exists(os.path.join(directory, "champion.pkl"))
        assert len(digest) == 40

    def test_environment_records_enough_to_repeat_the_run(self, tmp_path):
        environment = runlog.environment()
        assert environment["python"]
        assert environment["cpu_count"]
        assert "commit" in environment["git"]


class TestModelFormat:
    def test_round_trips(self, tmp_path):
        path = str(tmp_path / "model.pkl")
        network = Network(random.Random(3))
        runlog.export_network(path, [network], {"generation": 5})

        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        assert payload["version"] == runlog.MODEL_VERSION
        assert payload["normalise_inputs"] is True
        assert payload["meta"]["generation"] == 5
        assert payload["networks"][0].layers[0].weights == network.layers[0].weights

    def test_the_digest_identifies_the_contents(self, tmp_path):
        """Provenance: which generation of which run produced a shipped model."""
        network = Network(random.Random(4))
        first = runlog.export_network(str(tmp_path / "a.pkl"), [network], {"n": 1})
        same = runlog.export_network(str(tmp_path / "b.pkl"), [network], {"n": 1})
        other = runlog.export_network(str(tmp_path / "c.pkl"), [network], {"n": 2})
        assert first == same
        assert first != other

    def test_the_game_can_read_what_the_trainer_writes(self, tmp_path):
        """The two sides of the format, checked against each other."""
        path = str(tmp_path / "hard.pkl")
        networks = [Network(random.Random(seed)) for seed in range(3)]
        runlog.export_network(path, networks, {})

        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        assert len(payload["networks"]) == 3
        assert payload.get("normalise_inputs") is True
