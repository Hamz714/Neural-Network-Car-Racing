"""The headless seam, the tick clock, and rollout determinism."""

import os

import pytest

from conftest import requires_pygame

pytestmark = requires_pygame


def test_the_dummy_drivers_are_active():
    assert os.environ.get("SDL_VIDEODRIVER") == "dummy"
    assert os.environ.get("SDL_AUDIODRIVER") == "dummy"

    from nncar.sim import headless

    assert headless.is_headless()


def test_importing_entities_starts_no_music():
    """Importing a module should not have audible side effects."""
    import pygame

    from nncar import entities  # noqa: F401

    assert not pygame.mixer.music.get_busy()


def test_a_headless_track_skips_the_backdrop():
    from nncar import entities as v

    assert v.Track(1, load_visuals=False).image is None
    assert v.Track(1, load_visuals=False).grid is not None


def test_the_grid_backend_frees_the_border_surfaces():
    """The three decoded borders are about 133 MB the game no longer holds."""
    from nncar import entities as v

    track = v.Track(1, load_visuals=False, backend="grid")
    assert track.border is None
    assert track.mask is None


def test_get_pixel_alpha_refuses_a_grid_backed_track():
    """It used to swallow every exception and answer "no wall"."""
    from nncar import entities as v

    track = v.Track(1, load_visuals=False, backend="grid")
    with pytest.raises(RuntimeError, match="mask backend"):
        track.get_pixel_alpha(10, 10)


def test_out_of_range_pixels_still_read_as_open():
    """The narrowed catch must keep the case it was actually there for."""
    from nncar import entities as v

    track = v.Track(1, load_visuals=False, backend="mask")
    assert track.get_pixel_alpha(-5000, -5000) is False
    assert track.get_pixel_alpha(10**7, 10**7) is False


class TestTickClock:
    def test_time_advances_with_ticks_not_the_wall_clock(self):
        from nncar.sim.clock import TickClock

        clock = TickClock(50)
        assert clock.now() == 0.0
        clock.advance(75)
        assert clock.now() == 1.5

    def test_an_event_fires_on_the_expected_tick(self):
        """A 1.5 s timer at 50 fps is 75 ticks; strictly-greater means 76."""
        from nncar.entities import Event
        from nncar.sim.clock import TickClock

        clock = TickClock(50)
        event = Event(1.5, clock=clock)

        fired_at = None
        for tick in range(1, 200):
            clock.advance()
            if event.check():
                fired_at = tick
                break
        assert fired_at == 76

    def test_a_short_event_fires_every_tick(self):
        from nncar.entities import Event
        from nncar.sim.clock import TickClock

        clock = TickClock(50)
        event = Event(0.01, clock=clock)
        for _ in range(10):
            clock.advance()
            assert event.check()


class TestRollout:
    @staticmethod
    def _run(seed=5, network_seed=2):
        import random

        from nncar import neural_network as nn
        from nncar.sim.rollout import RolloutConfig, simulate

        random.seed(network_seed)
        network = nn.Network()
        cfg = RolloutConfig(max_ticks=400)
        return simulate(network, start_index=0, cfg=cfg, seed=seed)

    def test_is_reproducible(self):
        first = self._run()
        second = self._run()
        assert first.as_dict() == second.as_dict()

    def test_reports_why_it_stopped(self):
        result = self._run()
        assert result.terminated in {"crash", "stall", "timeout", "finished"}
        assert result.ticks > 0

    def test_early_termination_actually_fires(self):
        """Untrained networks should stop well short of the tick limit.

        This is the single biggest saving in the whole trainer, so it is worth
        an explicit check rather than trusting it stays wired up.
        """
        import random

        from nncar import neural_network as nn
        from nncar.sim.rollout import RolloutConfig, build_track, simulate

        cfg = RolloutConfig(max_ticks=3000)
        track = build_track(cfg)
        lengths = []
        for seed in range(8):
            random.seed(seed)
            lengths.append(simulate(nn.Network(), start_index=seed % 5, cfg=cfg,
                                    seed=seed, track=track).ticks)

        average = sum(lengths) / len(lengths)
        assert average < cfg.max_ticks / 4, \
            "early termination is not saving much: mean %d ticks" % average

    def test_the_result_survives_pickling(self):
        """It has to cross a process boundary to reach the parent."""
        import pickle

        result = self._run()
        restored = pickle.loads(pickle.dumps(result, protocol=4))
        assert restored.as_dict() == result.as_dict()

    def test_evaluation_noise_is_off_so_fitness_is_a_property_of_the_network(self):
        from nncar.sim.rollout import RolloutConfig

        assert RolloutConfig().exploration_noise == 0.0

    def test_the_game_keeps_its_noisy_opponents(self):
        from nncar import entities as v

        assert v.NPC.DEFAULT_NOISE == 0.15

    def test_the_clock_is_restored_afterwards(self):
        from nncar.sim import clock as sim_clock

        before = sim_clock.ambient
        self._run()
        assert sim_clock.ambient is before


@pytest.mark.slow
def test_one_worker_and_several_agree_exactly():
    """The reproducibility guarantee that makes the metrics quotable.

    All mutation happens in the parent from a single seeded stream, so the
    number of workers changes only how fast a run goes, never what it produces.
    """
    import random

    from nncar.ga import population as ga
    from nncar.ga.evaluate import Evaluator
    from nncar.sim.rollout import RolloutConfig

    cfg = RolloutConfig(max_ticks=300)
    networks = ga.initial_population(6, random.Random(21))

    with Evaluator(cfg, workers=1, start_indices=(0, 1)) as serial:
        expected = serial.evaluate(networks, generation=0, base_seed=1234)

    with Evaluator(cfg, workers=4, start_indices=(0, 1)) as parallel:
        actual = parallel.evaluate(networks, generation=0, base_seed=1234)

    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert [r.as_dict() for r in got] == [r.as_dict() for r in want], \
            "individual %d differed between 1 and 4 workers" % index
