"""Every screen renders without crashing.

The menus are written as blocking `while True` loops that only exit when
`game.quit()` sees a QUIT event, so they cannot simply be called from a test.
Counting frames in `pygame.display.flip` and posting a QUIT after a few gives
each loop a controlled exit, which is enough to prove it can build its buttons,
render a frame and read input.

Shallow by design - it will not catch a misplaced button - but it does catch
the failures that actually happen: a missing asset, a renamed attribute, a
screen nobody opened after a refactor.
"""

import pytest

from conftest import requires_pygame

pytestmark = requires_pygame

SCREENS = ["main", "start", "shop", "howtoplay"]


@pytest.fixture
def run_for_frames(monkeypatch):
    """Let a screen render `count` frames, then send it a QUIT."""
    import pygame

    def runner(function, count=5):
        original_flip = pygame.display.flip
        state = {"frames": 0}

        def counting_flip(*args, **kwargs):
            state["frames"] += 1
            if state["frames"] >= count:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            return original_flip(*args, **kwargs)

        monkeypatch.setattr(pygame.display, "flip", counting_flip)
        pygame.event.clear()
        try:
            function()
        except SystemExit:
            pass
        return state["frames"]

    return runner


@pytest.mark.parametrize("name", SCREENS)
def test_screen_renders(name, run_for_frames):
    import pygame

    from nncar import screens

    pygame.init()
    assert run_for_frames(getattr(screens, name)) >= 5


def test_the_race_loop_advances_the_world(run_for_frames):
    """The per-frame calls the race makes, driven directly.

    game() itself waits on input to leave, so this exercises the same sequence
    of calls its body makes rather than the loop around them.
    """
    from copy import deepcopy

    import pygame

    from nncar import entities as v
    from nncar import game as f

    pygame.init()
    v.reset_race_timer()
    v.player = v.PlayerCar()
    v.track = v.Track(1)
    v.NPC.start_positions = deepcopy(v.NPC_START_POS)
    v.NPC_cars = f.load("easy")
    v.track.leaderboard = v.NPC_cars + [v.player]

    before = [(car.world_x, car.world_y) for car in v.NPC_cars]
    for _ in range(40):
        f.move()
        f.checkpoints()
        f.NPC_collision()
        f.update_leaderboard()
        f.update_game()
    after = [(car.world_x, car.world_y) for car in v.NPC_cars]

    assert after != before, "no car moved in 40 frames"
    assert 1 <= v.player.placement <= 6


def test_the_leaderboard_ranks_by_progress():
    from nncar import game as f

    class Car:
        def __init__(self, laps, checkpoints, last, kind="NPC"):
            self.laps = laps
            self.checkpoints_passed = checkpoints
            self.last_checkpoint = last
            self.type = kind
            self.placement = 0

    leader = Car(1, 2, 30.0)
    middle = Car(0, 9, 25.0)
    trailing = Car(0, 9, 40.0)

    ordered = f.merge_sort([trailing, leader, middle],
                           key=f.custom_sort_key, reverse=True)
    assert ordered[0] is leader
    # Equal laps and checkpoints: whoever got there sooner is ahead.
    assert ordered.index(middle) < ordered.index(trailing)


def test_prize_money_scales_with_placement_difficulty_and_distance(monkeypatch):
    from nncar import entities as v
    from nncar import game as f

    monkeypatch.setattr(f, "update_progress", lambda: None)

    class Player:
        placement = 1

    monkeypatch.setattr(v, "player", Player)
    monkeypatch.setattr(v, "balance", 0)

    easy = f.calculate_balance("easy", 1)
    hard = f.calculate_balance("hard", 1)
    longer = f.calculate_balance("easy", 3)

    assert hard == easy * 3
    assert longer == easy * 3

    Player.placement = 5
    assert f.calculate_balance("easy", 1) < easy
