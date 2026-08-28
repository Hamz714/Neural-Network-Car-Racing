"""Every asset the code asks for exists, and every asset shipped is asked for.

Button images are located by string concatenation - `Button(..., "quit")` loads
`quitbutton.png` - so a mistyped name is not a syntax error, an import error or
anything else a linter would catch. It is a crash the first time somebody opens
that screen. Resolving the names statically turns that into a test failure.
"""

import ast
import os

import pytest

from conftest import requires_pygame

from nncar import assets

SOURCE_FILES = ["entities.py", "screens.py", "game.py"]


def _referenced_images():
    """Image filenames the source asks for, including concatenated button names."""
    names = set()
    stems = set()

    for filename in SOURCE_FILES:
        path = os.path.join(os.path.dirname(assets.__file__), filename)
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value.endswith(".png"):
                names.add(node.value)
            if isinstance(node, ast.Call):
                called = getattr(node.func, "attr", getattr(node.func, "id", "")) or ""
                if "Button" in called and len(node.args) >= 5:
                    argument = node.args[4]
                    if isinstance(argument, ast.Constant):
                        stems.add(argument.value)

    names |= {stem + "button.png" for stem in stems}
    # "button.png" only ever appears as the suffix in the concatenation, never
    # as a filename of its own.
    names.discard("button.png")
    names |= set(assets.BORDER_NAMES)

    from nncar import entities

    names |= {skin for skin, _price in entities.CAR_SKINS}
    return names, stems


@requires_pygame
def test_every_button_name_resolves_to_a_file():
    _, stems = _referenced_images()
    assert stems, "found no button names to check - the parser has stopped working"

    missing = [stem for stem in sorted(stems)
               if not os.path.exists(assets.image(stem + "button.png"))]
    assert not missing, "button images missing: %s" % missing


@requires_pygame
def test_every_referenced_image_exists():
    names, _ = _referenced_images()
    missing = sorted(name for name in names if not os.path.exists(assets.image(name)))
    assert not missing, "referenced but absent: %s" % missing


@requires_pygame
def test_no_unused_images_are_shipped():
    """Keeps the repository from accumulating assets nothing loads."""
    names, _ = _referenced_images()
    on_disk = {f for f in os.listdir(assets.IMAGE_DIR) if f.endswith(".png")}
    unused = sorted(on_disk - names)
    assert not unused, "shipped but never loaded: %s" % unused


@requires_pygame
def test_every_button_image_actually_loads():
    """Existing is not the same as being a readable image."""
    import pygame

    from nncar import entities as v

    _, stems = _referenced_images()
    for stem in sorted(stems):
        button = v.Button(0, 0, 10, 10, stem)
        assert isinstance(button.image, pygame.Surface)


def test_asset_paths_do_not_depend_on_the_working_directory():
    """The original code only ran from the repository root."""
    assert os.path.isabs(assets.image("red.png"))
    assert os.path.isabs(assets.audio("rockit.mp3"))
    assert os.path.isabs(assets.model("easy.pkl"))
    assert os.path.isabs(assets.PROGRESS_FILE)


def test_the_audio_track_is_present():
    assert os.path.exists(assets.audio("rockit.mp3"))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_a_model_ships_for_every_difficulty(difficulty):
    assert os.path.exists(assets.model(difficulty + ".pkl"))


@requires_pygame
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_every_shipped_model_loads_and_drives(difficulty):
    """A model that unpickles is not necessarily one the game can use."""
    from copy import deepcopy

    from nncar import entities as v
    from nncar import game as f

    networks, _normalise = f.load_model(difficulty)
    assert networks and all(len(net.layers) == 4 for net in networks)

    v.track = v.Track(1, load_visuals=False)
    v.NPC.start_positions = deepcopy(v.NPC_START_POS)
    cars = f.load(difficulty)
    assert len(cars) == 5
    assert len({id(car.network) for car in cars}) == 5, "opponents share a network"

    for car in cars[:2]:
        car.update_sensors()
        car.move()
