"""Filesystem locations, resolved from this file rather than the process cwd.

Every asset in the original code was a bare relative filename, so the game only
ran when launched from the repository root. Resolving from ``__file__`` instead
means the package works from any working directory, which the test suite and the
multiprocessing training workers both depend on.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, os.pardir, os.pardir))

ASSET_DIR = os.path.join(ROOT, "assets")
IMAGE_DIR = os.path.join(ASSET_DIR, "images")
AUDIO_DIR = os.path.join(ASSET_DIR, "audio")
MODEL_DIR = os.path.join(ROOT, "models")
CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "results")

#: Player save file. Local state, deliberately not tracked in git.
PROGRESS_FILE = os.path.join(ROOT, "progress.txt")

#: The three track border layers, in the order the collision code tests them.
BORDER_NAMES = ("trackborder1.png", "trackborder2.png", "trackborder3.png")


def image(name):
    """Absolute path to an image asset."""
    return os.path.join(IMAGE_DIR, name)


def audio(name):
    """Absolute path to an audio asset."""
    return os.path.join(AUDIO_DIR, name)


def model(name):
    """Absolute path to a saved network."""
    return os.path.join(MODEL_DIR, name)


def border_paths():
    """Absolute paths to the three track border images."""
    return [image(name) for name in BORDER_NAMES]


def ensure_dir(path):
    """Create ``path`` if it does not exist; return it."""
    os.makedirs(path, exist_ok=True)
    return path


def load_image(name):
    """Load an image asset as a pygame Surface.

    Imported lazily so that pure-Python consumers (the network, the GA, most of
    the test suite) never pull pygame in as a side effect of touching this
    module.
    """
    import pygame

    return pygame.image.load(image(name))
