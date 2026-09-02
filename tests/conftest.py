"""Test configuration.

The SDL driver variables must be set before pygame is imported anywhere, so
they are set at module scope here rather than in a fixture: conftest is
imported before any test module.
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("NNCAR_HEADLESS", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pytest  # noqa: E402


def _have(module):
    try:
        __import__(module)
    except ImportError:
        return False
    return True


HAVE_PYGAME = _have("pygame")

requires_pygame = pytest.mark.skipif(not HAVE_PYGAME, reason="pygame is not installed")


@pytest.fixture(scope="session")
def repo_root():
    return ROOT
