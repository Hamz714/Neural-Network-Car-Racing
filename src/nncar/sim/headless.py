"""Run the simulation without a window or a sound card.

SDL reads its driver choice from the environment when pygame.display /
pygame.mixer initialise, so these variables must be set before pygame is
imported anywhere in the process. Import this module first in any headless
entry point:

    from nncar.sim import headless
    headless.enable()
    from nncar import entities        # safe now

The training workers rely on the same thing. On Windows, multiprocessing uses
the "spawn" start method, and a spawned child imports the module containing the
task function *before* running the pool initialiser - so the variables have to
be inherited from the parent's environment rather than set in the initialiser.
"""

import os

#: Set by enable(), or by the environment before this module is imported.
_ENABLED = os.environ.get("NNCAR_HEADLESS", "") not in ("", "0", "false", "False")


def enable(audio=False):
    """Switch SDL to its dummy drivers. Idempotent, and safe to call early.

    Passing audio=True keeps real audio while still suppressing the window,
    which is occasionally useful when debugging a recorded run.
    """
    global _ENABLED
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    if not audio:
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["NNCAR_HEADLESS"] = "1"
    _ENABLED = True


def is_headless():
    """True when the process is running without a real display."""
    return _ENABLED


def environment():
    """The variables a child process needs to inherit to stay headless."""
    return {
        "SDL_VIDEODRIVER": os.environ.get("SDL_VIDEODRIVER", ""),
        "SDL_AUDIODRIVER": os.environ.get("SDL_AUDIODRIVER", ""),
        "NNCAR_HEADLESS": os.environ.get("NNCAR_HEADLESS", ""),
    }
