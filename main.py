"""Launch the game.

Kept at the repository root so `python main.py` still works. The package itself
lives in src/nncar; see scripts/ for the trainer and benchmarks.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from nncar.screens import main

if __name__ == "__main__":
    main()
