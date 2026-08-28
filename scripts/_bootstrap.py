"""Put src/ on sys.path so the scripts run from a plain checkout.

Not needed after `pip install -e .`, but the project should not require an
install step just to be tried out.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
