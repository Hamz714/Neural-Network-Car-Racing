"""Convert the original pickled networks to the versioned model format.

The shipped easy/medium/hard.txt files are raw pickles of a `Network` instance
from when the module lived at the repository root, so they name `neural_network`
as their module and stop loading the moment the package moves. This script
unpickles them under an alias and re-emits them as versioned dictionaries.

Run once:

    python scripts/migrate_models.py

The files it produces are placeholders; `scripts/train.py --export` replaces
them with genuinely evolved networks.
"""

import argparse
import os
import pickle
import sys

import _bootstrap  # noqa: F401

from nncar import assets, neural_network

MODEL_VERSION = 2
DIFFICULTIES = ("easy", "medium", "hard")


def load_legacy(path):
    """Unpickle a model written when neural_network was a top-level module."""
    # pickle resolves classes by module name, so make the old name resolve to
    # the new module for the duration of the load.
    previous = sys.modules.get("neural_network")
    sys.modules["neural_network"] = neural_network
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    finally:
        if previous is None:
            del sys.modules["neural_network"]
        else:
            sys.modules["neural_network"] = previous


def save(path, networks, meta):
    """Write the versioned model format.

    normalise_inputs records which input convention the networks were trained
    under, so that models predating normalisation keep driving correctly.
    """
    payload = {
        "version": MODEL_VERSION,
        "networks": list(networks),
        "normalise_inputs": meta.pop("normalise_inputs", False),
        "meta": meta,
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep-legacy", action="store_true",
                        help="leave the original .txt files in place")
    args = parser.parse_args()

    for difficulty in DIFFICULTIES:
        legacy = assets.model(difficulty + ".txt")
        if not os.path.exists(legacy):
            print("skip %-6s (no %s)" % (difficulty, os.path.basename(legacy)))
            continue

        obj = load_legacy(legacy)
        networks = obj if isinstance(obj, list) else [obj]

        target = assets.model(difficulty + ".pkl")
        save(target, networks, {
            "source": "migrated from %s" % os.path.basename(legacy),
            "trained": False,
            "normalise_inputs": False,
            "note": "placeholder: a hand-picked random network, not an evolved one",
        })
        params = sum(len(l.weights) * len(l.weights[0]) + len(l.bias)
                     for l in networks[0].layers)
        print("%-6s -> %s  (%d network(s), %d parameters)"
              % (difficulty, os.path.basename(target), len(networks), params))

        if not args.keep_legacy:
            os.remove(legacy)

    print("done")


if __name__ == "__main__":
    main()
