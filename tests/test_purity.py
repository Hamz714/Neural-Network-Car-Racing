"""The neural network is implemented from scratch, and stays that way.

This is the project's headline claim. Asserting it with the AST rather than
trusting a code review means the build breaks the first time somebody reaches
for numpy inside the network.
"""

import ast
import os

from nncar import neural_network

ALLOWED = {"random", "math", "copy"}


def _imported_modules(path):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)

    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def test_network_module_uses_only_the_standard_library():
    imported = _imported_modules(neural_network.__file__)
    assert imported <= ALLOWED, "third-party import in the network: %s" % sorted(imported - ALLOWED)


def test_ga_package_is_also_pure():
    """The genetic algorithm is part of the same claim.

    Logging and stdlib multiprocessing are fine; a numerical library is not.
    """
    from nncar import ga

    banned = {"numpy", "scipy", "torch", "tensorflow", "jax", "sklearn"}
    ga_dir = os.path.dirname(ga.__file__)
    for name in sorted(os.listdir(ga_dir)):
        if not name.endswith(".py"):
            continue
        offenders = _imported_modules(os.path.join(ga_dir, name)) & banned
        assert not offenders, "%s imports %s" % (name, sorted(offenders))
