"""The hand-written linear algebra, checked against numpy.

numpy is the oracle here, never the implementation - see test_purity.py.
"""

import random

import numpy as np
import pytest

from nncar import neural_network as nn


@pytest.mark.parametrize("rows,inner,cols", [(1, 1, 1), (2, 3, 1), (12, 6, 1), (4, 4, 4), (3, 5, 7)])
def test_dot_product_matches_numpy(rows, inner, cols):
    rng = random.Random(rows * 100 + inner * 10 + cols)
    a = [[rng.uniform(-5, 5) for _ in range(inner)] for _ in range(rows)]
    b = [[rng.uniform(-5, 5) for _ in range(cols)] for _ in range(inner)]
    assert np.allclose(nn.dot_product(a, b), np.matmul(np.array(a), np.array(b)), atol=1e-12)


def test_add_matrices_matches_numpy():
    rng = random.Random(0)
    a = [[rng.uniform(-5, 5) for _ in range(4)] for _ in range(3)]
    b = [[rng.uniform(-5, 5) for _ in range(4)] for _ in range(3)]
    assert np.allclose(nn.add_matrices(a, b), np.add(np.array(a), np.array(b)), atol=1e-12)


def test_random_matrix_shape_is_rows_by_columns():
    """Guards a genuine trap in the code.

    Layer.__init__(columns, rows) calls random_matrix(rows, columns), so the
    argument names invert between the two. If random_matrix's own convention
    ever drifts, every layer silently transposes.
    """
    m = nn.random_matrix(7, 3)
    assert len(m) == 7
    assert all(len(row) == 3 for row in m)


def test_layer_shapes_follow_columns_rows_signature():
    layer = nn.Layer(6, 12)          # 6 inputs, 12 outputs
    assert len(layer.weights) == 12
    assert len(layer.weights[0]) == 6
    assert len(layer.bias) == 12
    assert all(len(row) == 1 for row in layer.bias)


def test_dot_product_returns_a_new_matrix():
    a = [[1.0, 2.0]]
    b = [[3.0], [4.0]]
    out = nn.dot_product(a, b)
    out[0][0] = 999.0
    assert a == [[1.0, 2.0]] and b == [[3.0], [4.0]]


def test_layer_forward_matches_numpy_reference():
    random.seed(11)
    layer = nn.Layer(6, 12)
    x = [[random.uniform(-1, 1)] for _ in range(6)]
    expected = np.matmul(np.array(layer.weights), np.array(x)) + np.array(layer.bias)
    assert np.allclose(layer.forward(x), expected, atol=1e-12)
