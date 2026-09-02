"""The hand-written merge sort used for race placement and GA selection.

Checked against sorted() including stability, which the implementation gets
right in both directions via its `elif left_value == right_value` branch.
"""

import random

import pytest

from conftest import requires_pygame

pytestmark = requires_pygame


@pytest.fixture(scope="module")
def merge_sort():
    from nncar.game import merge_sort

    return merge_sort


@pytest.mark.parametrize("size", [0, 1, 2, 3, 17, 1000])
def test_matches_sorted(merge_sort, size):
    data = [random.Random(size).randint(-50, 50) for _ in range(size)]
    assert merge_sort(list(data)) == sorted(data)
    assert merge_sort(list(data), reverse=True) == sorted(data, reverse=True)


@pytest.mark.parametrize("data", [
    [],
    [1],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [2, 2, 2, 2],
])
def test_edge_cases(merge_sort, data):
    assert merge_sort(list(data)) == sorted(data)


def test_sorts_by_key(merge_sort):
    words = ["ccc", "a", "dddd", "bb"]
    assert merge_sort(list(words), key=len) == sorted(words, key=len)
    assert merge_sort(list(words), key=len, reverse=True) == sorted(words, key=len, reverse=True)


class Item:
    """Distinguishable by identity but equal by key, to test stability."""

    def __init__(self, key, tag):
        self.key = key
        self.tag = tag


@pytest.mark.parametrize("reverse", [False, True])
def test_is_stable(merge_sort, reverse):
    items = [Item(1, "a"), Item(0, "b"), Item(1, "c"), Item(0, "d"), Item(1, "e")]
    got = merge_sort(list(items), key=lambda i: i.key, reverse=reverse)
    expected = sorted(items, key=lambda i: i.key, reverse=reverse)
    assert [i.tag for i in got] == [i.tag for i in expected]


def test_sorts_by_the_leaderboard_tuple_key(merge_sort):
    """The real usage: (laps, checkpoints_passed, -last_checkpoint) descending."""
    rows = [(1, 3, -10.0), (2, 0, -5.0), (1, 3, -8.0), (0, 9, -1.0)]
    assert merge_sort(list(rows), reverse=True) == sorted(rows, reverse=True)


def test_returns_a_new_list(merge_sort):
    data = [3, 1, 2]
    assert merge_sort(data) is not data
    assert data == [3, 1, 2]
