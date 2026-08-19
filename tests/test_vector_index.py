"""Tests for deterministic construction-time vector search."""

import pytest

from kms.core.vector_index import ExactCosineIndex


def test_empty_index_and_zero_top_k() -> None:
    index = ExactCosineIndex(2)

    assert len(index) == 0
    assert index.search([1.0, 0.0], top_k=5) == []
    index.add([])
    assert index.search([1.0, 0.0], top_k=0) == []


def test_search_normalizes_vectors_and_returns_stable_keys() -> None:
    index = ExactCosineIndex(2)
    index.add([('far', [0.0, 2.0]), ('near', [3.0, 0.0])])

    matches = index.search([10.0, 0.0], top_k=2)

    assert [match.key for match in matches] == ['near', 'far']
    assert matches[0].score == pytest.approx(1.0)


def test_ties_are_sorted_by_stable_key() -> None:
    index = ExactCosineIndex(2)
    index.add([('zeta', [1.0, 0.0]), ('alpha', [2.0, 0.0])])

    assert [match.key for match in index.search([1.0, 0.0], top_k=2)] == [
        'alpha',
        'zeta',
    ]


@pytest.mark.parametrize(
    ('operation', 'expected'),
    [
        ('dimension', 'dimension'),
        ('zero', 'zero norm'),
        ('nan', 'finite'),
        ('negative_top_k', 'negative'),
    ],
)
def test_invalid_vector_inputs_are_rejected(operation: str, expected: str) -> None:
    index = ExactCosineIndex(2)
    with pytest.raises(ValueError, match=expected):
        if operation == 'dimension':
            index.add([('bad', [1.0])])
        elif operation == 'zero':
            index.add([('bad', [0.0, 0.0])])
        elif operation == 'nan':
            index.add([('bad', [float('nan'), 1.0])])
        else:
            index.search([1.0, 0.0], top_k=-1)


def test_duplicate_keys_are_rejected_atomically() -> None:
    index = ExactCosineIndex(2)
    with pytest.raises(ValueError, match='duplicate'):
        index.add([('same', [1.0, 0.0]), ('same', [0.0, 1.0])])
    assert len(index) == 0

    index.add([('same', [1.0, 0.0])])
    with pytest.raises(ValueError, match='already exists'):
        index.add([('same', [0.0, 1.0]), ('new', [1.0, 1.0])])
    assert len(index) == 1
