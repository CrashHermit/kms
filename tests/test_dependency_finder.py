"""Dependency finder: reference-grounded candidate pairs, pairwise judgment, and the cycle guard that
keeps the concept prerequisite graph a DAG. The judge is injected via a scripted module."""

import asyncio

from kms.core import models
from kms.entity.dependency_finder import (
    DependencyFinderNode,
    _acyclic,
    candidate_pairs,
    find_dependencies,
)


def _overlay():
    """A book defining "vector space", and an eigenvalue definition that cites it."""
    return [
        models.Entity(
            type='definition',
            id=0,
            title='Vector Space',
            concepts=['vector space', 'linear algebra'],
        ),
        models.Entity(
            type='definition',
            id=1,
            title='Eigenvalue',
            concepts=['eigenvalue', 'linear algebra'],
            refs=[
                models.Reference(
                    target='vector space',
                    kind='definition',
                    relation='is defined in terms of',
                )
            ],
        ),
    ]


class _ScriptedJudge:
    """A stand-in Module answering from a set of pairs it accepts; everything else is rejected."""

    def __init__(self, accept):
        self._accept = set(accept)
        self.asked = []

    async def depends(self, dependent, prerequisite):
        self.asked.append((dependent, prerequisite))
        return (dependent, prerequisite) in self._accept


# --- candidate generation ---


def test_candidates_are_grounded_in_a_reference_that_resolves_in_corpus():
    pairs = candidate_pairs(_overlay())
    assert ('eigenvalue', 'vector space', 1) in pairs
    # the shared coarse tag pairs with itself and is dropped as a self-pair
    assert all(a != b for a, b, _ in pairs)


def test_a_reference_to_something_the_book_never_defines_grounds_nothing():
    orphan = [
        models.Entity(
            type='theorem',
            id=0,
            title='Basis Theorem',
            concepts=['basis'],
            refs=[
                models.Reference(
                    target='Axiom of Choice',
                    kind='theorem',
                    relation='follows from',
                )
            ],
        )
    ]
    assert candidate_pairs(orphan) == []


def test_candidates_are_ranked_by_how_many_references_grounded_them():
    overlay = _overlay()
    # a second entity citing the same definition doubles the support for its concept pair
    overlay.append(
        models.Entity(
            type='theorem',
            id=2,
            title='Basis Theorem',
            concepts=['eigenvalue'],
            refs=[
                models.Reference(
                    target='Vector Space',
                    kind='definition',
                    relation='assumes',
                )
            ],
        )
    )
    assert candidate_pairs(overlay)[0][:2] == ('eigenvalue', 'vector space')


# --- judgment ---


def test_only_accepted_pairs_survive_the_judge():
    judge = _ScriptedJudge([('eigenvalue', 'vector space')])
    found = asyncio.run(find_dependencies(_overlay(), judge))
    assert [(d.dependent, d.prerequisite) for d in found] == [
        ('eigenvalue', 'vector space')
    ]


def test_no_candidates_means_no_judgments_at_all():
    judge = _ScriptedJudge([])
    assert asyncio.run(find_dependencies([], judge)) == []
    assert judge.asked == []


# --- the cycle guard ---


def test_a_co_defined_pair_keeps_only_its_better_evidenced_direction():
    # eigenvalue <-> eigenvector each look like the other's prerequisite; a DAG can hold one
    kept = _acyclic(
        [
            models.Dependency(
                dependent='eigenvalue', prerequisite='eigenvector', support=3
            ),
            models.Dependency(
                dependent='eigenvector', prerequisite='eigenvalue', support=1
            ),
        ]
    )
    assert [(d.dependent, d.prerequisite) for d in kept] == [
        ('eigenvalue', 'eigenvector')
    ]


def test_a_longer_cycle_is_broken_too():
    kept = _acyclic(
        [
            models.Dependency(dependent='a', prerequisite='b', support=3),
            models.Dependency(dependent='b', prerequisite='c', support=2),
            models.Dependency(dependent='c', prerequisite='a', support=1),
        ]
    )
    assert [(d.dependent, d.prerequisite) for d in kept] == [
        ('a', 'b'),
        ('b', 'c'),
    ]


def test_node_run_writes_the_dependency_channel():
    node = DependencyFinderNode(
        module=_ScriptedJudge([('eigenvalue', 'vector space')])
    )
    out = asyncio.run(node.run({'entities': _overlay()}))
    assert list(out) == ['concept_dependencies']
    assert out['concept_dependencies'][0].prerequisite == 'vector space'
