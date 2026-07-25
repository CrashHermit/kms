"""Open-relation referencer: one pass over any entity channel, with the target kind and the relation
both open (no REFERENCE_KINDS, no nine tactics). The extraction pass is injected via a scripted
module, so this exercises the text assembly, the vocabulary normalization, and the channel wiring."""

import asyncio

from kms.core import models
from kms.entity.referencers.open import (
    ReferencerNode,
    reference,
    reference_text,
)


class _ScriptedModule:
    """A stand-in Module returning fixed references, recording what text it was given."""

    def __init__(self, references):
        self._references = references
        self.seen = []

    async def references(self, content):
        self.seen.append(content)
        return list(self._references)


def test_references_are_drawn_from_the_statement_and_the_derivation():
    entity = models.Entity(
        type='example',
        contents=['Find the derivative.'],
        procedures=[
            models.Procedure(
                type='solution', contents=['By the Power Rule, $2x$.']
            )
        ],
    )
    assert reference_text(entity) == (
        'Find the derivative.\n\nBy the Power Rule, $2x$.'
    )


def test_an_entity_with_no_content_gets_no_references():
    module = _ScriptedModule(
        [models.Reference(target='X', kind='definition', relation='applies')]
    )
    e = asyncio.run(reference(models.Entity(type='definition'), module))
    assert e.refs == []
    assert module.seen == []  # the model was never called


def test_an_open_kind_and_relation_survive_unvalidated():
    # a physics law related by "conserves" — neither value is in any closed math vocabulary
    module = _ScriptedModule(
        [
            models.Reference(
                target='Conservation of Momentum',
                kind='law',
                relation='conserves',
            )
        ]
    )
    e = asyncio.run(
        reference(
            models.Entity(type='example', contents=['Two carts.']), module
        )
    )
    assert (e.refs[0].kind, e.refs[0].relation) == ('law', 'conserves')


def test_node_run_reads_and_writes_the_channel_it_was_constructed_with():
    module = _ScriptedModule(
        [models.Reference(target='Set', kind='definition', relation='assumes')]
    )
    node = ReferencerNode('theorem_entities', module=module)
    out = asyncio.run(
        node.run(
            {
                'theorem_entities': [
                    models.Entity(type='theorem', contents=['Let S be a Set.'])
                ]
            }
        )
    )
    assert list(out) == ['theorem_entities']
    assert out['theorem_entities'][0].refs[0].target == 'Set'
