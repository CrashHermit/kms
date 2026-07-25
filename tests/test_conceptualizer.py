"""Conceptualizer: the φ conceptualization axis over entities and over procedure steps. The two LLM
passes are injected via a scripted module, so this exercises the text assembly, the graph-context
inputs, and the tag cleaning — without dspy."""

import asyncio

from kms.core import models
from kms.entity.conceptualizer import (
    ConceptualizerNode,
    _clean,
    conceptualize,
    entity_text,
)


class _ScriptedModule:
    """A stand-in Module returning fixed tags, recording the inputs it was given."""

    def __init__(self, entity_tags, event_tags=None):
        self._entity_tags = entity_tags
        self._event_tags = event_tags or []
        self.entity_calls = []
        self.event_calls = []

    async def entity_concepts(self, content, entity_type, title, referenced):
        self.entity_calls.append((content, entity_type, title, referenced))
        return list(self._entity_tags)

    async def event_concepts(self, step, action):
        self.event_calls.append((step, action))
        return list(self._event_tags)


def _entity():
    return models.Entity(
        type='definition',
        title='Kernel',
        contents=['The kernel of a linear map $T$ is ...'],
        refs=[
            models.Reference(
                target='Linear Map', kind='definition', relation='assumes'
            )
        ],
        procedures=[
            models.Procedure(
                type='proof',
                contents=['Clear.'],
                steps=[
                    models.BodySegment(
                        description='Assume not.', action='assumption'
                    )
                ],
            )
        ],
    )


def test_the_entity_and_every_step_are_tagged():
    entity = _entity()
    asyncio.run(
        conceptualize(
            entity,
            _ScriptedModule(
                ['kernel of a linear map', 'linear algebra'],
                ['proof by contradiction'],
            ),
        )
    )
    assert entity.concepts == ['kernel of a linear map', 'linear algebra']
    assert entity.procedures[0].steps[0].concepts == ['proof by contradiction']


def test_the_entity_pass_gets_its_graph_context():
    # AutoSchemaKG's context enhancement: the neighbours disambiguate a polysemous concept
    module = _ScriptedModule(['kernel of a linear map'])
    asyncio.run(conceptualize(_entity(), module))
    content, entity_type, title, referenced = module.entity_calls[0]
    assert entity_type == 'definition' and title == 'Kernel'
    assert referenced == ['Linear Map']
    assert 'Clear.' in content  # the derivation is part of what it is about


def test_entity_text_joins_the_statement_and_every_procedure():
    assert entity_text(_entity()) == (
        'The kernel of a linear map $T$ is ...\n\nClear.'
    )


def test_an_entity_with_no_content_is_left_untagged():
    entity = models.Entity(type='definition')
    module = _ScriptedModule(['should not be used'])
    asyncio.run(conceptualize(entity, module))
    assert entity.concepts == []
    assert module.entity_calls == []


def test_clean_drops_blanks_and_dedupes_case_insensitively_keeping_order():
    assert _clean(['Group  Theory', '', 'group theory', 'algebra']) == [
        'Group Theory',
        'algebra',
    ]


def test_node_run_writes_back_to_the_entities_channel():
    node = ConceptualizerNode(module=_ScriptedModule(['algebra']))
    out = asyncio.run(
        node.run(
            {'entities': [models.Entity(type='definition', contents=['x'])]}
        )
    )
    assert list(out) == ['entities']
    assert out['entities'][0].concepts == ['algebra']
