import asyncio
from types import SimpleNamespace

from kms.construction import procedure_creator
from kms.core import content, models


def test_step_splitter_returns_model_stages_without_fallback():
    prediction = SimpleNamespace(steps=['Conclude.'])

    steps = procedure_creator.StepSplitter.decode(
        None,
        prediction,
        statement=procedure_creator.content.Content.from_text('Claim.'),
        procedure=procedure_creator.content.Content.from_text(
            'Set up the claim. Verify the first condition. Conclude.'
        ),
        entity_definitions='definition',
    )

    assert steps == prediction.steps


def test_source_local_materialization_cannot_touch_unprovided_sources(
    monkeypatch,
):
    async def _split(_splitter, _statement, procedure, _definitions):
        return [procedure.render()]

    monkeypatch.setattr(procedure_creator, '_split_procedure', _split)
    source_a = models.ProcedureMaterializationInput(
        source='source-a',
        statement_uuid='statement-a',
        statement=content.Content.from_text('Statement A.'),
        procedure_uuid='procedure-a',
        procedure=content.Content.from_text('Procedure A.'),
        member_count=1,
    )
    source_b = models.ProcedureMaterializationInput(
        source='source-b',
        statement_uuid='statement-b',
        statement=content.Content.from_text('Statement B.'),
        procedure_uuid='procedure-b',
        procedure=content.Content.from_text('Procedure B.'),
        member_count=1,
    )

    only_a = asyncio.run(
        procedure_creator.create_procedures([source_a], language_model=None)
    )
    assert only_a['procedures_created'] == 1
    assert [update.source for update in only_a['procedure_step_updates']] == [
        'source-a'
    ]
    assert all(
        'source-b' not in repr(value)
        for value in only_a.values()
        if isinstance(value, list)
    )

    both = asyncio.run(
        procedure_creator.create_procedures(
            [source_a, source_b], language_model=None
        )
    )
    assert both['procedures_created'] == 2
    assert sorted(
        update.source for update in both['procedure_step_updates']
    ) == ['source-a', 'source-b']


def test_step_splitter_accepts_complete_grouped_stages():
    prediction = SimpleNamespace(
        steps=['Set up the claim. Verify the first condition.', 'Conclude.']
    )

    steps = procedure_creator.StepSplitter.decode(
        None,
        prediction,
        statement=procedure_creator.content.Content.from_text('Claim.'),
        procedure=procedure_creator.content.Content.from_text(
            'Set up the claim. Verify the first condition. Conclude.'
        ),
        entity_definitions='definition',
    )

    assert steps == prediction.steps
