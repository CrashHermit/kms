from types import SimpleNamespace

from kms.construction import procedure_creator


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
