from types import SimpleNamespace

from kms.construction import procedure_creator


def test_step_splitter_returns_model_stages_without_fallback():
    prediction = SimpleNamespace(steps=['Conclude.'])

    steps = procedure_creator.StepSplitter.decode(
        None,
        prediction,
        procedure='Set up the claim. Verify the first condition. Conclude.',
    )

    assert steps == prediction.steps


def test_step_splitter_accepts_complete_grouped_stages():
    procedure = 'Set up the claim. Verify the first condition. Conclude.'
    prediction = SimpleNamespace(
        steps=['Set up the claim. Verify the first condition.', 'Conclude.']
    )

    steps = procedure_creator.StepSplitter.decode(
        None,
        prediction,
        procedure=procedure,
    )

    assert steps == prediction.steps
