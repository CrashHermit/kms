import asyncio

from kms2.core.model import (
    SourceEntityHubJudgeInput,
    SourceEventHubJudgeInput,
    SourcePredicateHubJudgeInput,
)
from kms2.module.semantic.source_entity_hub_judge import (
    SourceEntityHubJudgeModule,
    SourceEntityHubJudgeSignature,
)
from kms2.module.semantic.source_event_hub_judge import (
    SourceEventHubJudgeModule,
    SourceEventHubJudgeSignature,
)
from kms2.module.semantic.source_predicate_hub_judge import (
    SourcePredicateHubJudgeModule,
    SourcePredicateHubJudgeSignature,
)


class _Prediction:
    belongs_in_same_hub = [True, False]


class _Predictor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _Prediction()

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return _Prediction()


def test_typed_judges_forward_ordered_batches_and_boolean_results():
    entity_predictor = _Predictor()
    entity_requests = [
        SourceEntityHubJudgeInput(
            left_name='Alice',
            left_description='person',
            right_name='A. Smith',
            right_description='same person',
        )
    ]
    assert asyncio.run(
        SourceEntityHubJudgeModule(entity_predictor).aforward(
            requests=entity_requests
        )
    ) == [True, False]
    assert entity_predictor.calls[0]['requests'] == entity_requests

    event_predictor = _Predictor()
    event_requests = [
        SourceEventHubJudgeInput(
            left_name='integration',
            left_description='process',
            right_name='system integration',
            right_description='same process',
        )
    ]
    assert SourceEventHubJudgeModule(event_predictor)(
        requests=event_requests
    ) == [
        True,
        False,
    ]
    assert event_predictor.calls[0]['requests'] == event_requests

    predicate_predictor = _Predictor()
    predicate_requests = [
        SourcePredicateHubJudgeInput(
            left_subject='Alice',
            left_predicate='supports',
            left_object='Acme',
            left_description='provides support',
            right_subject='Bob',
            right_predicate='supports',
            right_object='Acme',
            right_description='provides support',
        )
    ]
    assert asyncio.run(
        SourcePredicateHubJudgeModule(predicate_predictor).aforward(
            requests=predicate_requests
        )
    ) == [True, False]
    assert predicate_predictor.calls[0]['requests'] == predicate_requests


def test_typed_judge_signatures_state_type_specific_equivalence_rules():
    assert 'canonical identity' in SourceEntityHubJudgeSignature.__doc__
    assert 'same occurrence' in SourceEventHubJudgeSignature.__doc__
    assert 'directed relation' in SourcePredicateHubJudgeSignature.__doc__
