import asyncio

from kms2.core.model import (
    SourceEntityHubJudgeInput,
    SourceEventHubJudgeInput,
    SourcePredicateHubJudgeInput,
)
from kms2.core.model.semantic.source_entity_hub import (
    SourceEntityHubJudgeDecision,
)
from kms2.core.model.semantic.source_event_hub import (
    SourceEventHubJudgeDecision,
)
from kms2.core.model.semantic.source_predicate_hub import (
    SourcePredicateHubJudgeDecision,
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
    def __init__(self, decisions: object) -> None:
        self.decisions = decisions


class _Predictor:
    def __init__(self, prediction: _Prediction) -> None:
        self.calls = []
        self.prediction = prediction

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.prediction

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return self.prediction


def test_typed_judges_forward_ordered_batches_and_indexed_results():
    entity_requests = [
        SourceEntityHubJudgeInput(
            index=0,
            left_name='Alice',
            left_description='person',
            right_name='A. Smith',
            right_description='same person',
        ),
        SourceEntityHubJudgeInput(
            index=1,
            left_name='Bob',
            left_description='person',
            right_name='Robert',
            right_description='different person',
        ),
    ]
    entity_decisions = [
        SourceEntityHubJudgeDecision(index=0, belongs_in_same_hub=True),
        SourceEntityHubJudgeDecision(index=1, belongs_in_same_hub=False),
    ]
    entity_predictor = _Predictor(_Prediction(entity_decisions))

    assert (
        asyncio.run(
            SourceEntityHubJudgeModule(entity_predictor).aforward(
                requests=entity_requests
            )
        )
        == entity_decisions
    )
    assert entity_predictor.calls[0]['requests'] == entity_requests

    event_requests = [
        SourceEventHubJudgeInput(
            index=0,
            left_name='integration',
            left_description='process',
            right_name='system integration',
            right_description='same process',
        ),
        SourceEventHubJudgeInput(
            index=1,
            left_name='deployment',
            left_description='release event',
            right_name='rollback',
            right_description='different event',
        ),
    ]
    event_decisions = [
        SourceEventHubJudgeDecision(index=0, belongs_in_same_hub=True),
        SourceEventHubJudgeDecision(index=1, belongs_in_same_hub=False),
    ]
    event_predictor = _Predictor(_Prediction(event_decisions))

    assert (
        SourceEventHubJudgeModule(event_predictor)(requests=event_requests)
        == event_decisions
    )
    assert event_predictor.calls[0]['requests'] == event_requests

    predicate_requests = [
        SourcePredicateHubJudgeInput(
            index=0,
            left_subject='Alice',
            left_predicate='supports',
            left_object='Acme',
            left_description='provides support',
            right_subject='Bob',
            right_predicate='supports',
            right_object='Acme',
            right_description='provides support',
        ),
        SourcePredicateHubJudgeInput(
            index=1,
            left_subject='Alice',
            left_predicate='causes',
            left_object='outage',
            left_description='creates a failure',
            right_subject='Alice',
            right_predicate='prevents',
            right_object='outage',
            right_description='avoids a failure',
        ),
    ]
    predicate_decisions = [
        SourcePredicateHubJudgeDecision(index=0, belongs_in_same_hub=True),
        SourcePredicateHubJudgeDecision(index=1, belongs_in_same_hub=False),
    ]
    predicate_predictor = _Predictor(_Prediction(predicate_decisions))

    assert (
        asyncio.run(
            SourcePredicateHubJudgeModule(predicate_predictor).aforward(
                requests=predicate_requests
            )
        )
        == predicate_decisions
    )
    assert predicate_predictor.calls[0]['requests'] == predicate_requests


def test_typed_judge_signatures_state_type_specific_equivalence_rules():
    assert 'canonical identity' in SourceEntityHubJudgeSignature.__doc__
    assert 'same occurrence' in SourceEventHubJudgeSignature.__doc__
    assert 'directed relation' in SourcePredicateHubJudgeSignature.__doc__
