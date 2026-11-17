"""Knowledge sufficiency decisions at the native inference boundary."""

import asyncio

import dspy
import pytest
from dspy.utils import DummyLM

from kms.config.inference import TextInferenceSettings
from kms.config.settings import Settings
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import NativePredictor, generation_signature
from kms.inference.predictors import build_token_budget
from kms.knowledge.models import Argument, KnowledgeContext, Proposition
from kms.knowledge.sufficiency.inference import (
    SufficiencyModule,
    SufficiencySignature,
)
from kms.knowledge.sufficiency.models import (
    InsufficientKnowledge,
    SufficiencyRequest,
    SufficientKnowledge,
)


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Tokenizers:
    def for_inference(
        self, inference: TextInferenceSettings
    ) -> MaterialTokenEstimator:
        return MaterialTokenEstimator(_Counter())


class _MustNotRun(dspy.Module):
    async def aforward(self, **kwargs: object) -> object:
        raise AssertionError('Over-budget judgment reached the model')


def _context() -> KnowledgeContext:
    premise = Proposition(
        uuid='premise',
        name='Force',
        statement='A net force accelerates a mass.',
    )
    conclusion = Proposition(
        uuid='conclusion',
        name='Gravity',
        statement='Gravity accelerates a falling object.',
    )
    return KnowledgeContext(
        matched=[conclusion],
        propositions=[premise, conclusion],
        arguments=[
            Argument(
                uuid='argument',
                premise_uuids=[premise.uuid],
                conclusion_uuid=conclusion.uuid,
            )
        ],
    )


def test_structured_sufficiency_and_missing_knowledge() -> None:
    settings = Settings(_env_file=None)
    budget = build_token_budget(
        settings,
        _Tokenizers(),
        settings.knowledge_sufficiency.inference,
        input_token_budget=100000,
        safety_margin_tokens=256,
        signature=SufficiencySignature,
    )
    lm = DummyLM(
        [
            {
                'judgment': {
                    'kind': 'sufficient',
                    'explanation': 'The supplied argument explains the acceleration.',
                }
            },
            {
                'judgment': {
                    'kind': 'insufficient',
                    'explanation': 'The context contains no relevant claims.',
                    'missing_knowledge': 'How gravity causes acceleration.',
                }
            },
        ]
    )
    backend = dspy.Predict(generation_signature(SufficiencySignature))
    backend.set_lm(lm)
    module = SufficiencyModule(
        NativePredictor(
            backend,
            SufficiencySignature,
            budget=budget,
            image_budget=None,
            supports_images=False,
        )
    )
    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        sufficient = asyncio.run(
            module.acall(
                request=SufficiencyRequest(
                    understanding='Why does gravity accelerate objects?',
                    context=_context(),
                )
            )
        )
        insufficient = asyncio.run(
            module.acall(
                request=SufficiencyRequest(
                    understanding='Why does gravity accelerate objects?',
                    context=KnowledgeContext(
                        matched=[], propositions=[], arguments=[]
                    ),
                )
            )
        )

    assert sufficient == SufficientKnowledge(
        explanation='The supplied argument explains the acceleration.'
    )
    assert insufficient == InsufficientKnowledge(
        explanation='The context contains no relevant claims.',
        missing_knowledge='How gravity causes acceleration.',
    )
    message = [
        item for item in lm.history[0]['messages'] if item['role'] == 'user'
    ][-1]
    content = str(message['content'])
    assert 'Why does gravity accelerate objects?' in content
    assert 'A net force accelerates a mass.' in content
    assert 'argument' in content
    assert '"origin"' not in content


def test_oversized_context_fails_instead_of_judging_a_partial_graph() -> None:
    budget = TokenBudget(MaterialTokenEstimator(_Counter()), 1)
    module = SufficiencyModule(
        NativePredictor(
            _MustNotRun(),
            SufficiencySignature,
            budget=budget,
            image_budget=None,
            supports_images=False,
        )
    )
    with pytest.raises(InputBudgetExceeded):
        asyncio.run(
            module.acall(
                request=SufficiencyRequest(
                    understanding='Why does it fall?', context=_context()
                )
            )
        )
