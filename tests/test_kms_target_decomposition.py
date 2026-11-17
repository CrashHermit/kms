"""Atomic target decomposition through the configured inference boundary."""

import asyncio

import dspy
import pytest
from dspy.utils import DummyLM

from kms.config.settings import Settings
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import NativePredictor, generation_signature
from kms.learning.probe.goal.models import ClarifiedGoal
from kms.learning.probe.targets.inference import (
    TargetDecompositionModule,
    TargetDecompositionSignature,
)


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def test_decomposition_preserves_target_order_and_clarified_goal():
    goal = ClarifiedGoal(goal='Understand derivatives and their use in motion')
    lm = DummyLM(
        [
            {
                'targets': [
                    'What does the derivative of a function represent?',
                    'How does the derivative of position describe velocity?',
                ]
            }
        ]
    )
    backend = dspy.Predict(generation_signature(TargetDecompositionSignature))
    backend.set_lm(lm)
    module = TargetDecompositionModule(
        NativePredictor(
            backend,
            TargetDecompositionSignature,
            budget=TokenBudget(MaterialTokenEstimator(_Counter()), 100000),
            image_budget=None,
            supports_images=False,
        )
    )
    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        targets = asyncio.run(module.acall(goal=goal))
    assert targets == [
        'What does the derivative of a function represent?',
        'How does the derivative of position describe velocity?',
    ]
    message = [
        item for item in lm.history[0]['messages'] if item['role'] == 'user'
    ][-1]
    assert goal.goal in str(message['content'])


def test_oversized_goal_is_not_sent_to_model():
    class MustNotRun(dspy.Module):
        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError('Oversized goal reached the model')

    module = TargetDecompositionModule(
        NativePredictor(
            MustNotRun(),
            TargetDecompositionSignature,
            budget=TokenBudget(MaterialTokenEstimator(_Counter()), 1),
            image_budget=None,
            supports_images=False,
        )
    )
    with pytest.raises(InputBudgetExceeded):
        asyncio.run(
            module.acall(goal=ClarifiedGoal(goal='Understand calculus'))
        )


def test_decomposition_settings_are_configurable(monkeypatch):
    monkeypatch.setenv('KMS_TARGET_DECOMPOSITION__INFERENCE__PROFILE', 'custom')
    assert (
        Settings(_env_file=None).target_decomposition.inference.profile
        == 'custom'
    )
