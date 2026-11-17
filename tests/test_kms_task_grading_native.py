"""Native-image and shared request-budget coverage for task grading."""

import asyncio
import base64
from io import BytesIO

import dspy
import pytest
from dspy.utils import DummyLM
from PIL import Image

from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import (
    NativePredictor,
    generation_prompt,
    generation_signature,
)
from kms.learning.references import GroundingReferences, SolutionTarget
from kms.learning.task.grading.examples import task_grading_examples
from kms.learning.task.grading.inference import (
    TaskGradingModule,
    TaskGradingSignature,
)
from kms.learning.task.grading.models import TaskGradingRequest
from kms.learning.task.models import (
    Confidence,
    FreeAnswer,
    Solution,
    Task,
    TaskResponse,
)
from kms.media.budget import ImagePayloadBudget
from kms.media.models import ImageAsset, ImagePart, Material, TextPart


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def _png(color: str) -> bytes:
    image = BytesIO()
    Image.new('RGB', (8, 8), color).save(image, format='PNG')
    return image.getvalue()


def _request() -> tuple[TaskGradingRequest, list[bytes]]:
    images = [_png(color) for color in ('red', 'green', 'blue')]
    assets = [
        ImageAsset((name,), image)
        for name, image in zip(
            ('prompt', 'solution', 'response'), images, strict=True
        )
    ]
    task = Task(
        uuid='task',
        understanding='Interpret the supplied diagram.',
        prompt=Material(
            parts=(TextPart(text='Prompt diagram'), ImagePart(asset=assets[0]))
        ),
        target=SolutionTarget(uuid='solution'),
        grounding=GroundingReferences(),
        solution=Solution(
            uuid='solution',
            material=Material(
                parts=(
                    TextPart(text='Reference diagram'),
                    ImagePart(asset=assets[1]),
                )
            ),
            assessment_criteria=['Uses the diagram.'],
            grounding=GroundingReferences(),
        ),
    )
    response = TaskResponse(
        uuid='response',
        task_uuid=task.uuid,
        answer=FreeAnswer(
            material=Material(
                parts=(TextPart(text='My work'), ImagePart(asset=assets[2]))
            ),
            confidence=Confidence.MEDIUM,
        ),
    )
    return TaskGradingRequest(task=task, response=response), images


def _grade_payload() -> dict[str, object]:
    return {
        'grade': {
            'correctness': 'correct',
            'interpretation': {
                'reasoning_verdict': 'sound',
                'explanation': 'The work uses the supplied diagram.',
                'demonstrated_understandings': ['Reads the diagram.'],
                'gaps': [],
                'possible_misconceptions': [],
                'unresolved_questions': [],
            },
            'review_rating': 'good',
        }
    }


def test_grader_sends_task_and_response_images_as_native_inputs() -> None:
    request, expected_images = _request()
    lm = DummyLM([_grade_payload()])
    backend = dspy.Predict(generation_signature(TaskGradingSignature))
    backend.set_lm(lm)
    module = TaskGradingModule(
        NativePredictor(
            backend,
            TaskGradingSignature,
            budget=TokenBudget(MaterialTokenEstimator(_Counter()), 100000),
            image_budget=ImagePayloadBudget(8, 100000, 100000),
            supports_images=True,
        )
    )
    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        grade = asyncio.run(module.acall(request=request))

    user_message = [
        message
        for message in lm.history[-1]['messages']
        if message['role'] == 'user'
    ][-1]
    content = user_message['content']
    text = '\n'.join(part['text'] for part in content if part['type'] == 'text')
    native_images = [part for part in content if part['type'] == 'image_url']
    received_images = [
        base64.b64decode(part['image_url']['url'].split(',', 1)[1])
        for part in native_images
    ]
    assert received_images == expected_images
    assert all(image.decode('latin1') not in text for image in expected_images)
    assert grade.correctness == 'correct'


def test_complete_request_budget_includes_examples_and_live_task_images() -> (
    None
):
    class MustNotRun(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dispatched = False

        async def aforward(self, **kwargs: object) -> object:
            self.dispatched = True
            raise AssertionError('Oversized request reached inference')

    request, _ = _request()
    examples = task_grading_examples()
    format_request, prompt_images = generation_prompt(
        TaskGradingSignature, examples
    )
    budget = TokenBudget(
        MaterialTokenEstimator(_Counter()),
        token_limit=100000,
        format_request=format_request,
        context_token_limit=1,
        prompt_images=prompt_images,
    )
    backend = MustNotRun()
    module = TaskGradingModule(
        NativePredictor(
            backend,
            TaskGradingSignature,
            budget=budget,
            image_budget=ImagePayloadBudget(8, 100000, 100000),
            supports_images=True,
        )
    )

    with pytest.raises(InputBudgetExceeded):
        asyncio.run(module.acall(request=request))

    assert prompt_images
    assert not backend.dispatched
