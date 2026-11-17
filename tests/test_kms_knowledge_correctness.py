"""Correctness judgments with native source evidence."""

import asyncio
import base64
from io import BytesIO

import dspy
import pytest
from dspy.utils import DummyLM
from PIL import Image

from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import NativePredictor, generation_signature
from kms.knowledge.correctness.inference import (
    CorrectnessModule,
    CorrectnessSignature,
)
from kms.knowledge.correctness.models import (
    ArgumentToCheck,
    CorrectnessRequest,
    PropositionToCheck,
)
from kms.knowledge.models import Argument, Proposition
from kms.media.budget import ImagePayloadBudget
from kms.media.models import ImageAsset, ImagePart, Material, TextPart
from kms.source.models.source import Source
from kms.source.models.unit import SourceUnitKind
from kms.source.retrieval.models import SourceUnitMaterial, SourceUnitWindow


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def _evidence() -> SourceUnitWindow:
    image = BytesIO()
    Image.new('RGB', (4, 4), 'red').save(image, format='PNG')
    return SourceUnitWindow(
        source=Source(uuid='source', key='calculus'),
        before=[],
        target=SourceUnitMaterial(
            uuid='unit',
            kind=SourceUnitKind.CONTENT,
            material=Material(
                parts=(
                    TextPart(text='Travel time T = 400/s for positive speeds.'),
                    ImagePart(asset=ImageAsset(('image',), image.getvalue())),
                )
            ),
        ),
        after=[],
    )


def test_proposition_and_argument_verdicts_include_native_evidence() -> None:
    proposition = Proposition(
        uuid='p', name='Travel', statement='The formula applies at every speed.'
    )
    conclusion = Proposition(
        uuid='c', name='Zero', statement='The formula applies at zero speed.'
    )
    argument = Argument(uuid='a', premise_uuids=['p'], conclusion_uuid='c')
    evidence = _evidence()
    lm = DummyLM(
        [
            {
                'judgment': {
                    'verdict': 'contradicted',
                    'explanation': 'The formula excludes zero speed.',
                }
            },
            {
                'judgment': {
                    'verdict': 'insufficient_evidence',
                    'explanation': 'The source does not support the premise.',
                }
            },
            {
                'judgment': {
                    'verdict': 'supported',
                    'explanation': 'The supplied evidence supports the proposition.',
                }
            },
        ]
    )
    backend = dspy.Predict(generation_signature(CorrectnessSignature))
    backend.set_lm(lm)
    module = CorrectnessModule(
        NativePredictor(
            backend,
            CorrectnessSignature,
            budget=TokenBudget(MaterialTokenEstimator(_Counter()), 100000),
            image_budget=ImagePayloadBudget(4, 100000, 100000),
            supports_images=True,
        )
    )
    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        first = asyncio.run(
            module.acall(
                request=CorrectnessRequest(
                    item=PropositionToCheck(proposition=proposition),
                    evidence=[evidence],
                )
            )
        )
        second = asyncio.run(
            module.acall(
                request=CorrectnessRequest(
                    item=ArgumentToCheck(
                        argument=argument,
                        propositions=[proposition, conclusion],
                    ),
                    evidence=[evidence],
                )
            )
        )
        third = asyncio.run(
            module.acall(
                request=CorrectnessRequest(
                    item=PropositionToCheck(proposition=proposition),
                    evidence=[evidence],
                )
            )
        )
    assert first.verdict == 'contradicted'
    assert second.verdict == 'insufficient_evidence'
    assert third.verdict == 'supported'
    message = [
        item for item in lm.history[1]['messages'] if item['role'] == 'user'
    ][-1]
    text = '\n'.join(
        part['text'] for part in message['content'] if part['type'] == 'text'
    )
    assert 'positive speeds' in text
    assert 'applies at zero speed' in text
    images = [
        part for part in message['content'] if part['type'] == 'image_url'
    ]
    assert len(images) == 1
    assert (
        base64.b64decode(images[0]['image_url']['url'].split(',', 1)[1])
        == evidence.target.material.images[0].payload
    )


def test_oversized_evidence_is_not_sent_to_predictor() -> None:
    class MustNotRun(dspy.Module):
        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError('Oversized request reached the model')

    module = CorrectnessModule(
        NativePredictor(
            MustNotRun(),
            CorrectnessSignature,
            budget=TokenBudget(MaterialTokenEstimator(_Counter()), 1),
            image_budget=ImagePayloadBudget(4, 100000, 100000),
            supports_images=True,
        )
    )
    request = CorrectnessRequest(
        item=PropositionToCheck(
            proposition=Proposition(name='Claim', statement='A claim')
        ),
        evidence=[_evidence()],
    )
    with pytest.raises(InputBudgetExceeded):
        asyncio.run(module.acall(request=request))
