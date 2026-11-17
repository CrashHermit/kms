"""Source-grounded extraction through the native multimodal inference boundary."""

import asyncio
import base64
from pathlib import Path

import dspy
from dspy.utils import DummyLM
from PIL import Image

from kms.config.inference import VisionInferenceSettings
from kms.config.settings import Settings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import NativePredictor, generation_signature
from kms.inference.predictors import build_token_budget
from kms.knowledge.candidates import (
    ExistingPropositionReference,
    NewPropositionReference,
)
from kms.knowledge.extraction.inference import (
    UnitExtractionModule,
    UnitExtractionSignature,
)
from kms.knowledge.extraction.node import UnitKnowledgeExtractionNode
from kms.knowledge.models import KnowledgeContext, Proposition
from kms.media.budget import ImagePayloadBudget
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.retrieval.models import SourceUnitRead
from kms.source.retrieval.read import SourceReader


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Tokenizers:
    def for_inference(
        self, inference: VisionInferenceSettings
    ) -> MaterialTokenEstimator:
        return MaterialTokenEstimator(_Counter())


class _Repository:
    def __init__(self, units: list[SourceUnit]) -> None:
        self.source = Source(uuid='source-id', key='source')
        self.units = units

    async def read_units(
        self,
        user_uuid: str,
        unit_uuids: list[str],
        *,
        before_units: int = 0,
        after_units: int = 0,
    ) -> list[SourceUnitRead]:
        if user_uuid != 'owner':
            return []
        for position, unit in enumerate(self.units):
            if unit.uuid == unit_uuids[0]:
                return [
                    SourceUnitRead(source=self.source, unit=neighbor)
                    for neighbor in self.units[
                        max(0, position - before_units) : position
                        + after_units
                        + 1
                    ]
                ]
        return []


def _unit(uuid: str, text: str, image: Path | None = None) -> SourceUnit:
    blocks = [SourceBlock(block_type=BlockType.PARAGRAPH, content=text)]
    if image is not None:
        blocks.append(
            SourceBlock(block_type=BlockType.IMAGE, image_path=str(image))
        )
    return SourceUnit(uuid=uuid, kind=SourceUnitKind.CONTENT, blocks=blocks)


def test_source_window_becomes_native_evidence_and_returns_grounded_candidates(
    tmp_path: Path,
) -> None:
    image = tmp_path / 'diagram.png'
    Image.new('RGB', (8, 8), 'red').save(image)
    units = [
        _unit('before', 'Context before', image),
        _unit('target', 'Target explanation'),
        _unit('after', 'Context after'),
    ]
    existing = Proposition(
        uuid='existing-1', name='Slope', statement='Slope is rise over run.'
    )
    response = {
        'new_propositions': [
            {
                'id': 'p1',
                'name': 'Derivative',
                'statement': 'A derivative is a slope.',
            }
        ],
        'existing_proposition_uuids': ['existing-1'],
        'new_arguments': [
            {
                'premises': [{'kind': 'existing', 'uuid': 'existing-1'}],
                'conclusion': {'kind': 'new', 'id': 'p1'},
            }
        ],
        'existing_argument_uuids': [],
    }
    backend = dspy.Predict(generation_signature(UnitExtractionSignature))
    lm = DummyLM([{'candidates': response}])
    backend.set_lm(lm)
    settings = Settings(_env_file=None)
    budget = build_token_budget(
        settings,
        _Tokenizers(),
        settings.knowledge_extraction.inference,
        input_token_budget=100000,
        safety_margin_tokens=256,
        signature=UnitExtractionSignature,
    )
    image_budget = ImagePayloadBudget(4, 100000, 100000)
    module = UnitExtractionModule(
        NativePredictor(
            backend,
            UnitExtractionSignature,
            budget=budget,
            image_budget=image_budget,
            supports_images=True,
        )
    )
    extractor = UnitKnowledgeExtractionNode(
        SourceReader(_Repository(units)),
        module,
        budget,
        image_budget,
        before_tokens=100,
        after_tokens=100,
    )
    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        state = {
            'user_uuid': 'owner',
            'unit_uuid': 'target',
            'understanding': 'Understand derivatives',
            'context': KnowledgeContext(
                matched=[], propositions=[existing], arguments=[]
            ),
        }
        original_context = state['context'].model_copy(deep=True)
        result = asyncio.run(extractor.run(state))['pending']
    assert result is not None
    assert result.source_uuid == 'source-id'
    assert result.target_unit_uuid == 'target'
    assert state['context'] == original_context
    assert result.candidates.new_propositions[0].id == 'p1'
    assert result.candidates.existing_proposition_uuids == ['existing-1']
    argument = result.candidates.new_arguments[0]
    assert argument.premises == [
        ExistingPropositionReference(uuid='existing-1')
    ]
    assert argument.conclusion == NewPropositionReference(id='p1')
    [entry] = lm.history
    message = [item for item in entry['messages'] if item['role'] == 'user'][-1]
    image_parts = [
        part for part in message['content'] if part['type'] == 'image_url'
    ]
    assert len(image_parts) == 1
    assert (
        base64.b64decode(image_parts[0]['image_url']['url'].split(',', 1)[1])
        == image.read_bytes()
    )
    text = '\n'.join(
        part['text'] for part in message['content'] if part['type'] == 'text'
    )
    assert 'Understand derivatives' in text
    assert (
        text.index('Context before')
        < text.index('Target explanation')
        < text.index('Context after')
    )
    assert 'Slope is rise over run.' in text
    assert str(image) not in text


def test_unowned_unit_returns_no_result_without_inference() -> None:
    budget = TokenBudget(MaterialTokenEstimator(_Counter()), 100000)
    image_budget = ImagePayloadBudget(4, 100000, 100000)
    extractor = UnitKnowledgeExtractionNode(
        SourceReader(_Repository([_unit('target', 'passage')])),
        UnitExtractionModule(dspy.Predict(UnitExtractionSignature)),
        budget,
        image_budget,
        before_tokens=100,
        after_tokens=100,
    )
    assert asyncio.run(
        extractor.run(
            {
                'user_uuid': 'other',
                'unit_uuid': 'target',
                'understanding': 'Understand',
                'context': KnowledgeContext(
                    matched=[], propositions=[], arguments=[]
                ),
            }
        )
    ) == {'pending': None}
