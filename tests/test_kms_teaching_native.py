"""Native image projection and complete-request budgets for teaching."""

import asyncio
import base64
from io import BytesIO
from pathlib import Path

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
    project_generation_inputs,
)
from kms.knowledge.models import KnowledgeContext, Proposition
from kms.learning.references import (
    GroundingReferences,
    PropositionTarget,
    SolutionTarget,
)
from kms.learning.session.models import SessionContext
from kms.learning.task.models import Solution, Task
from kms.learning.teaching.generation.examples import (
    teaching_generation_examples,
)
from kms.learning.teaching.generation.inference import (
    TeachingGenerationModule,
    TeachingGenerationSignature,
)
from kms.learning.teaching.generation.models import (
    TeachingGenerationRequest,
    TeachingState,
)
from kms.learning.teaching.generation.node import TeachingGenerationNode
from kms.learning.teaching.models import Teaching
from kms.media.budget import ImagePayloadBudget
from kms.media.models import ImageAsset, ImagePart, Material, TextPart
from kms.source.models.source import Source
from kms.source.models.unit import SourceUnitKind
from kms.source.retrieval.models import SourceUnitMaterial, SourceUnitWindow


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def _png(color: str) -> bytes:
    image = BytesIO()
    Image.new('RGB', (8, 8), color).save(image, format='PNG')
    return image.getvalue()


def _request(
    history_asset: ImageAsset, evidence_asset: ImageAsset
) -> TeachingGenerationRequest:
    history_task = Task(
        understanding='Read the earlier graph.',
        prompt=Material(
            parts=(
                TextPart(text='Earlier graph: '),
                ImagePart(asset=history_asset),
            )
        ),
        target=SolutionTarget(uuid='history-solution'),
        grounding=GroundingReferences(),
        solution=Solution(
            uuid='history-solution',
            material=Material.from_text('Earlier solution.'),
            assessment_criteria=[],
            grounding=GroundingReferences(),
        ),
    )
    proposition = Proposition(
        uuid='function-value',
        name='Function value',
        statement='The filled point determines the function value.',
    )
    window = SourceUnitWindow(
        source=Source(uuid='source', key='current graph'),
        before=[],
        target=SourceUnitMaterial(
            uuid='evidence-unit',
            kind=SourceUnitKind.CONTENT,
            material=Material(
                parts=(
                    TextPart(text='Current graph evidence: '),
                    ImagePart(asset=evidence_asset),
                )
            ),
        ),
        after=[],
    )
    return TeachingGenerationRequest(
        session=SessionContext(
            goal='Distinguish the function value from nearby behavior.',
            targets=['Read a function value from its graph.'],
            steps=[history_task],
            responses=[],
            assessments=[],
        ),
        understanding='Read the filled point as the function value.',
        objective='Explain why the filled point gives f(2).',
        target=PropositionTarget(uuid=proposition.uuid),
        knowledge=KnowledgeContext(
            matched=[proposition], propositions=[proposition], arguments=[]
        ),
        evidence=[window],
    )


def _draft(*, image_position: int) -> dict[str, object]:
    return {
        'teaching': {
            'material': {
                'parts': [
                    {'type': 'image', 'image_position': image_position},
                    {'type': 'text', 'text': 'The filled point gives f(2).'},
                    {'type': 'image', 'image_position': image_position},
                ]
            },
            'grounding': {
                'proposition_uuids': ['function-value'],
                'argument_uuids': [],
                'unit_uuids': ['evidence-unit'],
            },
        }
    }


class _Repository:
    def __init__(self) -> None:
        self.persisted: Teaching | None = None

    async def read_session(self, user_uuid: str, session_uuid: str) -> object:
        assert (user_uuid, session_uuid) == ('user', 'session')
        return object()

    async def append_teaching(
        self, user_uuid: str, session_uuid: str, teaching: Teaching
    ) -> Teaching:
        assert (user_uuid, session_uuid) == ('user', 'session')
        self.persisted = teaching.model_copy(
            update={'uuid': 'persisted-teaching'}
        )
        return self.persisted


def _state(request: TeachingGenerationRequest) -> TeachingState:
    return {'user_uuid': 'user', 'session_uuid': 'session', 'request': request}


def test_teaching_native_catalog_orders_history_before_evidence(
    tmp_path: Path,
) -> None:
    history_png = _png('red')
    evidence_png = _png('blue')
    history_path = tmp_path / 'history.png'
    evidence_path = tmp_path / 'evidence.png'
    history_path.write_bytes(history_png)
    evidence_path.write_bytes(evidence_png)
    history_asset = ImageAsset(('history',), history_path)
    evidence_asset = ImageAsset(('evidence',), evidence_path)
    request = _request(history_asset, evidence_asset)
    lm = DummyLM([_draft(image_position=1)])
    predictor = dspy.Predict(generation_signature(TeachingGenerationSignature))
    predictor.set_lm(lm)
    native = NativePredictor(
        predictor,
        TeachingGenerationSignature,
        budget=TokenBudget(MaterialTokenEstimator(_Counter()), 100000),
        image_budget=ImagePayloadBudget(4, 100000, 100000),
        supports_images=True,
    )
    repository = _Repository()
    node = TeachingGenerationNode(TeachingGenerationModule(native), repository)  # type: ignore[arg-type]

    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        teaching = asyncio.run(node.run(_state(request)))['teaching']

    user_message = next(
        message
        for message in reversed(lm.history[-1]['messages'])
        if message['role'] == 'user'
    )
    content = user_message['content']
    text = '\n'.join(part['text'] for part in content if part['type'] == 'text')
    model_images = [part for part in content if part['type'] == 'image_url']
    assert len(model_images) == 2
    received_pngs = [
        base64.b64decode(part['image_url']['url'].split(',', 1)[1])
        for part in model_images
    ]
    assert received_pngs == [history_png, evidence_png]
    assert str(history_asset.payload) not in text
    assert str(evidence_asset.payload) not in text

    assert repository.persisted is teaching
    assert teaching.uuid == 'persisted-teaching'
    assert teaching.material.images == (evidence_asset, evidence_asset)
    assert [part.type for part in teaching.material.parts] == [
        'image',
        'text',
        'image',
    ]


def test_teaching_prompt_examples_include_native_diagram() -> None:
    examples = teaching_generation_examples()
    prompt, prompt_images = generation_prompt(
        TeachingGenerationSignature, examples
    )
    rendered = prompt(
        project_generation_inputs({'request': examples[1]['request']}).inputs
    )

    assert len(examples) == 3
    assert len(prompt_images) == 1
    assert 'open point' in rendered.lower()
    assert str(prompt_images[0].payload) not in rendered


def test_complete_teaching_budget_counts_demo_and_live_native_images() -> None:
    class MustNotRun(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dispatched = False

        async def aforward(self, **kwargs: object) -> object:
            self.dispatched = True
            raise AssertionError('Oversized request reached the model')

    examples = teaching_generation_examples()
    request = _request(
        ImageAsset(('history',), _png('red')),
        ImageAsset(('evidence',), _png('blue')),
    )
    format_request, prompt_images = generation_prompt(
        TeachingGenerationSignature, examples
    )
    budget = TokenBudget(
        MaterialTokenEstimator(_Counter()),
        token_limit=100000,
        format_request=format_request,
        context_token_limit=1,
        prompt_images=prompt_images,
    )
    backend = MustNotRun()
    native = NativePredictor(
        backend,
        TeachingGenerationSignature,
        budget=budget,
        image_budget=ImagePayloadBudget(8, 1000000, 1000000),
        supports_images=True,
    )

    with pytest.raises(InputBudgetExceeded):
        asyncio.run(TeachingGenerationModule(native).acall(request=request))

    assert len(prompt_images) == 1
    assert not backend.dispatched
