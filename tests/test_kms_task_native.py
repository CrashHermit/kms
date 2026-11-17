"""Native image projection and complete-request budgets for task generation."""

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
from kms.knowledge.models import KnowledgeContext
from kms.learning.references import GroundingReferences, SolutionTarget
from kms.learning.session.models import SessionContext
from kms.learning.task.generation.examples import task_generation_examples
from kms.learning.task.generation.inference import (
    TaskGenerationModule,
    TaskGenerationSignature,
)
from kms.learning.task.generation.models import TaskGenerationRequest, TaskState
from kms.learning.task.generation.node import TaskGenerationNode
from kms.learning.task.models import Solution, Task
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
) -> TaskGenerationRequest:
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
    window = SourceUnitWindow(
        source=Source(uuid='source', key='current graph'),
        before=[],
        target=SourceUnitMaterial(
            uuid='evidence-unit',
            kind=SourceUnitKind.CONTENT,
            material=Material(
                parts=(
                    TextPart(text='Current source evidence: '),
                    ImagePart(asset=evidence_asset),
                )
            ),
        ),
        after=[],
    )
    return TaskGenerationRequest(
        session=SessionContext(
            goal='Distinguish function values from limits.',
            targets=['Read a function value from its graph.'],
            steps=[history_task],
            responses=[],
            assessments=[],
        ),
        understanding='Read the filled point as the function value.',
        knowledge=KnowledgeContext(matched=[], propositions=[], arguments=[]),
        evidence=[window],
    )


def _draft(*, image_position: int) -> dict[str, object]:
    material = {
        'parts': [
            {'type': 'image', 'image_position': image_position},
            {'type': 'text', 'text': 'Explain what the graph shows.'},
            {'type': 'image', 'image_position': image_position},
        ]
    }
    return {
        'task': {
            'prompt': material,
            'grounding': {
                'proposition_uuids': [],
                'argument_uuids': [],
                'unit_uuids': ['evidence-unit'],
            },
            'solution': {
                'material': material,
                'assessment_criteria': ['Identify the filled point.'],
                'grounding': {
                    'proposition_uuids': [],
                    'argument_uuids': [],
                    'unit_uuids': ['evidence-unit'],
                },
            },
        }
    }


class _Repository:
    def __init__(self) -> None:
        self.appended: Task | None = None
        self.persisted: Task | None = None

    async def read_session(self, user_uuid: str, session_uuid: str) -> object:
        assert (user_uuid, session_uuid) == ('user', 'session')
        return object()

    async def append_task(
        self, user_uuid: str, session_uuid: str, task: Task
    ) -> Task:
        assert (user_uuid, session_uuid) == ('user', 'session')
        self.appended = task
        self.persisted = task.model_copy(update={'uuid': 'persisted-task'})
        return self.persisted


def _state(request: TaskGenerationRequest) -> TaskState:
    return {'user_uuid': 'user', 'session_uuid': 'session', 'request': request}


def test_task_node_projects_catalog_and_resolves_prompt_and_solution_images() -> (
    None
):
    history_png = _png('red')
    evidence_png = _png('blue')
    history_asset = ImageAsset(('history',), history_png)
    evidence_asset = ImageAsset(('evidence',), evidence_png)
    request = _request(history_asset, evidence_asset)
    lm = DummyLM([_draft(image_position=1)])
    predictor = dspy.Predict(generation_signature(TaskGenerationSignature))
    predictor.set_lm(lm)
    native = NativePredictor(
        predictor,
        TaskGenerationSignature,
        budget=TokenBudget(MaterialTokenEstimator(_Counter()), 100000),
        image_budget=ImagePayloadBudget(4, 100000, 100000),
        supports_images=True,
    )
    repository = _Repository()
    node = TaskGenerationNode(TaskGenerationModule(native), repository)  # type: ignore[arg-type]

    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        task = asyncio.run(node.run(_state(request)))['task']

    user_message = next(
        message
        for message in reversed(lm.history[-1]['messages'])
        if message['role'] == 'user'
    )
    content = user_message['content']
    text = '\n'.join(part['text'] for part in content if part['type'] == 'text')
    model_images = [part for part in content if part['type'] == 'image_url']
    assert len(model_images) == 2
    received_pngs = {
        base64.b64decode(part['image_url']['url'].split(',', 1)[1])
        for part in model_images
    }
    assert received_pngs == {history_png, evidence_png}
    assert repr(history_png) not in text
    assert repr(evidence_png) not in text
    assert str(history_asset.payload) not in text
    assert str(evidence_asset.payload) not in text

    assert repository.persisted is task
    assert task is not repository.appended
    assert task.uuid == 'persisted-task'
    assert task.target == SolutionTarget(uuid=task.solution.uuid)
    assert task.prompt.images == (evidence_asset, evidence_asset)
    assert task.solution.material.images == (evidence_asset, evidence_asset)
    assert [part.type for part in task.prompt.parts] == [
        'image',
        'text',
        'image',
    ]


def test_complete_request_budget_counts_demo_and_live_native_images() -> None:
    class MustNotRun(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dispatched = False

        async def aforward(self, **kwargs: object) -> object:
            self.dispatched = True
            raise AssertionError('Oversized request reached the model')

    examples = task_generation_examples()
    request = _request(
        ImageAsset(('history',), _png('red')),
        ImageAsset(('evidence',), _png('blue')),
    )
    format_request, prompt_images = generation_prompt(
        TaskGenerationSignature, examples
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
        TaskGenerationSignature,
        budget=budget,
        image_budget=ImagePayloadBudget(8, 1000000, 1000000),
        supports_images=True,
    )
    module = TaskGenerationModule(native)

    with pytest.raises(InputBudgetExceeded):
        asyncio.run(module.acall(request=request))

    assert len(prompt_images) == 1
    assert not backend.dispatched
