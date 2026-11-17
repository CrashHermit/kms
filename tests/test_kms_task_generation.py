"""Behavioral checks for grounded task construction and persistence."""

import asyncio

import pytest

from kms.knowledge.models import KnowledgeContext, Proposition
from kms.learning.materials import ImageReference, MaterialDraft
from kms.learning.references import (
    GroundingReferences,
    PropositionTarget,
    SolutionTarget,
)
from kms.learning.session.models import SessionContext
from kms.learning.task.generation.models import (
    NewSolutionTarget,
    SolutionDraft,
    TaskDraft,
    TaskGenerationRequest,
    TaskState,
)
from kms.learning.task.generation.node import TaskGenerationNode
from kms.learning.task.models import PresentedTask, Solution, Task
from kms.media.models import ImageAsset, ImagePart, Material, TextPart
from kms.source.models.source import Source
from kms.source.models.unit import SourceUnitKind
from kms.source.retrieval.models import SourceUnitMaterial, SourceUnitWindow


class _Module:
    def __init__(self, events: list[str], draft: TaskDraft) -> None:
        self.events = events
        self.draft = draft

    async def acall(self, *, request: TaskGenerationRequest) -> TaskDraft:
        self.events.append('inference')
        return self.draft


class _Repository:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.persisted: Task | None = None
        self.fail = fail

    async def read_session(self, user_uuid: str, session_uuid: str) -> object:
        self.events.append('read')
        return object()

    async def append_task(
        self, user_uuid: str, session_uuid: str, task: Task
    ) -> Task:
        self.events.append('persist')
        if self.fail:
            raise RuntimeError('write failed')
        self.persisted = task.model_copy(update={'uuid': 'persisted-task'})
        return self.persisted


def _request(
    asset: ImageAsset, target: PropositionTarget | None = None
) -> TaskGenerationRequest:
    evidence = SourceUnitWindow(
        source=Source(uuid='source', key='diagram'),
        before=[],
        target=SourceUnitMaterial(
            uuid='unit',
            kind=SourceUnitKind.CONTENT,
            material=Material(
                parts=(
                    TextPart(text='Graph evidence: '),
                    ImagePart(asset=asset),
                )
            ),
        ),
        after=[],
    )
    propositions = (
        [
            Proposition(
                uuid=target.uuid,
                name='Function value',
                statement='The filled point determines f(2).',
            )
        ]
        if target is not None
        else []
    )
    return TaskGenerationRequest(
        session=SessionContext(
            goal='Read a value from a function graph.',
            targets=['Identify the function value.'],
            steps=[],
            responses=[],
            assessments=[],
        ),
        understanding='Read the filled point as f(2).',
        knowledge=KnowledgeContext(
            matched=propositions,
            propositions=propositions,
            arguments=[],
        ),
        evidence=[evidence],
        target=target or NewSolutionTarget(),
    )


def _draft(image_position: int = 0) -> TaskDraft:
    return TaskDraft(
        prompt=MaterialDraft(
            parts=(
                TextPart(text='Use this graph: '),
                ImageReference(image_position=image_position),
            )
        ),
        grounding=GroundingReferences(unit_uuids=['unit']),
        solution=SolutionDraft(
            material=MaterialDraft(
                parts=(
                    ImageReference(image_position=image_position),
                    TextPart(text='The filled point gives f(2)=5.'),
                    ImageReference(image_position=image_position),
                )
            ),
            assessment_criteria=['Identifies the filled point.'],
            grounding=GroundingReferences(unit_uuids=['unit']),
        ),
    )


def _state(request: TaskGenerationRequest) -> TaskState:
    return {'user_uuid': 'user', 'session_uuid': 'session', 'request': request}


def test_task_node_resolves_repeated_images_and_returns_persisted_hidden_task() -> (
    None
):
    asset = ImageAsset(('evidence',), b'evidence-image')
    events: list[str] = []
    repository = _Repository(events)
    state = _state(_request(asset))

    result = asyncio.run(
        TaskGenerationNode(_Module(events, _draft()), repository).run(state)
    )

    assert events == ['read', 'inference', 'persist']
    assert result['task'] is repository.persisted
    task = result['task']
    assert task.uuid == 'persisted-task'
    assert task.target == SolutionTarget(uuid=task.solution.uuid)
    assert [part.type for part in task.prompt.parts] == ['text', 'image']
    assert task.prompt.images == (asset,)
    assert [part.type for part in task.solution.material.parts] == [
        'image',
        'text',
        'image',
    ]
    assert task.solution.material.images == (asset, asset)
    assert task.solution.assessment_criteria == ['Identifies the filled point.']

    presented = PresentedTask(task_uuid=task.uuid, prompt=task.prompt)
    serialized = presented.model_dump(mode='json')
    assert set(serialized) == {'task_uuid', 'prompt'}
    assert 'solution' not in str(serialized)
    assert task.solution.material.text == 'The filled point gives f(2)=5.'


def test_task_node_preserves_the_caller_selected_assessment_target() -> None:
    asset = ImageAsset(('evidence',), b'evidence-image')
    selected = PropositionTarget(uuid='selected-proposition')
    repository = _Repository([])

    result = asyncio.run(
        TaskGenerationNode(_Module([], _draft()), repository).run(
            _state(_request(asset, selected))
        )
    )

    assert result['task'].target == selected


def test_new_solution_is_not_substituted_for_an_existing_solution_target() -> (
    None
):
    asset = ImageAsset(('evidence',), b'evidence-image')
    original_solution = Solution(
        uuid='existing-solution',
        material=Material.from_text('The original answer.'),
        assessment_criteria=['The original criterion.'],
        grounding=GroundingReferences(),
    )
    original_task = Task(
        uuid='original-task',
        understanding='Solve the original problem.',
        prompt=Material.from_text('Original problem.'),
        target=SolutionTarget(uuid=original_solution.uuid),
        grounding=GroundingReferences(),
        solution=original_solution,
    )
    request = _request(asset).model_copy(
        update={
            'target': SolutionTarget(uuid=original_solution.uuid),
            'solution_task': original_task,
        }
    )

    task = asyncio.run(
        TaskGenerationNode(_Module([], _draft()), _Repository([])).run(
            _state(request)
        )
    )['task']

    assert task.target == SolutionTarget(uuid=original_solution.uuid)
    assert task.solution.uuid != original_solution.uuid


def test_task_node_does_not_infer_for_foreign_session() -> None:
    class ForeignRepository(_Repository):
        async def read_session(
            self, user_uuid: str, session_uuid: str
        ) -> object:
            self.events.append('read')
            raise LookupError('foreign session')

    events: list[str] = []
    repository = ForeignRepository(events)
    node = TaskGenerationNode(_Module(events, _draft()), repository)

    with pytest.raises(LookupError, match='foreign session'):
        asyncio.run(
            node.run(_state(_request(ImageAsset(('evidence',), b'image'))))
        )

    assert events == ['read']


def test_task_node_propagates_persistence_failure() -> None:
    events: list[str] = []
    repository = _Repository(events, fail=True)
    node = TaskGenerationNode(_Module(events, _draft()), repository)

    with pytest.raises(RuntimeError, match='write failed'):
        asyncio.run(
            node.run(_state(_request(ImageAsset(('evidence',), b'image'))))
        )

    assert events == ['read', 'inference', 'persist']
    assert repository.persisted is None
