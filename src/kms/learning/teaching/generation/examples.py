"""Deterministic examples for conceptual, diagram, and solution teaching."""

from datetime import UTC, datetime

import dspy

from kms.knowledge.models import KnowledgeContext, Proposition
from kms.learning.examples.image_assets import learning_image
from kms.learning.materials import ImageReference, MaterialDraft
from kms.learning.references import (
    GroundingReferences,
    PropositionTarget,
    SolutionTarget,
)
from kms.learning.session.models import SessionContext
from kms.learning.task.models import Solution, Task
from kms.learning.teaching.generation.models import (
    TeachingDraft,
    TeachingGenerationRequest,
)
from kms.media.models import ImagePart, Material, TextPart
from kms.source.models.source import Source
from kms.source.models.unit import SourceUnitKind
from kms.source.retrieval.models import SourceUnitMaterial, SourceUnitWindow

_EXAMPLE_TIME = datetime(2020, 1, 1, tzinfo=UTC)


def _proposition(uuid: str, name: str, statement: str) -> Proposition:
    return Proposition(
        uuid=uuid,
        created_at=_EXAMPLE_TIME,
        updated_at=_EXAMPLE_TIME,
        name=name,
        statement=statement,
    )


def _request(
    understanding: str,
    objective: str,
    proposition: Proposition,
    *,
    evidence: list[SourceUnitWindow] | None = None,
    target: PropositionTarget | SolutionTarget | None = None,
    solution_task: Task | None = None,
) -> TeachingGenerationRequest:
    return TeachingGenerationRequest(
        session=SessionContext(
            goal='Understand the supplied mathematical ideas.',
            targets=[understanding],
            steps=[],
            responses=[],
            assessments=[],
        ),
        understanding=understanding,
        objective=objective,
        target=target or PropositionTarget(uuid=proposition.uuid),
        knowledge=KnowledgeContext(
            matched=[proposition], propositions=[proposition], arguments=[]
        ),
        evidence=evidence or [],
        solution_task=solution_task,
    )


def teaching_generation_examples() -> list[dspy.Example]:
    """Demonstrate useful conceptual, visual, and worked-solution teaching."""
    limit = _proposition(
        '00000000-0000-4000-8000-000000000101',
        'Limit and function value',
        'A limit describes nearby values, while a function value is determined by the filled point at the input.',
    )
    function_value = _proposition(
        '00000000-0000-4000-8000-000000000102',
        'Function value from graph',
        'The filled point at x=2 determines f(2), even if the graph approaches a different open point there.',
    )
    equation = _proposition(
        '00000000-0000-4000-8000-000000000103',
        'Solving a linear equation',
        'Applying the same operation to both sides preserves equality and the equation solution.',
    )
    image = learning_image('function_graph.png')
    evidence = SourceUnitWindow(
        source=Source(
            uuid='00000000-0000-4000-8000-000000000104',
            created_at=_EXAMPLE_TIME,
            updated_at=_EXAMPLE_TIME,
            key='function graph teaching example',
        ),
        before=[],
        target=SourceUnitMaterial(
            uuid='00000000-0000-4000-8000-000000000105',
            kind=SourceUnitKind.CONTENT,
            material=Material(
                parts=(
                    TextPart(
                        text='At x=2, the open point is at 15 and the filled point is at 5. '
                    ),
                    ImagePart(asset=image),
                )
            ),
        ),
        after=[],
    )
    solution = Solution(
        uuid='00000000-0000-4000-8000-000000000106',
        created_at=_EXAMPLE_TIME,
        updated_at=_EXAMPLE_TIME,
        material=Material.from_text(
            'Subtract 3 from both sides, then divide by 2 to get x=2.'
        ),
        assessment_criteria=['Obtains x=2 using valid equivalent operations.'],
        grounding=GroundingReferences(proposition_uuids=[equation.uuid]),
    )
    solution_task = Task(
        uuid='00000000-0000-4000-8000-000000000107',
        created_at=_EXAMPLE_TIME,
        updated_at=_EXAMPLE_TIME,
        understanding='Solve 2x+3=7.',
        prompt=Material.from_text('Solve 2x+3=7.'),
        target=SolutionTarget(uuid=solution.uuid),
        grounding=GroundingReferences(proposition_uuids=[equation.uuid]),
        solution=solution,
    )

    return [
        dspy.Example(
            request=_request(
                'Distinguish a limit from the function value.',
                'Connect nearby behavior to the filled point that defines the value.',
                limit,
            ),
            teaching=TeachingDraft(
                material=MaterialDraft(
                    parts=(
                        TextPart(
                            text='A limit follows what nearby points approach. '
                        ),
                        TextPart(
                            text='The function value at x itself is read from the filled point, so the two can differ.'
                        ),
                    )
                ),
                grounding=GroundingReferences(proposition_uuids=[limit.uuid]),
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=_request(
                'Read f(2) from the supplied graph.',
                'Repair the confusion between the open and filled points.',
                function_value,
                evidence=[evidence],
            ),
            teaching=TeachingDraft(
                material=MaterialDraft(
                    parts=(
                        TextPart(
                            text='At x=2, follow the vertical line: the open point shows the approaching value, while the filled point gives f(2). '
                        ),
                        ImageReference(image_position=0),
                        TextPart(text='Here the filled point is at 5.'),
                    )
                ),
                grounding=GroundingReferences(
                    proposition_uuids=[function_value.uuid],
                    unit_uuids=[evidence.target.uuid],
                ),
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=_request(
                'Solve 2x+3=7.',
                'Explain the valid solution to the same equation.',
                equation,
                target=SolutionTarget(uuid=solution.uuid),
                solution_task=solution_task,
            ),
            teaching=TeachingDraft(
                material=MaterialDraft(
                    parts=(
                        TextPart(
                            text='For 2x+3=7, subtract 3 from both sides to obtain 2x=4. '
                        ),
                        TextPart(
                            text='Dividing both sides by 2 gives x=2; each step keeps the two sides equal.'
                        ),
                    )
                ),
                grounding=GroundingReferences(
                    proposition_uuids=[equation.uuid]
                ),
            ),
        ).with_inputs('request'),
    ]
