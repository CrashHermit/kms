"""Deterministic examples for conceptual, diagram, and procedural tasks."""

from datetime import UTC, datetime

import dspy

from kms.knowledge.models import Argument, KnowledgeContext, Proposition
from kms.learning.examples.image_assets import learning_image
from kms.learning.materials import ImageReference, MaterialDraft
from kms.learning.references import (
    ArgumentTarget,
    GroundingReferences,
    LearningTarget,
    PropositionTarget,
    SolutionTarget,
)
from kms.learning.session.models import SessionContext
from kms.learning.task.generation.models import (
    NewSolutionTarget,
    SolutionDraft,
    TaskDraft,
    TaskGenerationRequest,
)
from kms.learning.task.models import Solution, Task
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
    propositions: list[Proposition],
    *,
    evidence: list[SourceUnitWindow] | None = None,
    target: LearningTarget | NewSolutionTarget | None = None,
    arguments: list[Argument] | None = None,
    solution_task: Task | None = None,
) -> TaskGenerationRequest:
    arguments = arguments or []
    return TaskGenerationRequest(
        session=SessionContext(
            goal='Practice applying the supplied mathematical ideas.',
            targets=[understanding],
            steps=[],
            responses=[],
            assessments=[],
        ),
        understanding=understanding,
        knowledge=KnowledgeContext(
            matched=propositions,
            propositions=propositions,
            arguments=arguments,
        ),
        evidence=evidence or [],
        target=target or NewSolutionTarget(),
        solution_task=solution_task,
    )


def task_generation_examples() -> list[dspy.Example]:
    """Show complete free-response tasks, including image and procedure use."""
    limit = _proposition(
        '00000000-0000-4000-8000-000000000001',
        'Nearby values determine a limit',
        'A limit describes the value approached by nearby function values, not necessarily the function value at the point.',
    )
    value = _proposition(
        '00000000-0000-4000-8000-000000000002',
        'The filled point gives the function value',
        'The filled point at x=2 has y-coordinate 5, so f(2)=5.',
    )
    graph_limit = _proposition(
        '00000000-0000-4000-8000-000000000003',
        'The open point shows the limit',
        'The open point at (2,15) shows that the limit of f(x) as x approaches 2 is 15.',
    )
    inverse = _proposition(
        '00000000-0000-4000-8000-000000000004',
        'Undo operations in reverse order',
        'To solve a linear equation, apply inverse operations to both sides in reverse order.',
    )
    equation = _proposition(
        '00000000-0000-4000-8000-000000000005',
        'Linear equation example',
        'The equation 3x+4=19 has solution x=5.',
    )

    image = learning_image('function_graph.png')
    graph_window = SourceUnitWindow(
        source=Source(
            uuid='00000000-0000-4000-8000-000000000007',
            created_at=_EXAMPLE_TIME,
            updated_at=_EXAMPLE_TIME,
            key='task-example-function-graph',
        ),
        before=[],
        target=SourceUnitMaterial(
            uuid='00000000-0000-4000-8000-000000000008',
            kind=SourceUnitKind.CONTENT,
            material=Material(
                parts=(
                    TextPart(text='Function graph near x=2. '),
                    ImagePart(asset=image),
                )
            ),
        ),
        after=[],
    )

    conceptual_request = _request(
        'Distinguish the value of a function at a point from its limit.',
        [limit],
        target=PropositionTarget(uuid=limit.uuid),
    )
    diagram_request = _request(
        'Read the function value from a graph with an open and filled point.',
        [value, graph_limit],
        evidence=[graph_window],
        target=PropositionTarget(uuid=value.uuid),
    )
    procedure_request = _request(
        'Solve a linear equation by applying inverse operations.',
        [inverse, equation],
        target=NewSolutionTarget(),
    )
    argument_premise = _proposition(
        '00000000-0000-4000-8000-000000000009',
        'Equal additions preserve equality',
        'Adding the same quantity to equal quantities preserves equality.',
    )
    argument_conclusion = _proposition(
        '00000000-0000-4000-8000-00000000000a',
        'Additive isolation',
        'Adding the same quantity to both sides isolates an equation while preserving its solutions.',
    )
    argument = Argument(
        uuid='00000000-0000-4000-8000-00000000000b',
        premise_uuids=[argument_premise.uuid],
        conclusion_uuid=argument_conclusion.uuid,
    )
    argument_request = _request(
        'Explain why adding the same quantity to both sides preserves solutions.',
        [argument_premise, argument_conclusion],
        target=ArgumentTarget(uuid=argument.uuid),
        arguments=[argument],
    )

    original_solution = Solution(
        uuid='00000000-0000-4000-8000-00000000000c',
        created_at=_EXAMPLE_TIME,
        updated_at=_EXAMPLE_TIME,
        material=Material.from_text(
            'Subtract 3 from both sides to get 2x=4; divide by 2 to get x=2.'
        ),
        assessment_criteria=['Finds x=2.'],
        grounding=GroundingReferences(),
    )
    original_task = Task(
        uuid='00000000-0000-4000-8000-00000000000d',
        created_at=_EXAMPLE_TIME,
        updated_at=_EXAMPLE_TIME,
        understanding='Solve 2x+3=7 using equality-preserving operations.',
        prompt=Material.from_text('Solve 2x+3=7.'),
        target=SolutionTarget(uuid=original_solution.uuid),
        grounding=GroundingReferences(),
        solution=original_solution,
    )
    solution_request = _request(
        original_task.understanding,
        [],
        target=SolutionTarget(uuid=original_solution.uuid),
        solution_task=original_task,
    )
    return [
        dspy.Example(
            request=conceptual_request,
            task=TaskDraft(
                prompt=MaterialDraft(
                    parts=(
                        TextPart(
                            text='Suppose h(x) approaches 12 as x approaches 4, while h(4)=7. What is the limit of h(x) as x approaches 4?'
                        ),
                    )
                ),
                grounding=GroundingReferences(proposition_uuids=[limit.uuid]),
                solution=SolutionDraft(
                    material=MaterialDraft(
                        parts=(
                            TextPart(
                                text='The limit is 12 because nearby values approach 12. The value h(4)=7 does not determine that limit.'
                            ),
                        )
                    ),
                    assessment_criteria=[
                        'Gives 12 as the limit, the value approached nearby.',
                        'Accepts an equivalent explanation distinguishing the limit from h(4).',
                    ],
                    grounding=GroundingReferences(
                        proposition_uuids=[limit.uuid]
                    ),
                ),
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=diagram_request,
            task=TaskDraft(
                prompt=MaterialDraft(
                    parts=(
                        TextPart(text='Use the graph to state f(2). '),
                        ImageReference(image_position=0),
                    )
                ),
                grounding=GroundingReferences(
                    proposition_uuids=[value.uuid, graph_limit.uuid],
                    unit_uuids=[graph_window.target.uuid],
                ),
                solution=SolutionDraft(
                    material=MaterialDraft(
                        parts=(
                            TextPart(
                                text='The filled point is (2,5), so f(2)=5. The open point at (2,15) does not give the function value at 2.'
                            ),
                        )
                    ),
                    assessment_criteria=[
                        'Identifies the y-coordinate of the filled point as 5.',
                        'Accepts any valid reading that identifies the filled point as the function value.',
                    ],
                    grounding=GroundingReferences(
                        proposition_uuids=[value.uuid, graph_limit.uuid],
                        unit_uuids=[graph_window.target.uuid],
                    ),
                ),
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=procedure_request,
            task=TaskDraft(
                prompt=MaterialDraft(
                    parts=(
                        TextPart(
                            text='Solve 3x+4=19. Show your steps if useful.'
                        ),
                    )
                ),
                grounding=GroundingReferences(
                    proposition_uuids=[inverse.uuid, equation.uuid]
                ),
                solution=SolutionDraft(
                    material=MaterialDraft(
                        parts=(
                            TextPart(
                                text='Subtract 4 from both sides to obtain 3x=15, then divide both sides by 3 to obtain x=5. Substitution gives 3(5)+4=19.'
                            ),
                        )
                    ),
                    assessment_criteria=[
                        'Gives x=5, or an equivalent value.',
                        'Accepts a valid sequence of inverse operations; do not require written steps when the answer establishes the result.',
                    ],
                    grounding=GroundingReferences(
                        proposition_uuids=[inverse.uuid, equation.uuid]
                    ),
                ),
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=argument_request,
            task=TaskDraft(
                prompt=MaterialDraft(
                    parts=(
                        TextPart(
                            text='Explain why adding 3 to both sides of x-3=4 preserves the solution.'
                        ),
                    )
                ),
                grounding=GroundingReferences(
                    proposition_uuids=[
                        argument_premise.uuid,
                        argument_conclusion.uuid,
                    ],
                    argument_uuids=[argument.uuid],
                ),
                solution=SolutionDraft(
                    material=MaterialDraft(
                        parts=(
                            TextPart(
                                text='Adding 3 to both sides preserves equality because the same quantity is added on each side. It gives x=7 without changing the equation’s solutions.'
                            ),
                        )
                    ),
                    assessment_criteria=[
                        'Explains that equal additions preserve equality and yields x=7.',
                    ],
                    grounding=GroundingReferences(
                        proposition_uuids=[
                            argument_premise.uuid,
                            argument_conclusion.uuid,
                        ],
                        argument_uuids=[argument.uuid],
                    ),
                ),
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=solution_request,
            task=TaskDraft(
                prompt=MaterialDraft(
                    parts=(
                        TextPart(text='Solve 2x+3=7. Explain your reasoning.'),
                    )
                ),
                grounding=GroundingReferences(),
                solution=SolutionDraft(
                    material=MaterialDraft(
                        parts=(
                            TextPart(
                                text='Subtract 3 from both sides to get 2x=4, then divide by 2 to get x=2.'
                            ),
                        )
                    ),
                    assessment_criteria=['Finds x=2 using a valid method.'],
                    grounding=GroundingReferences(),
                ),
            ),
        ).with_inputs('request'),
    ]
