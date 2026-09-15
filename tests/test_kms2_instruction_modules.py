import asyncio

from kms2.core.model import BlockType, PedagogicalMember, SourceBlockContext
from kms2.module.source.exercise_finder import (
    ExerciseBoundaryRouterModule,
    ExerciseBoundaryRouterSignature,
    ExerciseStartRouterModule,
    ExerciseStartRouterSignature,
)
from kms2.module.source.instruction_finder import (
    InstructionBoundaryRouterModule,
    InstructionStartRouterModule,
)
from kms2.module.source.instruction_governance import (
    InstructionGovernanceModule,
)
from kms2.module.source.pedagogical_finder import (
    PedagogicalBoundaryRouterModule,
    PedagogicalBoundaryRouterSignature,
    PedagogicalStartRouterModule,
)
from kms2.module.source.statement_procedure import (
    PedagogicalRoleModule,
    ProcedurePartitionModule,
    StatementPartitionModule,
)


class Prediction:
    is_instruction_start = True
    is_instruction_boundary = True
    is_exercise_start = True
    is_exercise_boundary = True
    is_pedagogical_start = True
    is_pedagogical_boundary = True
    has_statement = True
    has_procedure = True
    statement_positions = [0]
    procedure_positions = [1]
    governs = True


class RecordingPredictor:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return Prediction()

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return Prediction()


def _context(content: str) -> SourceBlockContext:
    return SourceBlockContext(block_type=BlockType.PARAGRAPH, content=content)


def _member(position: int) -> PedagogicalMember:
    return PedagogicalMember(
        position=position, source_block=_context(str(position))
    )


def test_source_modules_forward_named_inputs_and_outputs():
    context = _context('context')
    members = [_member(0), _member(1)]
    predictor = RecordingPredictor()

    assert InstructionStartRouterModule(predictor)(
        context_before=[context], target_block=context, context_after=[context]
    )
    assert InstructionBoundaryRouterModule(predictor)(
        start_block=context,
        context_before=[context],
        candidate_block=context,
        context_after=[context],
    )
    assert ExerciseStartRouterModule(predictor)(
        context_before=[context],
        target_block=context,
        context_after=[context],
    )
    assert ExerciseBoundaryRouterModule(predictor)(
        start_block=context,
        context_before=[context],
        candidate_block=context,
        context_after=[context],
    )
    assert PedagogicalStartRouterModule(predictor)(
        context_before=[context], target_block=context, context_after=[context]
    )
    assert PedagogicalBoundaryRouterModule(predictor)(
        start_block=context,
        context_before=[context],
        candidate_block=context,
        context_after=[context],
    )
    assert PedagogicalRoleModule(predictor)(members=members) == (True, True)
    assert StatementPartitionModule(predictor)(members=members) == [0]
    assert ProcedurePartitionModule(predictor)(members=members) == [1]
    assert InstructionGovernanceModule(predictor)(
        instruction_blocks=[context],
        context_before=[context],
        statement_blocks=[context],
        context_after=[context],
    )
    assert len(predictor.demos) == 10
    assert any(
        demo.governs is False
        and demo.statement_blocks[0].content.startswith('289.')
        for demo in predictor.demos
    )
    assert predictor.calls[-1].keys() == {
        'instruction_blocks',
        'context_before',
        'statement_blocks',
        'context_after',
    }


def test_finder_modules_install_corrected_live_demos():
    predictors = [RecordingPredictor() for _ in range(6)]
    _modules = [
        ExerciseStartRouterModule(predictors[0]),
        ExerciseBoundaryRouterModule(predictors[1]),
        InstructionStartRouterModule(predictors[2]),
        InstructionBoundaryRouterModule(predictors[3]),
        PedagogicalStartRouterModule(predictors[4]),
        PedagogicalBoundaryRouterModule(predictors[5]),
    ]

    assert [len(predictor.demos) for predictor in predictors] == [
        8,
        11,
        7,
        7,
        4,
        3,
    ]
    assert (
        predictors[0].demos[0].target_block.content
        == '280. Find the gradient of'
    )
    assert predictors[1].demos[2].is_exercise_boundary is True
    assert predictors[2].demos[0].is_instruction_start is True
    assert predictors[3].demos[0].is_instruction_boundary is True
    assert predictors[4].demos[2].is_pedagogical_start is True
    assert predictors[5].demos[1].is_pedagogical_boundary is True
    assert (
        'distinct individual exercise' in ExerciseStartRouterSignature.__doc__
    )
    assert 'exclusive' in ExerciseBoundaryRouterSignature.__doc__
    assert 'context_after' in ExerciseBoundaryRouterSignature.__doc__
    assert 'exclusive' in PedagogicalBoundaryRouterSignature.__doc__


def test_source_modules_forward_inputs_asynchronously():
    context = _context('context')
    predictor = RecordingPredictor()

    result = asyncio.run(
        InstructionGovernanceModule(predictor).aforward(
            instruction_blocks=[context],
            context_before=[],
            statement_blocks=[context, context],
            context_after=[],
        )
    )

    assert result is True
    assert predictor.calls[0]['statement_blocks'] == [context, context]
