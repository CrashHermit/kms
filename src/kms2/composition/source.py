"""Dependency composition for the KMS2 source graph."""

from kms2.composition.predictors import PredictorFactory
from kms2.config.settings import Settings
from kms2.database.client import DatabaseClient
from kms2.database.source.source_graph_repository import SourceGraphRepository
from kms2.langgraph.source.graph import SourceGraph
from kms2.local_models import LocalModelRuntime
from kms2.module.source.content_correction import (
    ContentCorrectorModule,
    ContentCorrectorSignature,
)
from kms2.module.source.exercise_finder import (
    ExerciseBoundaryRouterModule,
    ExerciseBoundaryRouterSignature,
    ExerciseStartRouterModule,
    ExerciseStartRouterSignature,
)
from kms2.module.source.exercise_splitter import (
    ExerciseSplitterModule,
    ExerciseSplitterSignature,
)
from kms2.module.source.exercise_strip_router import (
    ExerciseStripRouterModule,
    ExerciseStripRouterSignature,
)
from kms2.module.source.formatting import FormatterModule, FormatterSignature
from kms2.module.source.image_description import (
    ImageDescriptionModule,
    ImageDescriptionSignature,
)
from kms2.module.source.image_seam import (
    ImageSeamJudgeModule,
    ImageSeamSignature,
)
from kms2.module.source.instruction_finder import (
    InstructionBoundaryRouterModule,
    InstructionBoundaryRouterSignature,
    InstructionStartRouterModule,
    InstructionStartRouterSignature,
)
from kms2.module.source.instruction_governance import (
    InstructionGovernanceModule,
    InstructionGovernanceSignature,
)
from kms2.module.source.pedagogical_finder import (
    PedagogicalBoundaryRouterModule,
    PedagogicalBoundaryRouterSignature,
    PedagogicalStartRouterModule,
    PedagogicalStartRouterSignature,
)
from kms2.module.source.statement_procedure import (
    PedagogicalRoleModule,
    PedagogicalRoleSignature,
    ProcedurePartitionModule,
    ProcedurePartitionSignature,
    StatementPartitionModule,
    StatementPartitionSignature,
)
from kms2.module.source.text_seam_judge import (
    TextSeamJudgeModule,
    TextSeamSignature,
)
from kms2.module.source.text_seam_rewriter import (
    TextSeamRewriterModule,
    TextSeamRewriteSignature,
)
from kms2.node.source.content_correction import ContentCorrectionNode
from kms2.node.source.embedding import EmbeddingNode
from kms2.node.source.exercise_finder import ExerciseFinderNode
from kms2.node.source.exercise_splitter import ExerciseSplitterNode
from kms2.node.source.formatting import FormattingNode
from kms2.node.source.image_description import ImageDescriptionNode
from kms2.node.source.image_seam import ImageSeamNode
from kms2.node.source.instruction_finder import InstructionFinderNode
from kms2.node.source.instruction_governance import InstructionGovernanceNode
from kms2.node.source.ocr import OCRNode
from kms2.node.source.pedagogical_finder import PedagogicalFinderNode
from kms2.node.source.persistence import SourcePersistenceNode
from kms2.node.source.statement_procedure import StatementProcedureNode
from kms2.node.source.text_seam import TextSeamNode
from kms2.ocr.mistral import MistralOCRProvider
from kms2.train.recorder import Recorder


def build_source_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    recorder: Recorder | None = None,
) -> SourceGraph:
    """Compose the source graph from settings and started runtime resources."""
    source = settings.source
    predictors = PredictorFactory(local_models, recorder)
    content_corrector = ContentCorrectorModule(
        predictors.create(
            source.content_correction,
            ContentCorrectorSignature,
        )
    )
    formatter = FormatterModule(
        predictors.create(source.formatting, FormatterSignature)
    )
    text_seam_judge = TextSeamJudgeModule(
        predictors.create(source.text_seam.judge, TextSeamSignature)
    )
    text_seam_rewriter = TextSeamRewriterModule(
        predictors.create(
            source.text_seam.rewriter,
            TextSeamRewriteSignature,
        )
    )
    image_seam_judge = ImageSeamJudgeModule(
        predictors.create(source.image_seam, ImageSeamSignature)
    )
    image_describer = ImageDescriptionModule(
        predictors.create(
            source.image_description.inference,
            ImageDescriptionSignature,
        )
    )
    splitter_router = ExerciseStripRouterModule(
        predictors.create(
            source.exercise_splitter.router,
            ExerciseStripRouterSignature,
        )
    )
    exercise_splitter = ExerciseSplitterModule(
        predictors.create(
            source.exercise_splitter.splitter,
            ExerciseSplitterSignature,
        )
    )
    instruction_start_router = InstructionStartRouterModule(
        predictors.create(
            source.instruction_finder.start_router,
            InstructionStartRouterSignature,
        )
    )
    instruction_boundary_router = InstructionBoundaryRouterModule(
        predictors.create(
            source.instruction_finder.boundary_router,
            InstructionBoundaryRouterSignature,
        )
    )
    pedagogical_start_router = PedagogicalStartRouterModule(
        predictors.create(
            source.pedagogical_finder.start_router,
            PedagogicalStartRouterSignature,
        )
    )
    pedagogical_boundary_router = PedagogicalBoundaryRouterModule(
        predictors.create(
            source.pedagogical_finder.boundary_router,
            PedagogicalBoundaryRouterSignature,
        )
    )
    role_typer = PedagogicalRoleModule(
        predictors.create(
            source.statement_procedure.role_typer,
            PedagogicalRoleSignature,
        )
    )
    statement_partitioner = StatementPartitionModule(
        predictors.create(
            source.statement_procedure.statement_partitioner,
            StatementPartitionSignature,
        )
    )
    procedure_partitioner = ProcedurePartitionModule(
        predictors.create(
            source.statement_procedure.procedure_partitioner,
            ProcedurePartitionSignature,
        )
    )
    exercise_start_router = ExerciseStartRouterModule(
        predictors.create(
            source.exercise_finder.start_router,
            ExerciseStartRouterSignature,
        )
    )
    exercise_boundary_router = ExerciseBoundaryRouterModule(
        predictors.create(
            source.exercise_finder.boundary_router,
            ExerciseBoundaryRouterSignature,
        )
    )
    instruction_governance = InstructionGovernanceModule(
        predictors.create(
            source.instruction_governance.inference,
            InstructionGovernanceSignature,
        )
    )

    return SourceGraph(
        ocr=OCRNode(MistralOCRProvider(settings.ocr)),
        content_correction=ContentCorrectionNode(content_corrector),
        formatter=FormattingNode(formatter),
        text_seam=TextSeamNode(text_seam_judge, text_seam_rewriter),
        image_seam=ImageSeamNode(image_seam_judge),
        image_description=ImageDescriptionNode(
            image_describer,
            source.image_description.context_window,
        ),
        exercise_splitter=ExerciseSplitterNode(
            splitter_router,
            exercise_splitter,
            source.exercise_splitter.context_window,
        ),
        instruction_finder=InstructionFinderNode(
            instruction_start_router,
            instruction_boundary_router,
            source.instruction_finder,
        ),
        exercise_finder=ExerciseFinderNode(
            exercise_start_router,
            exercise_boundary_router,
            source.exercise_finder,
        ),
        pedagogical_finder=PedagogicalFinderNode(
            pedagogical_start_router,
            pedagogical_boundary_router,
            source.pedagogical_finder,
        ),
        statement_procedure=StatementProcedureNode(
            role_typer,
            statement_partitioner,
            procedure_partitioner,
        ),
        instruction_governance=InstructionGovernanceNode(
            instruction_governance,
            source.instruction_governance,
        ),
        embedding=EmbeddingNode(local_models),
        persistence=SourcePersistenceNode(
            SourceGraphRepository(database.session),
        ),
    )
