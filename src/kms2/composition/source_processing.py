"""Dependency composition for the KMS2 source graph."""

from kms2.composition.predictors import PredictorFactory
from kms2.config.settings import Settings
from kms2.database.client import DatabaseClient
from kms2.database.source.source_graph_repository import SourceGraphRepository
from kms2.langgraph.source_processing.graph import SourceProcessingGraph
from kms2.local_models.runtime import LocalModelRuntime
from kms2.local_models.token_counting import LocalTokenizers
from kms2.module.source_processing.content_correction import (
    ContentCorrectorModule,
    ContentCorrectorSignature,
)
from kms2.module.source_processing.exercise_finder import (
    ExerciseBoundaryRouterModule,
    ExerciseBoundaryRouterSignature,
    ExerciseStartRouterModule,
    ExerciseStartRouterSignature,
)
from kms2.module.source_processing.exercise_splitter import (
    ExerciseSplitterModule,
    ExerciseSplitterSignature,
)
from kms2.module.source_processing.exercise_strip_router import (
    ExerciseStripRouterModule,
    ExerciseStripRouterSignature,
)
from kms2.module.source_processing.formatting import (
    FormatterModule,
    FormatterSignature,
)
from kms2.module.source_processing.image_description import (
    ImageDescriptionModule,
    ImageDescriptionSignature,
)
from kms2.module.source_processing.image_seam import (
    ImageSeamJudgeModule,
    ImageSeamSignature,
)
from kms2.module.source_processing.instruction_finder import (
    InstructionBoundaryRouterModule,
    InstructionBoundaryRouterSignature,
    InstructionStartRouterModule,
    InstructionStartRouterSignature,
)
from kms2.module.source_processing.instruction_governance import (
    InstructionGovernanceModule,
    InstructionGovernanceSignature,
)
from kms2.module.source_processing.pedagogical_finder import (
    PedagogicalBoundaryRouterModule,
    PedagogicalBoundaryRouterSignature,
    PedagogicalStartRouterModule,
    PedagogicalStartRouterSignature,
)
from kms2.module.source_processing.statement_procedure import (
    PedagogicalRoleModule,
    PedagogicalRoleSignature,
    ProcedurePartitionModule,
    ProcedurePartitionSignature,
    StatementPartitionModule,
    StatementPartitionSignature,
)
from kms2.module.source_processing.text_seam_judge import (
    TextSeamJudgeModule,
    TextSeamSignature,
)
from kms2.module.source_processing.text_seam_rewriter import (
    TextSeamRewriterModule,
    TextSeamRewriteSignature,
)
from kms2.node.source_processing.content_correction import ContentCorrectionNode
from kms2.node.source_processing.embedding import EmbeddingNode
from kms2.node.source_processing.exercise_finder import ExerciseFinderNode
from kms2.node.source_processing.exercise_splitter import ExerciseSplitterNode
from kms2.node.source_processing.formatting import FormattingNode
from kms2.node.source_processing.image_description import ImageDescriptionNode
from kms2.node.source_processing.image_seam import ImageSeamNode
from kms2.node.source_processing.instruction_finder import InstructionFinderNode
from kms2.node.source_processing.instruction_governance import (
    InstructionGovernanceNode,
)
from kms2.node.source_processing.ocr import OCRNode
from kms2.node.source_processing.pedagogical_finder import PedagogicalFinderNode
from kms2.node.source_processing.persistence import SourcePersistenceNode
from kms2.node.source_processing.statement_procedure import (
    StatementProcedureNode,
)
from kms2.node.source_processing.text_seam import TextSeamNode
from kms2.ocr.mistral import MistralOCRProvider
from kms2.train.recorder import Recorder


def build_source_processing_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    tokenizers: LocalTokenizers,
    recorder: Recorder | None = None,
) -> SourceProcessingGraph:
    """Compose the source graph from settings and started runtime resources."""
    source = settings.source_processing
    predictors = PredictorFactory(local_models, recorder)
    content_corrector = ContentCorrectorModule(
        predictors.create(
            ContentCorrectorModule,
            source.content_correction,
            ContentCorrectorSignature,
        )
    )
    formatter = FormatterModule(
        predictors.create(
            FormatterModule,
            source.formatting,
            FormatterSignature,
        )
    )
    text_seam_judge = TextSeamJudgeModule(
        predictors.create(
            TextSeamJudgeModule,
            source.text_seam.judge,
            TextSeamSignature,
        )
    )
    text_seam_rewriter = TextSeamRewriterModule(
        predictors.create(
            TextSeamRewriterModule,
            source.text_seam.rewriter,
            TextSeamRewriteSignature,
        )
    )
    image_seam_judge = ImageSeamJudgeModule(
        predictors.create(
            ImageSeamJudgeModule,
            source.image_seam,
            ImageSeamSignature,
        )
    )
    image_describer = ImageDescriptionModule(
        predictors.create(
            ImageDescriptionModule,
            source.image_description.inference,
            ImageDescriptionSignature,
        )
    )
    splitter_router = ExerciseStripRouterModule(
        predictors.create(
            ExerciseStripRouterModule,
            source.exercise_splitter.router,
            ExerciseStripRouterSignature,
        )
    )
    exercise_splitter = ExerciseSplitterModule(
        predictors.create(
            ExerciseSplitterModule,
            source.exercise_splitter.splitter,
            ExerciseSplitterSignature,
        )
    )
    instruction_start_router = InstructionStartRouterModule(
        predictors.create(
            InstructionStartRouterModule,
            source.instruction_finder.start_router,
            InstructionStartRouterSignature,
        )
    )
    instruction_boundary_router = InstructionBoundaryRouterModule(
        predictors.create(
            InstructionBoundaryRouterModule,
            source.instruction_finder.boundary_router,
            InstructionBoundaryRouterSignature,
        )
    )
    pedagogical_start_router = PedagogicalStartRouterModule(
        predictors.create(
            PedagogicalStartRouterModule,
            source.pedagogical_finder.start_router,
            PedagogicalStartRouterSignature,
        )
    )
    pedagogical_boundary_router = PedagogicalBoundaryRouterModule(
        predictors.create(
            PedagogicalBoundaryRouterModule,
            source.pedagogical_finder.boundary_router,
            PedagogicalBoundaryRouterSignature,
        )
    )
    role_typer = PedagogicalRoleModule(
        predictors.create(
            PedagogicalRoleModule,
            source.statement_procedure.role_typer,
            PedagogicalRoleSignature,
        )
    )
    statement_partitioner = StatementPartitionModule(
        predictors.create(
            StatementPartitionModule,
            source.statement_procedure.statement_partitioner,
            StatementPartitionSignature,
        )
    )
    procedure_partitioner = ProcedurePartitionModule(
        predictors.create(
            ProcedurePartitionModule,
            source.statement_procedure.procedure_partitioner,
            ProcedurePartitionSignature,
        )
    )
    exercise_start_router = ExerciseStartRouterModule(
        predictors.create(
            ExerciseStartRouterModule,
            source.exercise_finder.start_router,
            ExerciseStartRouterSignature,
        )
    )
    exercise_boundary_router = ExerciseBoundaryRouterModule(
        predictors.create(
            ExerciseBoundaryRouterModule,
            source.exercise_finder.boundary_router,
            ExerciseBoundaryRouterSignature,
        )
    )
    instruction_governance = InstructionGovernanceModule(
        predictors.create(
            InstructionGovernanceModule,
            source.instruction_governance.inference,
            InstructionGovernanceSignature,
        )
    )

    return SourceProcessingGraph(
        ocr=OCRNode(MistralOCRProvider(settings.ocr)),
        content_correction=ContentCorrectionNode(content_corrector),
        formatter=FormattingNode(formatter),
        text_seam=TextSeamNode(text_seam_judge, text_seam_rewriter),
        image_seam=ImageSeamNode(image_seam_judge),
        image_description=ImageDescriptionNode(
            image_describer,
            source.image_description.context_window,
            token_counters=tokenizers.text_counters(
                source.image_description.inference.model_server_profile
            ),
        ),
        exercise_splitter=ExerciseSplitterNode(
            splitter_router,
            exercise_splitter,
            source.exercise_splitter.context_window,
            token_counters=tokenizers.text_counters(
                source.exercise_splitter.router.model_server_profile,
                source.exercise_splitter.splitter.model_server_profile,
            ),
        ),
        instruction_finder=InstructionFinderNode(
            instruction_start_router,
            instruction_boundary_router,
            source.instruction_finder,
            start_token_counters=tokenizers.text_counters(
                source.instruction_finder.start_router.model_server_profile
            ),
            boundary_token_counters=tokenizers.text_counters(
                source.instruction_finder.boundary_router.model_server_profile
            ),
        ),
        exercise_finder=ExerciseFinderNode(
            exercise_start_router,
            exercise_boundary_router,
            source.exercise_finder,
            start_token_counters=tokenizers.text_counters(
                source.exercise_finder.start_router.model_server_profile
            ),
            boundary_token_counters=tokenizers.text_counters(
                source.exercise_finder.boundary_router.model_server_profile
            ),
        ),
        pedagogical_finder=PedagogicalFinderNode(
            pedagogical_start_router,
            pedagogical_boundary_router,
            source.pedagogical_finder,
            start_token_counters=tokenizers.text_counters(
                source.pedagogical_finder.start_router.model_server_profile
            ),
            boundary_token_counters=tokenizers.text_counters(
                source.pedagogical_finder.boundary_router.model_server_profile
            ),
        ),
        statement_procedure=StatementProcedureNode(
            role_typer,
            statement_partitioner,
            procedure_partitioner,
        ),
        instruction_governance=InstructionGovernanceNode(
            instruction_governance,
            source.instruction_governance,
            token_counters=tokenizers.text_counters(
                source.instruction_governance.inference.model_server_profile
            ),
        ),
        embedding=EmbeddingNode(local_models),
        persistence=SourcePersistenceNode(
            SourceGraphRepository(database.session),
        ),
    )
