"""Explicit dependency composition for KMS2 runtime components."""

from functools import partial

from kms2.config import Settings
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.semantic.repository import SemanticRepository
from kms2.database.source.repository import SourceRepository
from kms2.langgraph.semantic.graph import SemanticGraph
from kms2.langgraph.source.graph import SourceGraph
from kms2.local_models import LocalModelRuntime
from kms2.module.semantic.source_entity_description import (
    SourceEntityDescriptionModule,
    SourceEntityDescriptionSignature,
)
from kms2.module.semantic.source_event_description import (
    SourceEventDescriptionModule,
    SourceEventDescriptionSignature,
)
from kms2.module.semantic.source_predicate_description import (
    SourcePredicateDescriptionModule,
    SourcePredicateDescriptionSignature,
)
from kms2.module.semantic.triplet import (
    FactExtractionSignature,
    FactExtractorModule,
    TripletDecomposerModule,
    TripletDecompositionSignature,
)
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
from kms2.module.source.splitter import (
    ExerciseSplitterModule,
    ExerciseSplitterSignature,
    ExerciseStripRouterModule,
    ExerciseStripRouterSignature,
)
from kms2.module.source.statement_procedure import (
    PedagogicalRoleModule,
    PedagogicalRoleSignature,
    ProcedurePartitionModule,
    ProcedurePartitionSignature,
    StatementPartitionModule,
    StatementPartitionSignature,
)
from kms2.module.source.text_seam import (
    TextSeamJudgeModule,
    TextSeamRewriterModule,
    TextSeamRewriteSignature,
    TextSeamSignature,
)
from kms2.node.semantic.source_entity_description import (
    SourceEntityDescriptionNode,
)
from kms2.node.semantic.source_entity_description_load import (
    SourceEntityDescriptionLoadNode,
)
from kms2.node.semantic.source_entity_embedding import SourceEntityEmbeddingNode
from kms2.node.semantic.source_entity_persistence import (
    SourceEntityPersistenceNode,
)
from kms2.node.semantic.source_event_description import (
    SourceEventDescriptionNode,
)
from kms2.node.semantic.source_event_description_load import (
    SourceEventDescriptionLoadNode,
)
from kms2.node.semantic.source_event_embedding import SourceEventEmbeddingNode
from kms2.node.semantic.source_event_persistence import (
    SourceEventPersistenceNode,
)
from kms2.node.semantic.source_predicate_description import (
    SourcePredicateDescriptionNode,
)
from kms2.node.semantic.source_predicate_description_load import (
    SourcePredicateDescriptionLoadNode,
)
from kms2.node.semantic.source_predicate_embedding import (
    SourcePredicateEmbeddingNode,
)
from kms2.node.semantic.source_predicate_persistence import (
    SourcePredicatePersistenceNode,
)
from kms2.node.semantic.triplet import (
    FactExtractionNode,
    TripletDecompositionNode,
)
from kms2.node.semantic.triplet_load import TripletSourceLoadNode
from kms2.node.semantic.triplet_persistence import TripletPersistenceNode
from kms2.node.source.content_correction import ContentCorrectionNode
from kms2.node.source.embedding import EmbeddingNode
from kms2.node.source.exercise_finder import ExerciseFinderNode
from kms2.node.source.formatting import FormattingNode
from kms2.node.source.image_description import ImageDescriptionNode
from kms2.node.source.image_seam import ImageSeamNode
from kms2.node.source.instruction_finder import InstructionFinderNode
from kms2.node.source.instruction_governance import InstructionGovernanceNode
from kms2.node.source.ocr import OCRNode
from kms2.node.source.pedagogical_finder import PedagogicalFinderNode
from kms2.node.source.persistence import SourcePersistenceNode
from kms2.node.source.splitter import SplitterNode
from kms2.node.source.statement_procedure import StatementProcedureNode
from kms2.node.source.text_seam import TextSeamNode
from kms2.ocr.mistral import MistralOCRProvider


def build_source_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
) -> SourceGraph:
    """Compose the source graph from settings and started runtime resources."""
    source = settings.source
    content_corrector = ContentCorrectorModule(
        local_models.predictor(
            source.content_correction,
            ContentCorrectorSignature,
        )
    )
    formatter = FormatterModule(
        local_models.predictor(source.formatting, FormatterSignature)
    )
    text_seam_judge = TextSeamJudgeModule(
        local_models.predictor(source.text_seam.judge, TextSeamSignature)
    )
    text_seam_rewriter = TextSeamRewriterModule(
        local_models.predictor(
            source.text_seam.rewriter,
            TextSeamRewriteSignature,
        )
    )
    image_seam_judge = ImageSeamJudgeModule(
        local_models.predictor(source.image_seam, ImageSeamSignature)
    )
    image_describer = ImageDescriptionModule(
        local_models.predictor(
            source.image_description.inference,
            ImageDescriptionSignature,
        )
    )
    splitter_router = ExerciseStripRouterModule(
        local_models.predictor(
            source.splitter.router,
            ExerciseStripRouterSignature,
        )
    )
    splitter = ExerciseSplitterModule(
        local_models.predictor(
            source.splitter.splitter,
            ExerciseSplitterSignature,
        )
    )
    instruction_start_router = InstructionStartRouterModule(
        local_models.predictor(
            source.instruction_finder.start_router,
            InstructionStartRouterSignature,
        )
    )
    instruction_boundary_router = InstructionBoundaryRouterModule(
        local_models.predictor(
            source.instruction_finder.boundary_router,
            InstructionBoundaryRouterSignature,
        )
    )
    pedagogical_start_router = PedagogicalStartRouterModule(
        local_models.predictor(
            source.pedagogical_finder.start_router,
            PedagogicalStartRouterSignature,
        )
    )
    pedagogical_boundary_router = PedagogicalBoundaryRouterModule(
        local_models.predictor(
            source.pedagogical_finder.boundary_router,
            PedagogicalBoundaryRouterSignature,
        )
    )
    role_typer = PedagogicalRoleModule(
        local_models.predictor(
            source.statement_procedure.role_typer,
            PedagogicalRoleSignature,
        )
    )
    statement_partitioner = StatementPartitionModule(
        local_models.predictor(
            source.statement_procedure.statement_partitioner,
            StatementPartitionSignature,
        )
    )
    procedure_partitioner = ProcedurePartitionModule(
        local_models.predictor(
            source.statement_procedure.procedure_partitioner,
            ProcedurePartitionSignature,
        )
    )
    exercise_start_router = ExerciseStartRouterModule(
        local_models.predictor(
            source.exercise_finder.start_router,
            ExerciseStartRouterSignature,
        )
    )
    exercise_boundary_router = ExerciseBoundaryRouterModule(
        local_models.predictor(
            source.exercise_finder.boundary_router,
            ExerciseBoundaryRouterSignature,
        )
    )
    instruction_governance = InstructionGovernanceModule(
        local_models.predictor(
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
        splitter=SplitterNode(
            splitter_router,
            splitter,
            source.splitter.context_window,
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
        embedding=EmbeddingNode(local_models.embedding),
        persistence=SourcePersistenceNode(
            SourceRepository(database.session),
            partial(
                schema.ensure_schema,
                database.session,
                embedding_dimension=settings.local_models.embedding.model.dimension,
            ),
        ),
    )


def build_semantic_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
) -> SemanticGraph:
    """Compose the complete semantic graph from shared runtime resources."""
    semantic = settings.semantic
    source_repository = SourceRepository(database.session)
    semantic_repository = SemanticRepository(database.session)
    fact_extractor = FactExtractorModule(
        local_models.predictor(
            semantic.fact_extraction,
            FactExtractionSignature,
        )
    )
    triplet_decomposer = TripletDecomposerModule(
        local_models.predictor(
            semantic.triplet_decomposition,
            TripletDecompositionSignature,
        )
    )
    return SemanticGraph(
        triplet_source_load=TripletSourceLoadNode(source_repository),
        fact_extraction=FactExtractionNode(
            fact_extractor,
            semantic.context_window,
        ),
        triplet_decomposition=TripletDecompositionNode(triplet_decomposer),
        triplet_persistence=TripletPersistenceNode(
            semantic_repository,
            partial(
                schema.ensure_schema,
                database.session,
                embedding_dimension=settings.local_models.embedding.model.dimension,
            ),
        ),
        source_entity_description_load=SourceEntityDescriptionLoadNode(
            source_repository,
            semantic_repository,
            semantic.source_entity_description.context_window,
        ),
        source_entity_description=SourceEntityDescriptionNode(
            SourceEntityDescriptionModule(
                local_models.predictor(
                    semantic.source_entity_description.inference,
                    SourceEntityDescriptionSignature,
                )
            )
        ),
        source_entity_embedding=SourceEntityEmbeddingNode(
            local_models.embedding
        ),
        source_entity_persistence=SourceEntityPersistenceNode(
            semantic_repository,
            partial(
                schema.ensure_schema,
                database.session,
                embedding_dimension=settings.local_models.embedding.model.dimension,
            ),
        ),
        source_event_description_load=SourceEventDescriptionLoadNode(
            source_repository,
            semantic_repository,
            semantic.source_event_description.context_window,
        ),
        source_event_description=SourceEventDescriptionNode(
            SourceEventDescriptionModule(
                local_models.predictor(
                    semantic.source_event_description.inference,
                    SourceEventDescriptionSignature,
                )
            )
        ),
        source_event_embedding=SourceEventEmbeddingNode(local_models.embedding),
        source_event_persistence=SourceEventPersistenceNode(
            semantic_repository,
            partial(
                schema.ensure_schema,
                database.session,
                embedding_dimension=settings.local_models.embedding.model.dimension,
            ),
        ),
        source_predicate_description_load=SourcePredicateDescriptionLoadNode(
            source_repository,
            semantic_repository,
            semantic.source_predicate_description.context_window,
        ),
        source_predicate_description=SourcePredicateDescriptionNode(
            SourcePredicateDescriptionModule(
                local_models.predictor(
                    semantic.source_predicate_description.inference,
                    SourcePredicateDescriptionSignature,
                )
            )
        ),
        source_predicate_embedding=SourcePredicateEmbeddingNode(
            local_models.embedding
        ),
        source_predicate_persistence=SourcePredicatePersistenceNode(
            semantic_repository,
            partial(
                schema.ensure_schema,
                database.session,
                embedding_dimension=settings.local_models.embedding.model.dimension,
            ),
        ),
    )


__all__ = ['build_semantic_graph', 'build_source_graph']
