"""Dependency composition for the source-pedagogical graph."""

from kms.composition.predictors import PredictorFactory, build_token_budget
from kms.composition.tokenization import LanguageModelTokenizers
from kms.config.settings import Settings
from kms.core.embedding import Embedder
from kms.core.media_budget import ImagePayloadBudget
from kms.database.client import DatabaseClient
from kms.database.source.source_graph_repository import SourceGraphRepository
from kms.langgraph.source_pedagogical.graph import SourcePedagogicalGraph
from kms.local_models.runtime import LocalModelRuntime
from kms.module.source_pedagogical.examples.exercise_finder import (
    exercise_boundary_examples,
    exercise_start_examples,
)
from kms.module.source_pedagogical.examples.exercise_splitter import (
    exercise_splitter_examples,
)
from kms.module.source_pedagogical.examples.exercise_strip_router import (
    exercise_strip_router_examples,
)
from kms.module.source_pedagogical.examples.instruction_finder import (
    instruction_boundary_examples,
    instruction_start_examples,
)
from kms.module.source_pedagogical.examples.instruction_governance import (
    instruction_governance_examples,
)
from kms.module.source_pedagogical.examples.pedagogical_finder import (
    pedagogical_boundary_examples,
    pedagogical_start_examples,
)
from kms.module.source_pedagogical.examples.source_procedure_reconstruction import (
    source_procedure_reconstruction_examples,
)
from kms.module.source_pedagogical.examples.source_statement_reconstruction import (
    source_statement_reconstruction_examples,
)
from kms.module.source_pedagogical.examples.statement_procedure import (
    pedagogical_role_examples,
    procedure_partition_examples,
    statement_partition_examples,
)
from kms.module.source_pedagogical.exercise_finder import (
    ExerciseBoundaryRouterModule,
    ExerciseBoundaryRouterSignature,
    ExerciseStartRouterModule,
    ExerciseStartRouterSignature,
)
from kms.module.source_pedagogical.exercise_splitter import (
    ExerciseSplitterModule,
    ExerciseSplitterSignature,
)
from kms.module.source_pedagogical.exercise_strip_router import (
    ExerciseStripRouterModule,
    ExerciseStripRouterSignature,
)
from kms.module.source_pedagogical.instruction_finder import (
    InstructionBoundaryRouterModule,
    InstructionBoundaryRouterSignature,
    InstructionStartRouterModule,
    InstructionStartRouterSignature,
)
from kms.module.source_pedagogical.instruction_governance import (
    InstructionGovernanceModule,
    InstructionGovernanceSignature,
)
from kms.module.source_pedagogical.pedagogical_finder import (
    PedagogicalBoundaryRouterModule,
    PedagogicalBoundaryRouterSignature,
    PedagogicalStartRouterModule,
    PedagogicalStartRouterSignature,
)
from kms.module.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionModule,
    SourceProcedureReconstructionSignature,
)
from kms.module.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionModule,
    SourceStatementReconstructionSignature,
)
from kms.module.source_pedagogical.statement_procedure import (
    PedagogicalRoleModule,
    PedagogicalRoleSignature,
    ProcedurePartitionModule,
    ProcedurePartitionSignature,
    StatementPartitionModule,
    StatementPartitionSignature,
)
from kms.node.source_pedagogical.embedding import (
    SourcePedagogicalEmbeddingNode,
)
from kms.node.source_pedagogical.exercise_finder import ExerciseFinderNode
from kms.node.source_pedagogical.exercise_splitter import ExerciseSplitterNode
from kms.node.source_pedagogical.instruction_finder import (
    InstructionFinderNode,
)
from kms.node.source_pedagogical.instruction_governance import (
    InstructionGovernanceNode,
)
from kms.node.source_pedagogical.pedagogical_finder import (
    PedagogicalFinderNode,
)
from kms.node.source_pedagogical.persistence import (
    SourcePedagogicalPersistenceNode,
)
from kms.node.source_pedagogical.source_load import (
    SourcePedagogicalSourceLoadNode,
)
from kms.node.source_pedagogical.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms.node.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionNode,
)
from kms.node.source_pedagogical.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms.node.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionNode,
)
from kms.node.source_pedagogical.statement_procedure import (
    StatementProcedureNode,
)
from kms.train.recorder import Recorder


def build_source_pedagogical_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    embedder: Embedder,
    tokenizers: LanguageModelTokenizers,
    recorder: Recorder | None = None,
) -> SourcePedagogicalGraph:
    """Compose pedagogical modules and nodes from shared stage resources."""
    source = settings.source_pedagogical
    predictors = PredictorFactory(settings, local_models, recorder)
    statement_reconstruction_settings = source.source_statement_reconstruction
    procedure_reconstruction_settings = source.source_procedure_reconstruction
    splitter_router = ExerciseStripRouterModule(
        predictors.create(
            ExerciseStripRouterModule,
            source.exercise_splitter.router,
            ExerciseStripRouterSignature,
            examples=exercise_strip_router_examples(),
        )
    )
    splitter = ExerciseSplitterModule(
        predictors.create(
            ExerciseSplitterModule,
            source.exercise_splitter.splitter,
            ExerciseSplitterSignature,
            examples=exercise_splitter_examples(),
        )
    )
    instruction_start = InstructionStartRouterModule(
        predictors.create(
            InstructionStartRouterModule,
            source.instruction_finder.start_router,
            InstructionStartRouterSignature,
            examples=instruction_start_examples(),
        )
    )
    instruction_boundary = InstructionBoundaryRouterModule(
        predictors.create(
            InstructionBoundaryRouterModule,
            source.instruction_finder.boundary_router,
            InstructionBoundaryRouterSignature,
            examples=instruction_boundary_examples(),
        )
    )
    exercise_start = ExerciseStartRouterModule(
        predictors.create(
            ExerciseStartRouterModule,
            source.exercise_finder.start_router,
            ExerciseStartRouterSignature,
            examples=exercise_start_examples(),
        )
    )
    exercise_boundary = ExerciseBoundaryRouterModule(
        predictors.create(
            ExerciseBoundaryRouterModule,
            source.exercise_finder.boundary_router,
            ExerciseBoundaryRouterSignature,
            examples=exercise_boundary_examples(),
        )
    )
    pedagogical_start = PedagogicalStartRouterModule(
        predictors.create(
            PedagogicalStartRouterModule,
            source.pedagogical_finder.start_router,
            PedagogicalStartRouterSignature,
            examples=pedagogical_start_examples(),
        )
    )
    pedagogical_boundary = PedagogicalBoundaryRouterModule(
        predictors.create(
            PedagogicalBoundaryRouterModule,
            source.pedagogical_finder.boundary_router,
            PedagogicalBoundaryRouterSignature,
            examples=pedagogical_boundary_examples(),
        )
    )
    role_typer = PedagogicalRoleModule(
        predictors.create(
            PedagogicalRoleModule,
            source.statement_procedure.role_typer,
            PedagogicalRoleSignature,
            examples=pedagogical_role_examples(),
        )
    )
    statement_partitioner = StatementPartitionModule(
        predictors.create(
            StatementPartitionModule,
            source.statement_procedure.statement_partitioner,
            StatementPartitionSignature,
            examples=statement_partition_examples(),
        )
    )
    procedure_partitioner = ProcedurePartitionModule(
        predictors.create(
            ProcedurePartitionModule,
            source.statement_procedure.procedure_partitioner,
            ProcedurePartitionSignature,
            examples=procedure_partition_examples(),
        )
    )
    governance = InstructionGovernanceModule(
        predictors.create(
            InstructionGovernanceModule,
            source.instruction_governance.inference,
            InstructionGovernanceSignature,
            examples=instruction_governance_examples(),
        )
    )
    statement_reconstruction = SourceStatementReconstructionModule(
        predictors.create(
            SourceStatementReconstructionModule,
            statement_reconstruction_settings.inference,
            SourceStatementReconstructionSignature,
            examples=source_statement_reconstruction_examples(),
        )
    )
    procedure_reconstruction = SourceProcedureReconstructionModule(
        predictors.create(
            SourceProcedureReconstructionModule,
            procedure_reconstruction_settings.inference,
            SourceProcedureReconstructionSignature,
            examples=source_procedure_reconstruction_examples(),
        )
    )
    return SourcePedagogicalGraph(
        source_load=SourcePedagogicalSourceLoadNode(
            SourceGraphRepository(database.session)
        ),
        exercise_splitter=ExerciseSplitterNode(
            splitter_router,
            splitter,
            source.exercise_splitter.context_window,
            batch_size=source.exercise_splitter.batch_size,
            token_counters=tokenizers.counters(
                source.exercise_splitter.router,
                source.exercise_splitter.splitter,
            ),
            image_budget=ImagePayloadBudget(
                **source.exercise_splitter.image_payload.model_dump()
            ),
        ),
        instruction_finder=InstructionFinderNode(
            instruction_start,
            instruction_boundary,
            source.instruction_finder,
            start_budget=build_token_budget(
                settings,
                tokenizers,
                source.instruction_finder.start_router,
                input_token_budget=source.instruction_finder.start_input_token_budget,
                safety_margin_tokens=source.instruction_finder.safety_margin_tokens,
            ),
            boundary_budget=build_token_budget(
                settings,
                tokenizers,
                source.instruction_finder.boundary_router,
                input_token_budget=source.instruction_finder.boundary_input_token_budget,
                safety_margin_tokens=source.instruction_finder.safety_margin_tokens,
            ),
            start_token_counters=tokenizers.counters(
                source.instruction_finder.start_router
            ),
            boundary_token_counters=tokenizers.counters(
                source.instruction_finder.boundary_router
            ),
            image_budget=ImagePayloadBudget(
                **source.instruction_finder.image_payload.model_dump()
            ),
        ),
        exercise_finder=ExerciseFinderNode(
            exercise_start,
            exercise_boundary,
            source.exercise_finder,
            start_budget=build_token_budget(
                settings,
                tokenizers,
                source.exercise_finder.start_router,
                input_token_budget=source.exercise_finder.start_input_token_budget,
                safety_margin_tokens=source.exercise_finder.safety_margin_tokens,
            ),
            boundary_budget=build_token_budget(
                settings,
                tokenizers,
                source.exercise_finder.boundary_router,
                input_token_budget=source.exercise_finder.boundary_input_token_budget,
                safety_margin_tokens=source.exercise_finder.safety_margin_tokens,
            ),
            start_token_counters=tokenizers.counters(
                source.exercise_finder.start_router
            ),
            boundary_token_counters=tokenizers.counters(
                source.exercise_finder.boundary_router
            ),
            image_budget=ImagePayloadBudget(
                **source.exercise_finder.image_payload.model_dump()
            ),
        ),
        pedagogical_finder=PedagogicalFinderNode(
            pedagogical_start,
            pedagogical_boundary,
            source.pedagogical_finder,
            start_budget=build_token_budget(
                settings,
                tokenizers,
                source.pedagogical_finder.start_router,
                input_token_budget=source.pedagogical_finder.start_input_token_budget,
                safety_margin_tokens=source.pedagogical_finder.safety_margin_tokens,
            ),
            boundary_budget=build_token_budget(
                settings,
                tokenizers,
                source.pedagogical_finder.boundary_router,
                input_token_budget=source.pedagogical_finder.boundary_input_token_budget,
                safety_margin_tokens=source.pedagogical_finder.safety_margin_tokens,
            ),
            start_token_counters=tokenizers.counters(
                source.pedagogical_finder.start_router
            ),
            boundary_token_counters=tokenizers.counters(
                source.pedagogical_finder.boundary_router
            ),
            image_budget=ImagePayloadBudget(
                **source.pedagogical_finder.image_payload.model_dump()
            ),
        ),
        statement_procedure=StatementProcedureNode(
            role_typer,
            statement_partitioner,
            procedure_partitioner,
            source.statement_procedure,
            role_budget=build_token_budget(
                settings,
                tokenizers,
                source.statement_procedure.role_typer,
                input_token_budget=source.statement_procedure.role_input_token_budget,
                safety_margin_tokens=source.statement_procedure.safety_margin_tokens,
            ),
            statement_budget=build_token_budget(
                settings,
                tokenizers,
                source.statement_procedure.statement_partitioner,
                input_token_budget=source.statement_procedure.statement_input_token_budget,
                safety_margin_tokens=source.statement_procedure.safety_margin_tokens,
            ),
            procedure_budget=build_token_budget(
                settings,
                tokenizers,
                source.statement_procedure.procedure_partitioner,
                input_token_budget=source.statement_procedure.procedure_input_token_budget,
                safety_margin_tokens=source.statement_procedure.safety_margin_tokens,
            ),
            image_budget=ImagePayloadBudget(
                **source.statement_procedure.image_payload.model_dump()
            ),
        ),
        instruction_governance=InstructionGovernanceNode(
            governance,
            source.instruction_governance,
            budget=build_token_budget(
                settings,
                tokenizers,
                source.instruction_governance.inference,
                input_token_budget=source.instruction_governance.input_token_budget,
                safety_margin_tokens=source.instruction_governance.safety_margin_tokens,
            ),
            image_budget=ImagePayloadBudget(
                **source.instruction_governance.image_payload.model_dump()
            ),
            token_counters=tokenizers.counters(
                source.instruction_governance.inference
            ),
        ),
        source_statement_reconstruction=SourceStatementReconstructionNode(
            statement_reconstruction,
            budget=build_token_budget(
                settings,
                tokenizers,
                statement_reconstruction_settings.inference,
                input_token_budget=(
                    statement_reconstruction_settings.input_token_budget
                ),
                safety_margin_tokens=(
                    statement_reconstruction_settings.safety_margin_tokens
                ),
            ),
            batch_size=statement_reconstruction_settings.batch_size,
            image_budget=ImagePayloadBudget(
                **statement_reconstruction_settings.image_payload.model_dump()
            ),
        ),
        source_procedure_reconstruction=SourceProcedureReconstructionNode(
            procedure_reconstruction,
            budget=build_token_budget(
                settings,
                tokenizers,
                procedure_reconstruction_settings.inference,
                input_token_budget=(
                    procedure_reconstruction_settings.input_token_budget
                ),
                safety_margin_tokens=(
                    procedure_reconstruction_settings.safety_margin_tokens
                ),
            ),
            batch_size=procedure_reconstruction_settings.batch_size,
            image_budget=ImagePayloadBudget(
                **procedure_reconstruction_settings.image_payload.model_dump()
            ),
        ),
        source_statement_embedding=SourceStatementEmbeddingNode(embedder),
        source_procedure_embedding=SourceProcedureEmbeddingNode(embedder),
        embedding=SourcePedagogicalEmbeddingNode(embedder),
        persistence=SourcePedagogicalPersistenceNode(
            SourceGraphRepository(database.session)
        ),
    )
