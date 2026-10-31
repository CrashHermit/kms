"""Dependency composition for the source-pedagogical graph."""

from kms2.composition.predictors import PredictorFactory, build_token_budget
from kms2.config.settings import Settings
from kms2.core.embedding import EmbeddingClient
from kms2.database.client import DatabaseClient
from kms2.database.source.source_graph_repository import SourceGraphRepository
from kms2.inference.token_counting import InferenceTokenizers
from kms2.langgraph.source_pedagogical.graph import SourcePedagogicalGraph
from kms2.local_models.runtime import LocalModelRuntime
from kms2.module.source_pedagogical.examples.exercise_finder import (
    exercise_boundary_examples,
    exercise_start_examples,
)
from kms2.module.source_pedagogical.examples.exercise_splitter import (
    exercise_splitter_examples,
)
from kms2.module.source_pedagogical.examples.exercise_strip_router import (
    exercise_strip_router_examples,
)
from kms2.module.source_pedagogical.examples.instruction_finder import (
    instruction_boundary_examples,
    instruction_start_examples,
)
from kms2.module.source_pedagogical.examples.instruction_governance import (
    instruction_governance_examples,
)
from kms2.module.source_pedagogical.examples.pedagogical_finder import (
    pedagogical_boundary_examples,
    pedagogical_start_examples,
)
from kms2.module.source_pedagogical.examples.source_procedure_reconstruction import (
    source_procedure_reconstruction_examples,
)
from kms2.module.source_pedagogical.examples.source_statement_reconstruction import (
    source_statement_reconstruction_examples,
)
from kms2.module.source_pedagogical.examples.statement_procedure import (
    pedagogical_role_examples,
    procedure_partition_examples,
    statement_partition_examples,
)
from kms2.module.source_pedagogical.exercise_finder import (
    ExerciseBoundaryRouterModule,
    ExerciseBoundaryRouterSignature,
    ExerciseStartRouterModule,
    ExerciseStartRouterSignature,
)
from kms2.module.source_pedagogical.exercise_splitter import (
    ExerciseSplitterModule,
    ExerciseSplitterSignature,
)
from kms2.module.source_pedagogical.exercise_strip_router import (
    ExerciseStripRouterModule,
    ExerciseStripRouterSignature,
)
from kms2.module.source_pedagogical.instruction_finder import (
    InstructionBoundaryRouterModule,
    InstructionBoundaryRouterSignature,
    InstructionStartRouterModule,
    InstructionStartRouterSignature,
)
from kms2.module.source_pedagogical.instruction_governance import (
    InstructionGovernanceModule,
    InstructionGovernanceSignature,
)
from kms2.module.source_pedagogical.pedagogical_finder import (
    PedagogicalBoundaryRouterModule,
    PedagogicalBoundaryRouterSignature,
    PedagogicalStartRouterModule,
    PedagogicalStartRouterSignature,
)
from kms2.module.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionModule,
    SourceProcedureReconstructionSignature,
)
from kms2.module.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionModule,
    SourceStatementReconstructionSignature,
)
from kms2.module.source_pedagogical.statement_procedure import (
    PedagogicalRoleModule,
    PedagogicalRoleSignature,
    ProcedurePartitionModule,
    ProcedurePartitionSignature,
    StatementPartitionModule,
    StatementPartitionSignature,
)
from kms2.node.source_pedagogical.embedding import (
    SourcePedagogicalEmbeddingNode,
)
from kms2.node.source_pedagogical.exercise_finder import ExerciseFinderNode
from kms2.node.source_pedagogical.exercise_splitter import ExerciseSplitterNode
from kms2.node.source_pedagogical.instruction_finder import (
    InstructionFinderNode,
)
from kms2.node.source_pedagogical.instruction_governance import (
    InstructionGovernanceNode,
)
from kms2.node.source_pedagogical.pedagogical_finder import (
    PedagogicalFinderNode,
)
from kms2.node.source_pedagogical.persistence import (
    SourcePedagogicalPersistenceNode,
)
from kms2.node.source_pedagogical.source_load import (
    SourcePedagogicalSourceLoadNode,
)
from kms2.node.source_pedagogical.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms2.node.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionNode,
)
from kms2.node.source_pedagogical.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms2.node.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionNode,
)
from kms2.node.source_pedagogical.statement_procedure import (
    StatementProcedureNode,
)
from kms2.train.recorder import Recorder


def build_source_pedagogical_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    embedding_client: EmbeddingClient,
    tokenizers: InferenceTokenizers,
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
            token_counters=tokenizers.text_counters(
                source.exercise_splitter.router,
                source.exercise_splitter.splitter,
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
            start_token_counters=tokenizers.text_counters(
                source.instruction_finder.start_router
            ),
            boundary_token_counters=tokenizers.text_counters(
                source.instruction_finder.boundary_router
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
            start_token_counters=tokenizers.text_counters(
                source.exercise_finder.start_router
            ),
            boundary_token_counters=tokenizers.text_counters(
                source.exercise_finder.boundary_router
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
            start_token_counters=tokenizers.text_counters(
                source.pedagogical_finder.start_router
            ),
            boundary_token_counters=tokenizers.text_counters(
                source.pedagogical_finder.boundary_router
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
            token_counters=tokenizers.text_counters(
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
        ),
        source_statement_embedding=SourceStatementEmbeddingNode(
            embedding_client
        ),
        source_procedure_embedding=SourceProcedureEmbeddingNode(
            embedding_client
        ),
        embedding=SourcePedagogicalEmbeddingNode(embedding_client),
        persistence=SourcePedagogicalPersistenceNode(
            SourceGraphRepository(database.session)
        ),
    )
