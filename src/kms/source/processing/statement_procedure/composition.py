"""Construction for statement and procedure partitioning."""

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.statement_procedure.examples import (
    pedagogical_role_examples,
    procedure_partition_examples,
    statement_partition_examples,
)
from kms.source.processing.statement_procedure.inference import (
    PedagogicalRoleModule,
    PedagogicalRoleSignature,
    ProcedurePartitionModule,
    ProcedurePartitionSignature,
    StatementPartitionModule,
    StatementPartitionSignature,
)
from kms.source.processing.statement_procedure.node import (
    StatementProcedurePartitionNode,
)


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> StatementProcedurePartitionNode:
    source = settings.source_processing
    stage = source.statement_procedure
    image_budget = ImagePayloadBudget(**stage.image_payload.model_dump())
    role_examples = pedagogical_role_examples()
    role_budget = build_token_budget(
        settings,
        tokenizers,
        stage.role_typer,
        input_token_budget=stage.role_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=PedagogicalRoleSignature,
        examples=role_examples,
    )
    statement_examples = statement_partition_examples()
    statement_budget = build_token_budget(
        settings,
        tokenizers,
        stage.statement_partitioner,
        input_token_budget=stage.statement_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=StatementPartitionSignature,
        examples=statement_examples,
    )
    procedure_examples = procedure_partition_examples()
    procedure_budget = build_token_budget(
        settings,
        tokenizers,
        stage.procedure_partitioner,
        input_token_budget=stage.procedure_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=ProcedurePartitionSignature,
        examples=procedure_examples,
    )
    role = PedagogicalRoleModule(
        predictors.create(
            PedagogicalRoleModule,
            stage.role_typer,
            PedagogicalRoleSignature,
            examples=role_examples,
            budget=role_budget,
            image_budget=image_budget,
        )
    )
    statement = StatementPartitionModule(
        predictors.create(
            StatementPartitionModule,
            stage.statement_partitioner,
            StatementPartitionSignature,
            examples=statement_examples,
            budget=statement_budget,
            image_budget=image_budget,
        )
    )
    procedure = ProcedurePartitionModule(
        predictors.create(
            ProcedurePartitionModule,
            stage.procedure_partitioner,
            ProcedurePartitionSignature,
            examples=procedure_examples,
            budget=procedure_budget,
            image_budget=image_budget,
        )
    )
    return StatementProcedurePartitionNode(
        role,
        statement,
        procedure,
        stage,
        role_budget=role_budget,
        statement_budget=statement_budget,
        procedure_budget=procedure_budget,
        image_budget=image_budget,
    )
