"""Dependency composition for the KMS2 semantic graph."""

from kms.composition.predictors import PredictorFactory, build_token_budget
from kms.composition.tokenization import LanguageModelTokenizers
from kms.config.settings import Settings
from kms.core.embedding import Embedder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.reranking import Reranker
from kms.database.client import DatabaseClient
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.database.source.source_entity_repository import (
    SourceEntityRepository,
)
from kms.database.source_semantic.source_procedure_repository import (
    SourceProcedureRepository,
)
from kms.database.source_semantic.source_statement_repository import (
    SourceStatementRepository,
)
from kms.langgraph.source_semantic.entity_hub.graph import SourceEntityHubGraph
from kms.langgraph.source_semantic.graph import SourceSemanticGraph
from kms.local_models.runtime import LocalModelRuntime
from kms.module.source_semantic.examples.source_entity_description import (
    source_entity_description_examples,
)
from kms.module.source_semantic.examples.source_entity_hub_judge import (
    source_entity_hub_judge_examples,
)
from kms.module.source_semantic.source_entity_description import (
    SourceEntityDescriptionModule,
    SourceEntityDescriptionSignature,
)
from kms.module.source_semantic.source_entity_extraction import (
    SourceEntityExtractionSignature,
    SourceEntityExtractorModule,
)
from kms.module.source_semantic.source_entity_hub_judge import (
    SourceEntityHubJudgeModule,
    SourceEntityHubJudgeSignature,
)
from kms.module.source_semantic.source_procedure_hub import (
    SourceProcedureHubModule,
    SourceProcedureHubSignature,
    SourceProcedureHubSummaryMergeModule,
    SourceProcedureHubSummaryMergeSignature,
    SourceProcedureHubSummaryModule,
    SourceProcedureHubSummarySignature,
)
from kms.module.source_semantic.source_procedure_hub_judge import (
    SourceProcedureHubJudgeModule,
    SourceProcedureHubJudgeSignature,
)
from kms.module.source_semantic.source_statement_hub import (
    SourceStatementHubModule,
    SourceStatementHubSignature,
    SourceStatementHubSummaryMergeModule,
    SourceStatementHubSummaryMergeSignature,
    SourceStatementHubSummaryModule,
    SourceStatementHubSummarySignature,
)
from kms.module.source_semantic.source_statement_hub_judge import (
    SourceStatementHubJudgeModule,
    SourceStatementHubJudgeSignature,
)
from kms.node.source_semantic.source_entity_description import (
    SourceEntityDescriptionNode,
)
from kms.node.source_semantic.source_entity_description_load import (
    SourceEntityDescriptionLoadNode,
)
from kms.node.source_semantic.source_entity_embedding import (
    SourceEntityEmbeddingNode,
)
from kms.node.source_semantic.source_entity_extraction import (
    SourceEntityExtractionNode,
)
from kms.node.source_semantic.source_entity_extraction_persistence import (
    SourceEntityExtractionPersistenceNode,
)
from kms.node.source_semantic.source_entity_hub import SourceEntityHubNode
from kms.node.source_semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
from kms.node.source_semantic.source_entity_hub_reconciliation import (
    SourceEntityHubReconciliationNode,
)
from kms.node.source_semantic.source_entity_persistence import (
    SourceEntityPersistenceNode,
)
from kms.node.source_semantic.source_load import SourceSemanticSourceLoadNode
from kms.node.source_semantic.source_procedure_hub import (
    SourceProcedureHubNode,
)
from kms.node.source_semantic.source_procedure_hub_persistence import (
    SourceProcedureHubPersistenceNode,
)
from kms.node.source_semantic.source_statement_hub import (
    SourceStatementHubNode,
)
from kms.node.source_semantic.source_statement_hub_persistence import (
    SourceStatementHubPersistenceNode,
)
from kms.train.recorder import Recorder


def build_source_semantic_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    embedder: Embedder,
    reranker: Reranker,
    tokenizers: LanguageModelTokenizers,
    recorder: Recorder | None = None,
) -> SourceSemanticGraph:
    """Compose the semantic graph from shared runtime resources."""
    semantic = settings.source_semantic
    predictors = PredictorFactory(settings, local_models, recorder)
    source_block_repository = SourceBlockRepository(database.session)
    source_procedure_repository = SourceProcedureRepository(database.session)
    source_statement_repository = SourceStatementRepository(database.session)
    source_entity_repository = SourceEntityRepository(database.session)
    entity_extractor = SourceEntityExtractorModule(
        predictors.create(
            SourceEntityExtractorModule,
            semantic.source_entity_extraction,
            SourceEntityExtractionSignature,
        )
    )
    entity_description_settings = semantic.source_entity_description
    entity_describer = SourceEntityDescriptionModule(
        predictors.create(
            SourceEntityDescriptionModule,
            entity_description_settings.inference,
            SourceEntityDescriptionSignature,
            examples=source_entity_description_examples(),
        )
    )
    entity_hub_settings = semantic.source_entity_hubs
    entity_hub_judge = SourceEntityHubJudgeModule(
        predictors.create(
            SourceEntityHubJudgeModule,
            entity_hub_settings.judge,
            SourceEntityHubJudgeSignature,
            examples=source_entity_hub_judge_examples(),
        )
    )
    entity_hub_judge_budget = build_token_budget(
        settings,
        tokenizers,
        entity_hub_settings.judge,
        input_token_budget=entity_hub_settings.judge_token_budget,
        safety_margin_tokens=entity_hub_settings.safety_margin_tokens,
    )
    source_statement_hub_budget_settings = (
        semantic.source_statement_hubs.synthesis_budget
    )
    source_statement_hub_summary_inference = semantic.source_statement_hubs.inference.model_copy(
        update={
            'max_tokens': source_statement_hub_budget_settings.summary_completion_tokens
        }
    )
    source_statement_hub_final_predictor, source_statement_hub_final_budget = (
        predictors.create(
            SourceStatementHubModule,
            semantic.source_statement_hubs.inference,
            SourceStatementHubSignature,
        ),
        build_token_budget(
            settings,
            tokenizers,
            semantic.source_statement_hubs.inference,
            input_token_budget=source_statement_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_statement_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_statement_hub_module = SourceStatementHubModule(
        source_statement_hub_final_predictor
    )
    (
        source_statement_hub_summary_predictor,
        source_statement_hub_summary_budget,
    ) = (
        predictors.create(
            SourceStatementHubSummaryModule,
            source_statement_hub_summary_inference,
            SourceStatementHubSummarySignature,
        ),
        build_token_budget(
            settings,
            tokenizers,
            source_statement_hub_summary_inference,
            input_token_budget=source_statement_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_statement_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_statement_hub_summary_module = SourceStatementHubSummaryModule(
        source_statement_hub_summary_predictor
    )
    source_statement_hub_merge_predictor, source_statement_hub_merge_budget = (
        predictors.create(
            SourceStatementHubSummaryMergeModule,
            source_statement_hub_summary_inference,
            SourceStatementHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings,
            tokenizers,
            source_statement_hub_summary_inference,
            input_token_budget=source_statement_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_statement_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_statement_hub_merge_module = SourceStatementHubSummaryMergeModule(
        source_statement_hub_merge_predictor
    )
    source_statement_hub_judge_predictor, source_statement_hub_judge_budget = (
        predictors.create(
            SourceStatementHubJudgeModule,
            semantic.source_statement_hubs.judge,
            SourceStatementHubJudgeSignature,
        ),
        build_token_budget(
            settings,
            tokenizers,
            semantic.source_statement_hubs.judge,
            input_token_budget=semantic.source_statement_hubs.judge_token_budget,
            safety_margin_tokens=source_statement_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_statement_hub_judge = SourceStatementHubJudgeModule(
        source_statement_hub_judge_predictor
    )
    source_procedure_hub_budget_settings = (
        semantic.source_procedure_hubs.synthesis_budget
    )
    source_procedure_hub_summary_inference = semantic.source_procedure_hubs.inference.model_copy(
        update={
            'max_tokens': source_procedure_hub_budget_settings.summary_completion_tokens
        }
    )
    source_procedure_hub_final_predictor, source_procedure_hub_final_budget = (
        predictors.create(
            SourceProcedureHubModule,
            semantic.source_procedure_hubs.inference,
            SourceProcedureHubSignature,
        ),
        build_token_budget(
            settings,
            tokenizers,
            semantic.source_procedure_hubs.inference,
            input_token_budget=source_procedure_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_procedure_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_procedure_hub_module = SourceProcedureHubModule(
        source_procedure_hub_final_predictor
    )
    (
        source_procedure_hub_summary_predictor,
        source_procedure_hub_summary_budget,
    ) = (
        predictors.create(
            SourceProcedureHubSummaryModule,
            source_procedure_hub_summary_inference,
            SourceProcedureHubSummarySignature,
        ),
        build_token_budget(
            settings,
            tokenizers,
            source_procedure_hub_summary_inference,
            input_token_budget=source_procedure_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_procedure_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_procedure_hub_summary_module = SourceProcedureHubSummaryModule(
        source_procedure_hub_summary_predictor
    )
    source_procedure_hub_merge_predictor, source_procedure_hub_merge_budget = (
        predictors.create(
            SourceProcedureHubSummaryMergeModule,
            source_procedure_hub_summary_inference,
            SourceProcedureHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings,
            tokenizers,
            source_procedure_hub_summary_inference,
            input_token_budget=source_procedure_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_procedure_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_procedure_hub_merge_module = SourceProcedureHubSummaryMergeModule(
        source_procedure_hub_merge_predictor
    )
    source_procedure_hub_judge_predictor, source_procedure_hub_judge_budget = (
        predictors.create(
            SourceProcedureHubJudgeModule,
            semantic.source_procedure_hubs.judge,
            SourceProcedureHubJudgeSignature,
        ),
        build_token_budget(
            settings,
            tokenizers,
            semantic.source_procedure_hubs.judge,
            input_token_budget=semantic.source_procedure_hubs.judge_token_budget,
            safety_margin_tokens=source_procedure_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_procedure_hub_judge = SourceProcedureHubJudgeModule(
        source_procedure_hub_judge_predictor
    )
    entity_hub = SourceEntityHubNode(
        source_entity_repository,
        entity_hub_judge,
        reranker,
        entity_hub_settings,
        judge_budget=entity_hub_judge_budget,
        image_budget=ImagePayloadBudget(
            max_images=entity_hub_settings.image_payload.max_images,
            max_bytes=entity_hub_settings.image_payload.max_bytes,
            max_pixels=entity_hub_settings.image_payload.max_pixels,
        ),
    )
    return SourceSemanticGraph(
        source_entity_extraction=SourceEntityExtractionNode(
            entity_extractor,
            semantic.source_entity_context_window,
            token_counters=tokenizers.counters(
                semantic.source_entity_extraction
            ),
            image_budget=ImagePayloadBudget(
                max_images=semantic.source_entity_image_payload.max_images,
                max_bytes=semantic.source_entity_image_payload.max_bytes,
                max_pixels=semantic.source_entity_image_payload.max_pixels,
            ),
        ),
        source_entity_extraction_persistence=(
            SourceEntityExtractionPersistenceNode(source_entity_repository)
        ),
        source_entity_description_load=SourceEntityDescriptionLoadNode(
            source_entity_repository,
            entity_description_settings.context_window,
            token_counters=tokenizers.counters(
                entity_description_settings.inference
            ),
        ),
        source_entity_description=SourceEntityDescriptionNode(
            entity_describer,
            budget=build_token_budget(
                settings,
                tokenizers,
                entity_description_settings.inference,
                input_token_budget=(
                    entity_description_settings.input_token_budget
                ),
                safety_margin_tokens=(
                    entity_description_settings.safety_margin_tokens
                ),
            ),
            batch_size=entity_description_settings.batch_size,
            image_budget=ImagePayloadBudget(
                max_images=entity_description_settings.image_payload.max_images,
                max_bytes=entity_description_settings.image_payload.max_bytes,
                max_pixels=entity_description_settings.image_payload.max_pixels,
            ),
        ),
        source_entity_embedding=SourceEntityEmbeddingNode(embedder),
        source_entity_persistence=SourceEntityPersistenceNode(
            source_entity_repository
        ),
        source_entity_hub_reconciliation=SourceEntityHubReconciliationNode(
            entity_hub,
            SourceEntityHubGraph(entity_hub),
        ),
        source_entity_hub_persistence=SourceEntityHubPersistenceNode(
            source_entity_repository
        ),
        source_semantic_load=SourceSemanticSourceLoadNode(
            source_block_repository
        ),
        source_statement_hub=SourceStatementHubNode(
            source_statement_repository,
            source_statement_hub_module,
            source_statement_hub_judge,
            reranker,
            embedder,
            semantic.source_statement_hubs,
            judge_budget=source_statement_hub_judge_budget,
            summary_module=source_statement_hub_summary_module,
            merge_module=source_statement_hub_merge_module,
            final_budget=source_statement_hub_final_budget,
            summary_budget=source_statement_hub_summary_budget,
            merge_budget=source_statement_hub_merge_budget,
        ),
        source_procedure_hub=SourceProcedureHubNode(
            source_procedure_repository,
            source_procedure_hub_module,
            source_procedure_hub_judge,
            reranker,
            embedder,
            semantic.source_procedure_hubs,
            judge_budget=source_procedure_hub_judge_budget,
            summary_module=source_procedure_hub_summary_module,
            merge_module=source_procedure_hub_merge_module,
            final_budget=source_procedure_hub_final_budget,
            summary_budget=source_procedure_hub_summary_budget,
            merge_budget=source_procedure_hub_merge_budget,
        ),
        source_statement_hub_persistence=SourceStatementHubPersistenceNode(
            source_statement_repository
        ),
        source_procedure_hub_persistence=SourceProcedureHubPersistenceNode(
            source_procedure_repository
        ),
    )
