"""Dependency composition for the KMS2 semantic graph."""

from kms2.composition.predictors import PredictorFactory, build_token_budget
from kms2.composition.reranking import reranker_overhead_tokens
from kms2.config.settings import Settings
from kms2.core.embedding import EmbeddingClient
from kms2.core.reranking import RerankerClient
from kms2.core.windowing import TokenBudget
from kms2.database.client import DatabaseClient
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.database.source.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.database.source_semantic.source_procedure_repository import (
    SourceProcedureRepository,
)
from kms2.database.source_semantic.source_proposition_repository import (
    SourcePropositionRepository,
)
from kms2.database.source_semantic.source_statement_repository import (
    SourceStatementRepository,
)
from kms2.inference.token_counting import InferenceTokenizers
from kms2.langgraph.source_semantic.graph import SourceSemanticGraph
from kms2.local_models.runtime import LocalModelRuntime
from kms2.module.source_semantic.examples.source_entity_description import (
    source_entity_description_examples,
)
from kms2.module.source_semantic.examples.source_entity_hub_judge import (
    source_entity_hub_judge_examples,
)
from kms2.module.source_semantic.examples.source_proposition_extraction import (
    source_proposition_extraction_examples,
)
from kms2.module.source_semantic.source_entity_description import (
    SourceEntityDescriptionModule,
    SourceEntityDescriptionSignature,
)
from kms2.module.source_semantic.source_entity_extraction import (
    SourceEntityExtractionSignature,
    SourceEntityExtractorModule,
)
from kms2.module.source_semantic.source_entity_hub_judge import (
    SourceEntityHubJudgeModule,
    SourceEntityHubJudgeSignature,
)
from kms2.module.source_semantic.source_procedure_hub import (
    SourceProcedureHubModule,
    SourceProcedureHubSignature,
    SourceProcedureHubSummaryMergeModule,
    SourceProcedureHubSummaryMergeSignature,
    SourceProcedureHubSummaryModule,
    SourceProcedureHubSummarySignature,
)
from kms2.module.source_semantic.source_procedure_hub_judge import (
    SourceProcedureHubJudgeModule,
    SourceProcedureHubJudgeSignature,
)
from kms2.module.source_semantic.source_proposition_extraction import (
    SourcePropositionExtractionSignature,
    SourcePropositionExtractorModule,
)
from kms2.module.source_semantic.source_proposition_hub_judge import (
    SourcePropositionHubJudgeModule,
    SourcePropositionHubJudgeSignature,
)
from kms2.module.source_semantic.source_statement_hub import (
    SourceStatementHubModule,
    SourceStatementHubSignature,
    SourceStatementHubSummaryMergeModule,
    SourceStatementHubSummaryMergeSignature,
    SourceStatementHubSummaryModule,
    SourceStatementHubSummarySignature,
)
from kms2.module.source_semantic.source_statement_hub_judge import (
    SourceStatementHubJudgeModule,
    SourceStatementHubJudgeSignature,
)
from kms2.node.source_semantic.source_entity_description import (
    SourceEntityDescriptionNode,
)
from kms2.node.source_semantic.source_entity_description_load import (
    SourceEntityDescriptionLoadNode,
)
from kms2.node.source_semantic.source_entity_embedding import (
    SourceEntityEmbeddingNode,
)
from kms2.node.source_semantic.source_entity_extraction import (
    SourceEntityExtractionNode,
)
from kms2.node.source_semantic.source_entity_extraction_persistence import (
    SourceEntityExtractionPersistenceNode,
)
from kms2.node.source_semantic.source_entity_hub import SourceEntityHubNode
from kms2.node.source_semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
from kms2.node.source_semantic.source_entity_persistence import (
    SourceEntityPersistenceNode,
)
from kms2.node.source_semantic.source_load import SourceSemanticSourceLoadNode
from kms2.node.source_semantic.source_procedure_hub import (
    SourceProcedureHubNode,
)
from kms2.node.source_semantic.source_procedure_hub_persistence import (
    SourceProcedureHubPersistenceNode,
)
from kms2.node.source_semantic.source_proposition_embedding import (
    SourcePropositionEmbeddingNode,
)
from kms2.node.source_semantic.source_proposition_embedding_persistence import (
    SourcePropositionEmbeddingPersistenceNode,
)
from kms2.node.source_semantic.source_proposition_extraction import (
    SourcePropositionExtractionNode,
)
from kms2.node.source_semantic.source_proposition_hub import (
    SourcePropositionHubNode,
)
from kms2.node.source_semantic.source_proposition_hub_persistence import (
    SourcePropositionHubPersistenceNode,
)
from kms2.node.source_semantic.source_proposition_persistence import (
    SourcePropositionPersistenceNode,
)
from kms2.node.source_semantic.source_statement_hub import (
    SourceStatementHubNode,
)
from kms2.node.source_semantic.source_statement_hub_persistence import (
    SourceStatementHubPersistenceNode,
)
from kms2.train.recorder import Recorder


def build_source_semantic_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    embedding_client: EmbeddingClient,
    reranker_client: RerankerClient,
    tokenizers: InferenceTokenizers,
    recorder: Recorder | None = None,
) -> SourceSemanticGraph:
    """Compose the semantic graph from shared runtime resources."""
    semantic = settings.source_semantic
    predictors = PredictorFactory(settings, local_models, recorder)
    reranker_token_counter = tokenizers.reranker
    reranker_overhead = reranker_overhead_tokens(settings)
    source_block_repository = SourceBlockRepository(database.session)
    source_proposition_repository = SourcePropositionRepository(
        database.session
    )
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
    proposition_extractor = SourcePropositionExtractorModule(
        predictors.create(
            SourcePropositionExtractorModule,
            semantic.source_proposition_extraction,
            SourcePropositionExtractionSignature,
            examples=source_proposition_extraction_examples(),
        )
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
    return SourceSemanticGraph(
        source_entity_extraction=SourceEntityExtractionNode(
            entity_extractor,
            semantic.source_entity_context_window,
            token_counters=tokenizers.text_counters(
                semantic.source_entity_extraction
            ),
        ),
        source_entity_extraction_persistence=(
            SourceEntityExtractionPersistenceNode(source_entity_repository)
        ),
        source_entity_description_load=SourceEntityDescriptionLoadNode(
            source_entity_repository,
            entity_description_settings.context_window,
            token_counters=tokenizers.text_counters(
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
        ),
        source_entity_embedding=SourceEntityEmbeddingNode(embedding_client),
        source_entity_persistence=SourceEntityPersistenceNode(
            source_entity_repository
        ),
        source_entity_hub=SourceEntityHubNode(
            source_entity_repository,
            entity_hub_judge,
            reranker_client,
            entity_hub_settings,
            reranker_token_counter=reranker_token_counter,
            reranker_overhead_tokens=reranker_overhead,
            judge_budget=entity_hub_judge_budget,
        ),
        source_entity_hub_persistence=SourceEntityHubPersistenceNode(
            source_entity_repository
        ),
        source_semantic_load=SourceSemanticSourceLoadNode(
            source_block_repository
        ),
        source_proposition_extraction=SourcePropositionExtractionNode(
            proposition_extractor,
            semantic.context_window,
            token_counters=tokenizers.text_counters(
                semantic.source_proposition_extraction
            ),
        ),
        source_proposition_persistence=SourcePropositionPersistenceNode(
            source_proposition_repository
        ),
        source_proposition_embedding=SourcePropositionEmbeddingNode(
            embedding_client
        ),
        source_proposition_embedding_persistence=SourcePropositionEmbeddingPersistenceNode(
            source_proposition_repository
        ),
        source_proposition_hub=SourcePropositionHubNode(
            source_proposition_repository,
            SourcePropositionHubJudgeModule(
                predictors.create(
                    SourcePropositionHubJudgeModule,
                    semantic.source_proposition_hubs.judge,
                    SourcePropositionHubJudgeSignature,
                )
            ),
            reranker_client,
            semantic.source_proposition_hubs,
            reranker_budget=TokenBudget(
                tokenizers.reranker,
                semantic.source_proposition_hubs.reranker_token_budget
                - reranker_overhead,
            ),
            judge_budget=build_token_budget(
                settings,
                tokenizers,
                semantic.source_proposition_hubs.judge,
                input_token_budget=semantic.source_proposition_hubs.judge_token_budget,
                safety_margin_tokens=semantic.source_proposition_hubs.safety_margin_tokens,
            ),
        ),
        source_proposition_hub_persistence=SourcePropositionHubPersistenceNode(
            source_proposition_repository
        ),
        source_statement_hub=SourceStatementHubNode(
            source_statement_repository,
            source_statement_hub_module,
            source_statement_hub_judge,
            reranker_client,
            embedding_client,
            semantic.source_statement_hubs,
            reranker_token_counter=reranker_token_counter,
            reranker_overhead_tokens=reranker_overhead,
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
            reranker_client,
            embedding_client,
            semantic.source_procedure_hubs,
            reranker_token_counter=reranker_token_counter,
            reranker_overhead_tokens=reranker_overhead,
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
