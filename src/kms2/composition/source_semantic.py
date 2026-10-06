"""Dependency composition for the KMS2 semantic graph."""

from kms2.composition.predictors import PredictorFactory, build_token_budget
from kms2.config.settings import Settings
from kms2.database.client import DatabaseClient
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.database.source_semantic.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.database.source_semantic.source_event_repository import (
    SourceEventRepository,
)
from kms2.database.source_semantic.source_fact_repository import (
    SourceFactRepository,
)
from kms2.database.source_semantic.source_predicate_repository import (
    SourcePredicateRepository,
)
from kms2.database.source_semantic.source_procedure_repository import (
    SourceProcedureRepository,
)
from kms2.database.source_semantic.source_statement_repository import (
    SourceStatementRepository,
)
from kms2.database.source_semantic.source_triplet_repository import (
    SourceTripletRepository,
)
from kms2.langgraph.source_semantic.graph import SourceSemanticGraph
from kms2.local_models.runtime import LocalModelRuntime
from kms2.local_models.token_counting import LocalTokenizers
from kms2.module.source_semantic.source_entity_description import (
    SourceEntityDescriptionModule,
    SourceEntityDescriptionSignature,
)
from kms2.module.source_semantic.source_entity_hub import (
    SourceEntityHubModule,
    SourceEntityHubSignature,
    SourceEntityHubSummaryMergeModule,
    SourceEntityHubSummaryMergeSignature,
    SourceEntityHubSummaryModule,
    SourceEntityHubSummarySignature,
)
from kms2.module.source_semantic.source_entity_hub_judge import (
    SourceEntityHubJudgeModule,
    SourceEntityHubJudgeSignature,
)
from kms2.module.source_semantic.source_event_description import (
    SourceEventDescriptionModule,
    SourceEventDescriptionSignature,
)
from kms2.module.source_semantic.source_event_hub import (
    SourceEventHubModule,
    SourceEventHubSignature,
    SourceEventHubSummaryMergeModule,
    SourceEventHubSummaryMergeSignature,
    SourceEventHubSummaryModule,
    SourceEventHubSummarySignature,
)
from kms2.module.source_semantic.source_event_hub_judge import (
    SourceEventHubJudgeModule,
    SourceEventHubJudgeSignature,
)
from kms2.module.source_semantic.source_fact_extraction import (
    SourceFactExtractionSignature,
    SourceFactExtractorModule,
)
from kms2.module.source_semantic.source_predicate_description import (
    SourcePredicateDescriptionModule,
    SourcePredicateDescriptionSignature,
)
from kms2.module.source_semantic.source_predicate_hub import (
    SourcePredicateHubModule,
    SourcePredicateHubSignature,
    SourcePredicateHubSummaryMergeModule,
    SourcePredicateHubSummaryMergeSignature,
    SourcePredicateHubSummaryModule,
    SourcePredicateHubSummarySignature,
)
from kms2.module.source_semantic.source_predicate_hub_judge import (
    SourcePredicateHubJudgeModule,
    SourcePredicateHubJudgeSignature,
)
from kms2.module.source_semantic.source_procedure_description import (
    SourceProcedureDescriptionModule,
    SourceProcedureDescriptionSignature,
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
from kms2.module.source_semantic.source_statement_description import (
    SourceStatementDescriptionModule,
    SourceStatementDescriptionSignature,
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
from kms2.module.source_semantic.source_triplet_decomposition import (
    SourceTripletDecomposerModule,
    SourceTripletDecompositionSignature,
)
from kms2.module.source_semantic.source_triplet_hub import (
    SourceTripletHubModule,
    SourceTripletHubSignature,
    SourceTripletHubSummaryMergeModule,
    SourceTripletHubSummaryMergeSignature,
    SourceTripletHubSummaryModule,
    SourceTripletHubSummarySignature,
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
from kms2.node.source_semantic.source_entity_hub import SourceEntityHubNode
from kms2.node.source_semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
from kms2.node.source_semantic.source_entity_persistence import (
    SourceEntityPersistenceNode,
)
from kms2.node.source_semantic.source_event_description import (
    SourceEventDescriptionNode,
)
from kms2.node.source_semantic.source_event_description_load import (
    SourceEventDescriptionLoadNode,
)
from kms2.node.source_semantic.source_event_embedding import (
    SourceEventEmbeddingNode,
)
from kms2.node.source_semantic.source_event_hub import SourceEventHubNode
from kms2.node.source_semantic.source_event_hub_persistence import (
    SourceEventHubPersistenceNode,
)
from kms2.node.source_semantic.source_event_persistence import (
    SourceEventPersistenceNode,
)
from kms2.node.source_semantic.source_fact_extraction import (
    SourceFactExtractionNode,
)
from kms2.node.source_semantic.source_fact_persistence import (
    SourceFactPersistenceNode,
)
from kms2.node.source_semantic.source_predicate_description import (
    SourcePredicateDescriptionNode,
)
from kms2.node.source_semantic.source_predicate_description_load import (
    SourcePredicateDescriptionLoadNode,
)
from kms2.node.source_semantic.source_predicate_embedding import (
    SourcePredicateEmbeddingNode,
)
from kms2.node.source_semantic.source_predicate_hub import (
    SourcePredicateHubNode,
)
from kms2.node.source_semantic.source_predicate_hub_persistence import (
    SourcePredicateHubPersistenceNode,
)
from kms2.node.source_semantic.source_predicate_persistence import (
    SourcePredicatePersistenceNode,
)
from kms2.node.source_semantic.source_procedure_description import (
    SourceProcedureDescriptionNode,
)
from kms2.node.source_semantic.source_procedure_description_load import (
    SourceProcedureDescriptionLoadNode,
)
from kms2.node.source_semantic.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms2.node.source_semantic.source_procedure_hub import (
    SourceProcedureHubNode,
)
from kms2.node.source_semantic.source_procedure_hub_persistence import (
    SourceProcedureHubPersistenceNode,
)
from kms2.node.source_semantic.source_procedure_persistence import (
    SourceProcedurePersistenceNode,
)
from kms2.node.source_semantic.source_statement_description import (
    SourceStatementDescriptionNode,
)
from kms2.node.source_semantic.source_statement_description_load import (
    SourceStatementDescriptionLoadNode,
)
from kms2.node.source_semantic.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms2.node.source_semantic.source_statement_hub import (
    SourceStatementHubNode,
)
from kms2.node.source_semantic.source_statement_hub_persistence import (
    SourceStatementHubPersistenceNode,
)
from kms2.node.source_semantic.source_statement_persistence import (
    SourceStatementPersistenceNode,
)
from kms2.node.source_semantic.source_triplet_decomposition import (
    SourceTripletDecompositionNode,
)
from kms2.node.source_semantic.source_triplet_fact_load import (
    SourceTripletFactLoadNode,
)
from kms2.node.source_semantic.source_triplet_hub import SourceTripletHubNode
from kms2.node.source_semantic.source_triplet_hub_persistence import (
    SourceTripletHubPersistenceNode,
)
from kms2.node.source_semantic.source_triplet_load import (
    SourceFactSourceLoadNode,
)
from kms2.node.source_semantic.source_triplet_persistence import (
    SourceTripletPersistenceNode,
)
from kms2.train.recorder import Recorder


def build_source_semantic_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    tokenizers: LocalTokenizers,
    recorder: Recorder | None = None,
) -> SourceSemanticGraph:
    """Compose the complete semantic graph from shared runtime resources."""
    semantic = settings.source_semantic
    predictors = PredictorFactory(local_models, recorder)
    reranker_token_counter = tokenizers.reranker
    reranker_overhead_tokens = (
        settings.local_models.reranker.safety_margin_tokens
    )
    source_block_repository = SourceBlockRepository(database.session)
    source_entity_repository = SourceEntityRepository(database.session)
    source_fact_repository = SourceFactRepository(database.session)
    source_event_repository = SourceEventRepository(database.session)
    source_predicate_repository = SourcePredicateRepository(database.session)
    source_procedure_repository = SourceProcedureRepository(database.session)
    source_statement_repository = SourceStatementRepository(database.session)
    source_triplet_repository = SourceTripletRepository(database.session)
    fact_extractor = SourceFactExtractorModule(
        predictors.create(
            SourceFactExtractorModule,
            semantic.source_fact_extraction,
            SourceFactExtractionSignature,
        )
    )
    triplet_decomposer = SourceTripletDecomposerModule(
        predictors.create(
            SourceTripletDecomposerModule,
            semantic.source_triplet_decomposition,
            SourceTripletDecompositionSignature,
        )
    )
    source_entity_hub_budget_settings = (
        semantic.source_entity_hubs.synthesis_budget
    )
    source_entity_hub_summary_inference = semantic.source_entity_hubs.inference.model_copy(
        update={
            'max_tokens': source_entity_hub_budget_settings.summary_completion_tokens
        }
    )
    source_entity_hub_final_predictor, source_entity_hub_final_budget = (
        predictors.create(
            SourceEntityHubModule,
            semantic.source_entity_hubs.inference,
            SourceEntityHubSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            semantic.source_entity_hubs.inference,
            input_token_budget=source_entity_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_entity_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_entity_hub_module = SourceEntityHubModule(
        source_entity_hub_final_predictor
    )
    source_entity_hub_summary_predictor, source_entity_hub_summary_budget = (
        predictors.create(
            SourceEntityHubSummaryModule,
            source_entity_hub_summary_inference,
            SourceEntityHubSummarySignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            source_entity_hub_summary_inference,
            input_token_budget=source_entity_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_entity_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_entity_hub_summary_module = SourceEntityHubSummaryModule(
        source_entity_hub_summary_predictor
    )
    source_entity_hub_merge_predictor, source_entity_hub_merge_budget = (
        predictors.create(
            SourceEntityHubSummaryMergeModule,
            source_entity_hub_summary_inference,
            SourceEntityHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            source_entity_hub_summary_inference,
            input_token_budget=source_entity_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_entity_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_entity_hub_merge_module = SourceEntityHubSummaryMergeModule(
        source_entity_hub_merge_predictor
    )
    source_entity_hub_judge_predictor, source_entity_hub_judge_budget = (
        predictors.create(
            SourceEntityHubJudgeModule,
            semantic.source_entity_hubs.judge,
            SourceEntityHubJudgeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            semantic.source_entity_hubs.judge,
            input_token_budget=semantic.source_entity_hubs.judge_token_budget,
            safety_margin_tokens=source_entity_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_entity_hub_judge = SourceEntityHubJudgeModule(
        source_entity_hub_judge_predictor
    )
    source_event_hub_budget_settings = (
        semantic.source_event_hubs.synthesis_budget
    )
    source_event_hub_summary_inference = semantic.source_event_hubs.inference.model_copy(
        update={
            'max_tokens': source_event_hub_budget_settings.summary_completion_tokens
        }
    )
    source_event_hub_final_predictor, source_event_hub_final_budget = (
        predictors.create(
            SourceEventHubModule,
            semantic.source_event_hubs.inference,
            SourceEventHubSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            semantic.source_event_hubs.inference,
            input_token_budget=source_event_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_event_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_event_hub_module = SourceEventHubModule(
        source_event_hub_final_predictor
    )
    source_event_hub_summary_predictor, source_event_hub_summary_budget = (
        predictors.create(
            SourceEventHubSummaryModule,
            source_event_hub_summary_inference,
            SourceEventHubSummarySignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            source_event_hub_summary_inference,
            input_token_budget=source_event_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_event_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_event_hub_summary_module = SourceEventHubSummaryModule(
        source_event_hub_summary_predictor
    )
    source_event_hub_merge_predictor, source_event_hub_merge_budget = (
        predictors.create(
            SourceEventHubSummaryMergeModule,
            source_event_hub_summary_inference,
            SourceEventHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            source_event_hub_summary_inference,
            input_token_budget=source_event_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_event_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_event_hub_merge_module = SourceEventHubSummaryMergeModule(
        source_event_hub_merge_predictor
    )
    source_event_hub_judge_predictor, source_event_hub_judge_budget = (
        predictors.create(
            SourceEventHubJudgeModule,
            semantic.source_event_hubs.judge,
            SourceEventHubJudgeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            semantic.source_event_hubs.judge,
            input_token_budget=semantic.source_event_hubs.judge_token_budget,
            safety_margin_tokens=source_event_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_event_hub_judge = SourceEventHubJudgeModule(
        source_event_hub_judge_predictor
    )
    source_predicate_hub_budget_settings = (
        semantic.source_predicate_hubs.synthesis_budget
    )
    source_predicate_hub_summary_inference = semantic.source_predicate_hubs.inference.model_copy(
        update={
            'max_tokens': source_predicate_hub_budget_settings.summary_completion_tokens
        }
    )
    source_predicate_hub_final_predictor, source_predicate_hub_final_budget = (
        predictors.create(
            SourcePredicateHubModule,
            semantic.source_predicate_hubs.inference,
            SourcePredicateHubSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            semantic.source_predicate_hubs.inference,
            input_token_budget=source_predicate_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_predicate_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_predicate_hub_module = SourcePredicateHubModule(
        source_predicate_hub_final_predictor
    )
    (
        source_predicate_hub_summary_predictor,
        source_predicate_hub_summary_budget,
    ) = (
        predictors.create(
            SourcePredicateHubSummaryModule,
            source_predicate_hub_summary_inference,
            SourcePredicateHubSummarySignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            source_predicate_hub_summary_inference,
            input_token_budget=source_predicate_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_predicate_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_predicate_hub_summary_module = SourcePredicateHubSummaryModule(
        source_predicate_hub_summary_predictor
    )
    source_predicate_hub_merge_predictor, source_predicate_hub_merge_budget = (
        predictors.create(
            SourcePredicateHubSummaryMergeModule,
            source_predicate_hub_summary_inference,
            SourcePredicateHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            source_predicate_hub_summary_inference,
            input_token_budget=source_predicate_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_predicate_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_predicate_hub_merge_module = SourcePredicateHubSummaryMergeModule(
        source_predicate_hub_merge_predictor
    )
    source_predicate_hub_judge_predictor, source_predicate_hub_judge_budget = (
        predictors.create(
            SourcePredicateHubJudgeModule,
            semantic.source_predicate_hubs.judge,
            SourcePredicateHubJudgeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            semantic.source_predicate_hubs.judge,
            input_token_budget=semantic.source_predicate_hubs.judge_token_budget,
            safety_margin_tokens=source_predicate_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_predicate_hub_judge = SourcePredicateHubJudgeModule(
        source_predicate_hub_judge_predictor
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
            settings.local_models,
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
            settings.local_models,
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
            settings.local_models,
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
            settings.local_models,
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
            settings.local_models,
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
            settings.local_models,
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
            settings.local_models,
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
            settings.local_models,
            tokenizers,
            semantic.source_procedure_hubs.judge,
            input_token_budget=semantic.source_procedure_hubs.judge_token_budget,
            safety_margin_tokens=source_procedure_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_procedure_hub_judge = SourceProcedureHubJudgeModule(
        source_procedure_hub_judge_predictor
    )
    source_triplet_hub_budget_settings = (
        semantic.source_triplet_hubs.synthesis_budget
    )
    source_triplet_hub_summary_inference = semantic.source_triplet_hubs.inference.model_copy(
        update={
            'max_tokens': source_triplet_hub_budget_settings.summary_completion_tokens
        }
    )
    source_triplet_hub_final_predictor, source_triplet_hub_final_budget = (
        predictors.create(
            SourceTripletHubModule,
            semantic.source_triplet_hubs.inference,
            SourceTripletHubSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            semantic.source_triplet_hubs.inference,
            input_token_budget=source_triplet_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_triplet_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_triplet_hub_module = SourceTripletHubModule(
        source_triplet_hub_final_predictor
    )
    source_triplet_hub_summary_predictor, source_triplet_hub_summary_budget = (
        predictors.create(
            SourceTripletHubSummaryModule,
            source_triplet_hub_summary_inference,
            SourceTripletHubSummarySignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            source_triplet_hub_summary_inference,
            input_token_budget=source_triplet_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_triplet_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_triplet_hub_summary_module = SourceTripletHubSummaryModule(
        source_triplet_hub_summary_predictor
    )
    source_triplet_hub_merge_predictor, source_triplet_hub_merge_budget = (
        predictors.create(
            SourceTripletHubSummaryMergeModule,
            source_triplet_hub_summary_inference,
            SourceTripletHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            source_triplet_hub_summary_inference,
            input_token_budget=source_triplet_hub_budget_settings.input_token_budget,
            safety_margin_tokens=source_triplet_hub_budget_settings.safety_margin_tokens,
        ),
    )
    source_triplet_hub_merge_module = SourceTripletHubSummaryMergeModule(
        source_triplet_hub_merge_predictor
    )

    return SourceSemanticGraph(
        source_fact_source_load=SourceFactSourceLoadNode(
            source_block_repository
        ),
        source_fact_extraction=SourceFactExtractionNode(
            fact_extractor,
            semantic.context_window,
            token_counters=tokenizers.text_counters(
                semantic.source_fact_extraction.model_server_profile
            ),
        ),
        source_fact_persistence=SourceFactPersistenceNode(
            source_fact_repository
        ),
        source_triplet_fact_load=SourceTripletFactLoadNode(
            source_fact_repository
        ),
        source_triplet_decomposition=SourceTripletDecompositionNode(
            triplet_decomposer
        ),
        source_triplet_persistence=SourceTripletPersistenceNode(
            source_triplet_repository
        ),
        source_entity_description_load=SourceEntityDescriptionLoadNode(
            source_block_repository,
            source_entity_repository,
            semantic.source_entity_description.context_window,
            token_counters=tokenizers.text_counters(
                semantic.source_entity_description.inference.model_server_profile
            ),
        ),
        source_entity_description=SourceEntityDescriptionNode(
            SourceEntityDescriptionModule(
                predictors.create(
                    SourceEntityDescriptionModule,
                    semantic.source_entity_description.inference,
                    SourceEntityDescriptionSignature,
                )
            )
        ),
        source_entity_embedding=SourceEntityEmbeddingNode(local_models),
        source_entity_persistence=SourceEntityPersistenceNode(
            source_entity_repository
        ),
        source_event_description_load=SourceEventDescriptionLoadNode(
            source_block_repository,
            source_event_repository,
            semantic.source_event_description.context_window,
            token_counters=tokenizers.text_counters(
                semantic.source_event_description.inference.model_server_profile
            ),
        ),
        source_event_description=SourceEventDescriptionNode(
            SourceEventDescriptionModule(
                predictors.create(
                    SourceEventDescriptionModule,
                    semantic.source_event_description.inference,
                    SourceEventDescriptionSignature,
                )
            )
        ),
        source_event_embedding=SourceEventEmbeddingNode(local_models),
        source_event_persistence=SourceEventPersistenceNode(
            source_event_repository
        ),
        source_predicate_description_load=SourcePredicateDescriptionLoadNode(
            source_block_repository,
            source_predicate_repository,
            semantic.source_predicate_description.context_window,
            token_counters=tokenizers.text_counters(
                semantic.source_predicate_description.inference.model_server_profile
            ),
        ),
        source_predicate_description=SourcePredicateDescriptionNode(
            SourcePredicateDescriptionModule(
                predictors.create(
                    SourcePredicateDescriptionModule,
                    semantic.source_predicate_description.inference,
                    SourcePredicateDescriptionSignature,
                )
            )
        ),
        source_predicate_embedding=SourcePredicateEmbeddingNode(local_models),
        source_predicate_persistence=SourcePredicatePersistenceNode(
            source_predicate_repository
        ),
        source_statement_description_load=SourceStatementDescriptionLoadNode(
            source_block_repository,
            source_statement_repository,
            semantic.source_statement_description.context_window,
            token_counters=tokenizers.text_counters(
                semantic.source_statement_description.inference.model_server_profile
            ),
        ),
        source_statement_description=SourceStatementDescriptionNode(
            SourceStatementDescriptionModule(
                predictors.create(
                    SourceStatementDescriptionModule,
                    semantic.source_statement_description.inference,
                    SourceStatementDescriptionSignature,
                )
            )
        ),
        source_statement_embedding=SourceStatementEmbeddingNode(local_models),
        source_statement_persistence=SourceStatementPersistenceNode(
            source_statement_repository
        ),
        source_procedure_description_load=SourceProcedureDescriptionLoadNode(
            source_block_repository,
            source_procedure_repository,
            semantic.source_procedure_description.context_window,
            token_counters=tokenizers.text_counters(
                semantic.source_procedure_description.inference.model_server_profile
            ),
        ),
        source_procedure_description=SourceProcedureDescriptionNode(
            SourceProcedureDescriptionModule(
                predictors.create(
                    SourceProcedureDescriptionModule,
                    semantic.source_procedure_description.inference,
                    SourceProcedureDescriptionSignature,
                )
            )
        ),
        source_procedure_embedding=SourceProcedureEmbeddingNode(local_models),
        source_procedure_persistence=SourceProcedurePersistenceNode(
            source_procedure_repository
        ),
        source_entity_hub=SourceEntityHubNode(
            source_entity_repository,
            source_entity_hub_module,
            source_entity_hub_judge,
            local_models,
            local_models,
            semantic.source_entity_hubs,
            reranker_token_counter=reranker_token_counter,
            reranker_overhead_tokens=reranker_overhead_tokens,
            judge_budget=source_entity_hub_judge_budget,
            summary_module=source_entity_hub_summary_module,
            merge_module=source_entity_hub_merge_module,
            final_budget=source_entity_hub_final_budget,
            summary_budget=source_entity_hub_summary_budget,
            merge_budget=source_entity_hub_merge_budget,
        ),
        source_event_hub=SourceEventHubNode(
            source_event_repository,
            source_event_hub_module,
            source_event_hub_judge,
            local_models,
            local_models,
            semantic.source_event_hubs,
            reranker_token_counter=reranker_token_counter,
            reranker_overhead_tokens=reranker_overhead_tokens,
            judge_budget=source_event_hub_judge_budget,
            summary_module=source_event_hub_summary_module,
            merge_module=source_event_hub_merge_module,
            final_budget=source_event_hub_final_budget,
            summary_budget=source_event_hub_summary_budget,
            merge_budget=source_event_hub_merge_budget,
        ),
        source_predicate_hub=SourcePredicateHubNode(
            source_predicate_repository,
            source_predicate_hub_module,
            source_predicate_hub_judge,
            local_models,
            local_models,
            semantic.source_predicate_hubs,
            reranker_token_counter=reranker_token_counter,
            reranker_overhead_tokens=reranker_overhead_tokens,
            judge_budget=source_predicate_hub_judge_budget,
            summary_module=source_predicate_hub_summary_module,
            merge_module=source_predicate_hub_merge_module,
            final_budget=source_predicate_hub_final_budget,
            summary_budget=source_predicate_hub_summary_budget,
            merge_budget=source_predicate_hub_merge_budget,
        ),
        source_triplet_hub=SourceTripletHubNode(
            source_triplet_repository,
            source_triplet_hub_module,
            local_models,
            summary_module=source_triplet_hub_summary_module,
            merge_module=source_triplet_hub_merge_module,
            final_budget=source_triplet_hub_final_budget,
            summary_budget=source_triplet_hub_summary_budget,
            merge_budget=source_triplet_hub_merge_budget,
        ),
        source_statement_hub=SourceStatementHubNode(
            source_statement_repository,
            source_statement_hub_module,
            source_statement_hub_judge,
            local_models,
            local_models,
            semantic.source_statement_hubs,
            reranker_token_counter=reranker_token_counter,
            reranker_overhead_tokens=reranker_overhead_tokens,
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
            local_models,
            local_models,
            semantic.source_procedure_hubs,
            reranker_token_counter=reranker_token_counter,
            reranker_overhead_tokens=reranker_overhead_tokens,
            judge_budget=source_procedure_hub_judge_budget,
            summary_module=source_procedure_hub_summary_module,
            merge_module=source_procedure_hub_merge_module,
            final_budget=source_procedure_hub_final_budget,
            summary_budget=source_procedure_hub_summary_budget,
            merge_budget=source_procedure_hub_merge_budget,
        ),
        source_entity_hub_persistence=SourceEntityHubPersistenceNode(
            source_entity_repository
        ),
        source_event_hub_persistence=SourceEventHubPersistenceNode(
            source_event_repository
        ),
        source_predicate_hub_persistence=SourcePredicateHubPersistenceNode(
            source_predicate_repository
        ),
        source_statement_hub_persistence=SourceStatementHubPersistenceNode(
            source_statement_repository
        ),
        source_procedure_hub_persistence=SourceProcedureHubPersistenceNode(
            source_procedure_repository
        ),
        source_triplet_hub_persistence=SourceTripletHubPersistenceNode(
            source_triplet_repository
        ),
    )
