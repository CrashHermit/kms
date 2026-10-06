"""Dependency composition for the global semantic graph."""

from kms2.composition.predictors import PredictorFactory, build_token_budget
from kms2.config.settings import Settings
from kms2.database.client import DatabaseClient
from kms2.database.global_semantic.global_entity_hub_repository import (
    GlobalEntityHubRepository,
)
from kms2.database.global_semantic.global_event_hub_repository import (
    GlobalEventHubRepository,
)
from kms2.database.global_semantic.global_predicate_hub_repository import (
    GlobalPredicateHubRepository,
)
from kms2.database.global_semantic.global_procedure_hub_repository import (
    GlobalProcedureHubRepository,
)
from kms2.database.global_semantic.global_statement_hub_repository import (
    GlobalStatementHubRepository,
)
from kms2.database.global_semantic.global_triplet_repository import (
    GlobalTripletRepository,
)
from kms2.langgraph.global_semantic.graph import GlobalSemanticGraph
from kms2.local_models.runtime import LocalModelRuntime
from kms2.local_models.token_counting import LocalTokenizers
from kms2.module.global_semantic.global_entity_hub import (
    GlobalEntityHubModule,
    GlobalEntityHubSignature,
    GlobalEntityHubSummaryMergeModule,
    GlobalEntityHubSummaryMergeSignature,
    GlobalEntityHubSummaryModule,
    GlobalEntityHubSummarySignature,
)
from kms2.module.global_semantic.global_entity_hub_judge import (
    GlobalEntityHubJudgeModule,
    GlobalEntityHubJudgeSignature,
)
from kms2.module.global_semantic.global_event_hub import (
    GlobalEventHubModule,
    GlobalEventHubSignature,
    GlobalEventHubSummaryMergeModule,
    GlobalEventHubSummaryMergeSignature,
    GlobalEventHubSummaryModule,
    GlobalEventHubSummarySignature,
)
from kms2.module.global_semantic.global_event_hub_judge import (
    GlobalEventHubJudgeModule,
    GlobalEventHubJudgeSignature,
)
from kms2.module.global_semantic.global_predicate_hub import (
    GlobalPredicateHubModule,
    GlobalPredicateHubSignature,
    GlobalPredicateHubSummaryMergeModule,
    GlobalPredicateHubSummaryMergeSignature,
    GlobalPredicateHubSummaryModule,
    GlobalPredicateHubSummarySignature,
)
from kms2.module.global_semantic.global_predicate_hub_judge import (
    GlobalPredicateHubJudgeModule,
    GlobalPredicateHubJudgeSignature,
)
from kms2.module.global_semantic.global_procedure_hub import (
    GlobalProcedureHubModule,
    GlobalProcedureHubSignature,
    GlobalProcedureHubSummaryMergeModule,
    GlobalProcedureHubSummaryMergeSignature,
    GlobalProcedureHubSummaryModule,
    GlobalProcedureHubSummarySignature,
)
from kms2.module.global_semantic.global_procedure_hub_judge import (
    GlobalProcedureHubJudgeModule,
    GlobalProcedureHubJudgeSignature,
)
from kms2.module.global_semantic.global_statement_hub import (
    GlobalStatementHubModule,
    GlobalStatementHubSignature,
    GlobalStatementHubSummaryMergeModule,
    GlobalStatementHubSummaryMergeSignature,
    GlobalStatementHubSummaryModule,
    GlobalStatementHubSummarySignature,
)
from kms2.module.global_semantic.global_statement_hub_judge import (
    GlobalStatementHubJudgeModule,
    GlobalStatementHubJudgeSignature,
)
from kms2.module.global_semantic.global_triplet_hub import (
    GlobalTripletHubModule,
    GlobalTripletHubSignature,
    GlobalTripletHubSummaryMergeModule,
    GlobalTripletHubSummaryMergeSignature,
    GlobalTripletHubSummaryModule,
    GlobalTripletHubSummarySignature,
)
from kms2.node.global_semantic.global_entity_hub import GlobalEntityHubNode
from kms2.node.global_semantic.global_entity_hub_persistence import (
    GlobalEntityHubPersistenceNode,
)
from kms2.node.global_semantic.global_event_hub import GlobalEventHubNode
from kms2.node.global_semantic.global_event_hub_persistence import (
    GlobalEventHubPersistenceNode,
)
from kms2.node.global_semantic.global_predicate_hub import (
    GlobalPredicateHubNode,
)
from kms2.node.global_semantic.global_predicate_hub_persistence import (
    GlobalPredicateHubPersistenceNode,
)
from kms2.node.global_semantic.global_procedure_hub import (
    GlobalProcedureHubNode,
)
from kms2.node.global_semantic.global_procedure_hub_persistence import (
    GlobalProcedureHubPersistenceNode,
)
from kms2.node.global_semantic.global_statement_hub import (
    GlobalStatementHubNode,
)
from kms2.node.global_semantic.global_statement_hub_persistence import (
    GlobalStatementHubPersistenceNode,
)
from kms2.node.global_semantic.global_triplet_hub import GlobalTripletHubNode
from kms2.node.global_semantic.global_triplet_hub_persistence import (
    GlobalTripletHubPersistenceNode,
)
from kms2.node.global_semantic.global_triplet_projection import (
    GlobalTripletProjectionNode,
)
from kms2.train.recorder import Recorder


def build_global_semantic_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    tokenizers: LocalTokenizers,
    recorder: Recorder | None = None,
) -> GlobalSemanticGraph:
    """Compose the global semantic graph from shared runtime resources."""
    global_semantic = settings.global_semantic
    predictors = PredictorFactory(local_models, recorder)
    reranker_token_counter = tokenizers.reranker
    reranker_overhead_tokens = (
        settings.local_models.reranker.safety_margin_tokens
    )

    entity_repository = GlobalEntityHubRepository(database.session)
    entity_hub_budget_settings = (
        global_semantic.global_entity_hubs.synthesis_budget
    )
    entity_hub_summary_inference = global_semantic.global_entity_hubs.inference.model_copy(
        update={
            'max_tokens': entity_hub_budget_settings.summary_completion_tokens
        }
    )
    entity_hub_final_predictor, entity_hub_final_budget = (
        predictors.create(
            GlobalEntityHubModule,
            global_semantic.global_entity_hubs.inference,
            GlobalEntityHubSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_entity_hubs.inference,
            input_token_budget=entity_hub_budget_settings.input_token_budget,
            safety_margin_tokens=entity_hub_budget_settings.safety_margin_tokens,
        ),
    )
    entity_hub_module = GlobalEntityHubModule(entity_hub_final_predictor)
    entity_hub_summary_predictor, entity_hub_summary_budget = (
        predictors.create(
            GlobalEntityHubSummaryModule,
            entity_hub_summary_inference,
            GlobalEntityHubSummarySignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            entity_hub_summary_inference,
            input_token_budget=entity_hub_budget_settings.input_token_budget,
            safety_margin_tokens=entity_hub_budget_settings.safety_margin_tokens,
        ),
    )
    entity_hub_summary_module = GlobalEntityHubSummaryModule(
        entity_hub_summary_predictor
    )
    entity_hub_merge_predictor, entity_hub_merge_budget = (
        predictors.create(
            GlobalEntityHubSummaryMergeModule,
            entity_hub_summary_inference,
            GlobalEntityHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            entity_hub_summary_inference,
            input_token_budget=entity_hub_budget_settings.input_token_budget,
            safety_margin_tokens=entity_hub_budget_settings.safety_margin_tokens,
        ),
    )
    entity_hub_merge_module = GlobalEntityHubSummaryMergeModule(
        entity_hub_merge_predictor
    )
    entity_hub_judge_predictor, entity_hub_judge_budget = (
        predictors.create(
            GlobalEntityHubJudgeModule,
            global_semantic.global_entity_hubs.judge,
            GlobalEntityHubJudgeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_entity_hubs.judge,
            input_token_budget=global_semantic.global_entity_hubs.judge_token_budget,
            safety_margin_tokens=entity_hub_budget_settings.safety_margin_tokens,
        ),
    )
    entity_hub_judge = GlobalEntityHubJudgeModule(entity_hub_judge_predictor)
    entity_hub = GlobalEntityHubNode(
        entity_repository,
        entity_hub_module,
        entity_hub_judge,
        local_models,
        local_models,
        global_semantic.global_entity_hubs,
        reranker_token_counter=reranker_token_counter,
        reranker_overhead_tokens=reranker_overhead_tokens,
        judge_budget=entity_hub_judge_budget,
        summary_module=entity_hub_summary_module,
        merge_module=entity_hub_merge_module,
        final_budget=entity_hub_final_budget,
        summary_budget=entity_hub_summary_budget,
        merge_budget=entity_hub_merge_budget,
    )

    event_repository = GlobalEventHubRepository(database.session)
    event_hub_budget_settings = (
        global_semantic.global_event_hubs.synthesis_budget
    )
    event_hub_summary_inference = global_semantic.global_event_hubs.inference.model_copy(
        update={
            'max_tokens': event_hub_budget_settings.summary_completion_tokens
        }
    )
    event_hub_final_predictor, event_hub_final_budget = (
        predictors.create(
            GlobalEventHubModule,
            global_semantic.global_event_hubs.inference,
            GlobalEventHubSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_event_hubs.inference,
            input_token_budget=event_hub_budget_settings.input_token_budget,
            safety_margin_tokens=event_hub_budget_settings.safety_margin_tokens,
        ),
    )
    event_hub_module = GlobalEventHubModule(event_hub_final_predictor)
    event_hub_summary_predictor, event_hub_summary_budget = (
        predictors.create(
            GlobalEventHubSummaryModule,
            event_hub_summary_inference,
            GlobalEventHubSummarySignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            event_hub_summary_inference,
            input_token_budget=event_hub_budget_settings.input_token_budget,
            safety_margin_tokens=event_hub_budget_settings.safety_margin_tokens,
        ),
    )
    event_hub_summary_module = GlobalEventHubSummaryModule(
        event_hub_summary_predictor
    )
    event_hub_merge_predictor, event_hub_merge_budget = (
        predictors.create(
            GlobalEventHubSummaryMergeModule,
            event_hub_summary_inference,
            GlobalEventHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            event_hub_summary_inference,
            input_token_budget=event_hub_budget_settings.input_token_budget,
            safety_margin_tokens=event_hub_budget_settings.safety_margin_tokens,
        ),
    )
    event_hub_merge_module = GlobalEventHubSummaryMergeModule(
        event_hub_merge_predictor
    )
    event_hub_judge_predictor, event_hub_judge_budget = (
        predictors.create(
            GlobalEventHubJudgeModule,
            global_semantic.global_event_hubs.judge,
            GlobalEventHubJudgeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_event_hubs.judge,
            input_token_budget=global_semantic.global_event_hubs.judge_token_budget,
            safety_margin_tokens=event_hub_budget_settings.safety_margin_tokens,
        ),
    )
    event_hub_judge = GlobalEventHubJudgeModule(event_hub_judge_predictor)
    event_hub = GlobalEventHubNode(
        event_repository,
        event_hub_module,
        event_hub_judge,
        local_models,
        local_models,
        global_semantic.global_event_hubs,
        reranker_token_counter=reranker_token_counter,
        reranker_overhead_tokens=reranker_overhead_tokens,
        judge_budget=event_hub_judge_budget,
        summary_module=event_hub_summary_module,
        merge_module=event_hub_merge_module,
        final_budget=event_hub_final_budget,
        summary_budget=event_hub_summary_budget,
        merge_budget=event_hub_merge_budget,
    )

    predicate_repository = GlobalPredicateHubRepository(database.session)
    predicate_hub_budget_settings = (
        global_semantic.global_predicate_hubs.synthesis_budget
    )
    predicate_hub_summary_inference = global_semantic.global_predicate_hubs.inference.model_copy(
        update={
            'max_tokens': predicate_hub_budget_settings.summary_completion_tokens
        }
    )
    predicate_hub_final_predictor, predicate_hub_final_budget = (
        predictors.create(
            GlobalPredicateHubModule,
            global_semantic.global_predicate_hubs.inference,
            GlobalPredicateHubSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_predicate_hubs.inference,
            input_token_budget=predicate_hub_budget_settings.input_token_budget,
            safety_margin_tokens=predicate_hub_budget_settings.safety_margin_tokens,
        ),
    )
    predicate_hub_module = GlobalPredicateHubModule(
        predicate_hub_final_predictor
    )
    predicate_hub_summary_predictor, predicate_hub_summary_budget = (
        predictors.create(
            GlobalPredicateHubSummaryModule,
            predicate_hub_summary_inference,
            GlobalPredicateHubSummarySignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            predicate_hub_summary_inference,
            input_token_budget=predicate_hub_budget_settings.input_token_budget,
            safety_margin_tokens=predicate_hub_budget_settings.safety_margin_tokens,
        ),
    )
    predicate_hub_summary_module = GlobalPredicateHubSummaryModule(
        predicate_hub_summary_predictor
    )
    predicate_hub_merge_predictor, predicate_hub_merge_budget = (
        predictors.create(
            GlobalPredicateHubSummaryMergeModule,
            predicate_hub_summary_inference,
            GlobalPredicateHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            predicate_hub_summary_inference,
            input_token_budget=predicate_hub_budget_settings.input_token_budget,
            safety_margin_tokens=predicate_hub_budget_settings.safety_margin_tokens,
        ),
    )
    predicate_hub_merge_module = GlobalPredicateHubSummaryMergeModule(
        predicate_hub_merge_predictor
    )
    predicate_hub_judge_predictor, predicate_hub_judge_budget = (
        predictors.create(
            GlobalPredicateHubJudgeModule,
            global_semantic.global_predicate_hubs.judge,
            GlobalPredicateHubJudgeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_predicate_hubs.judge,
            input_token_budget=global_semantic.global_predicate_hubs.judge_token_budget,
            safety_margin_tokens=predicate_hub_budget_settings.safety_margin_tokens,
        ),
    )
    predicate_hub_judge = GlobalPredicateHubJudgeModule(
        predicate_hub_judge_predictor
    )
    predicate_hub = GlobalPredicateHubNode(
        predicate_repository,
        predicate_hub_module,
        predicate_hub_judge,
        local_models,
        local_models,
        global_semantic.global_predicate_hubs,
        reranker_token_counter=reranker_token_counter,
        reranker_overhead_tokens=reranker_overhead_tokens,
        judge_budget=predicate_hub_judge_budget,
        summary_module=predicate_hub_summary_module,
        merge_module=predicate_hub_merge_module,
        final_budget=predicate_hub_final_budget,
        summary_budget=predicate_hub_summary_budget,
        merge_budget=predicate_hub_merge_budget,
    )

    triplet_repository = GlobalTripletRepository(database.session)
    triplet_hub_budget_settings = (
        global_semantic.global_triplet_hubs.synthesis_budget
    )
    triplet_hub_summary_inference = global_semantic.global_triplet_hubs.inference.model_copy(
        update={
            'max_tokens': triplet_hub_budget_settings.summary_completion_tokens
        }
    )
    triplet_hub_final_predictor, triplet_hub_final_budget = (
        predictors.create(
            GlobalTripletHubModule,
            global_semantic.global_triplet_hubs.inference,
            GlobalTripletHubSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_triplet_hubs.inference,
            input_token_budget=triplet_hub_budget_settings.input_token_budget,
            safety_margin_tokens=triplet_hub_budget_settings.safety_margin_tokens,
        ),
    )
    triplet_hub_module = GlobalTripletHubModule(triplet_hub_final_predictor)
    triplet_hub_summary_predictor, triplet_hub_summary_budget = (
        predictors.create(
            GlobalTripletHubSummaryModule,
            triplet_hub_summary_inference,
            GlobalTripletHubSummarySignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            triplet_hub_summary_inference,
            input_token_budget=triplet_hub_budget_settings.input_token_budget,
            safety_margin_tokens=triplet_hub_budget_settings.safety_margin_tokens,
        ),
    )
    triplet_hub_summary_module = GlobalTripletHubSummaryModule(
        triplet_hub_summary_predictor
    )
    triplet_hub_merge_predictor, triplet_hub_merge_budget = (
        predictors.create(
            GlobalTripletHubSummaryMergeModule,
            triplet_hub_summary_inference,
            GlobalTripletHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            triplet_hub_summary_inference,
            input_token_budget=triplet_hub_budget_settings.input_token_budget,
            safety_margin_tokens=triplet_hub_budget_settings.safety_margin_tokens,
        ),
    )
    triplet_hub_merge_module = GlobalTripletHubSummaryMergeModule(
        triplet_hub_merge_predictor
    )
    triplet_projection = GlobalTripletProjectionNode(triplet_repository)
    triplet_hub = GlobalTripletHubNode(
        triplet_repository,
        triplet_hub_module,
        local_models,
        summary_module=triplet_hub_summary_module,
        merge_module=triplet_hub_merge_module,
        final_budget=triplet_hub_final_budget,
        summary_budget=triplet_hub_summary_budget,
        merge_budget=triplet_hub_merge_budget,
    )
    triplet_hub_persistence = GlobalTripletHubPersistenceNode(
        triplet_repository
    )

    statement_repository = GlobalStatementHubRepository(database.session)
    statement_hub_budget_settings = (
        global_semantic.global_statement_hubs.synthesis_budget
    )
    statement_hub_summary_inference = global_semantic.global_statement_hubs.inference.model_copy(
        update={
            'max_tokens': statement_hub_budget_settings.summary_completion_tokens
        }
    )
    statement_hub_final_predictor, statement_hub_final_budget = (
        predictors.create(
            GlobalStatementHubModule,
            global_semantic.global_statement_hubs.inference,
            GlobalStatementHubSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_statement_hubs.inference,
            input_token_budget=statement_hub_budget_settings.input_token_budget,
            safety_margin_tokens=statement_hub_budget_settings.safety_margin_tokens,
        ),
    )
    statement_hub_module = GlobalStatementHubModule(
        statement_hub_final_predictor
    )
    statement_hub_summary_predictor, statement_hub_summary_budget = (
        predictors.create(
            GlobalStatementHubSummaryModule,
            statement_hub_summary_inference,
            GlobalStatementHubSummarySignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            statement_hub_summary_inference,
            input_token_budget=statement_hub_budget_settings.input_token_budget,
            safety_margin_tokens=statement_hub_budget_settings.safety_margin_tokens,
        ),
    )
    statement_hub_summary_module = GlobalStatementHubSummaryModule(
        statement_hub_summary_predictor
    )
    statement_hub_merge_predictor, statement_hub_merge_budget = (
        predictors.create(
            GlobalStatementHubSummaryMergeModule,
            statement_hub_summary_inference,
            GlobalStatementHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            statement_hub_summary_inference,
            input_token_budget=statement_hub_budget_settings.input_token_budget,
            safety_margin_tokens=statement_hub_budget_settings.safety_margin_tokens,
        ),
    )
    statement_hub_merge_module = GlobalStatementHubSummaryMergeModule(
        statement_hub_merge_predictor
    )
    statement_hub_judge_predictor, statement_hub_judge_budget = (
        predictors.create(
            GlobalStatementHubJudgeModule,
            global_semantic.global_statement_hubs.judge,
            GlobalStatementHubJudgeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_statement_hubs.judge,
            input_token_budget=global_semantic.global_statement_hubs.judge_token_budget,
            safety_margin_tokens=statement_hub_budget_settings.safety_margin_tokens,
        ),
    )
    statement_hub_judge = GlobalStatementHubJudgeModule(
        statement_hub_judge_predictor
    )
    statement_hub = GlobalStatementHubNode(
        statement_repository,
        statement_hub_module,
        statement_hub_judge,
        local_models,
        local_models,
        global_semantic.global_statement_hubs,
        reranker_token_counter=reranker_token_counter,
        reranker_overhead_tokens=reranker_overhead_tokens,
        judge_budget=statement_hub_judge_budget,
        summary_module=statement_hub_summary_module,
        merge_module=statement_hub_merge_module,
        final_budget=statement_hub_final_budget,
        summary_budget=statement_hub_summary_budget,
        merge_budget=statement_hub_merge_budget,
    )

    procedure_repository = GlobalProcedureHubRepository(database.session)
    procedure_hub_budget_settings = (
        global_semantic.global_procedure_hubs.synthesis_budget
    )
    procedure_hub_summary_inference = global_semantic.global_procedure_hubs.inference.model_copy(
        update={
            'max_tokens': procedure_hub_budget_settings.summary_completion_tokens
        }
    )
    procedure_hub_final_predictor, procedure_hub_final_budget = (
        predictors.create(
            GlobalProcedureHubModule,
            global_semantic.global_procedure_hubs.inference,
            GlobalProcedureHubSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_procedure_hubs.inference,
            input_token_budget=procedure_hub_budget_settings.input_token_budget,
            safety_margin_tokens=procedure_hub_budget_settings.safety_margin_tokens,
        ),
    )
    procedure_hub_module = GlobalProcedureHubModule(
        procedure_hub_final_predictor
    )
    procedure_hub_summary_predictor, procedure_hub_summary_budget = (
        predictors.create(
            GlobalProcedureHubSummaryModule,
            procedure_hub_summary_inference,
            GlobalProcedureHubSummarySignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            procedure_hub_summary_inference,
            input_token_budget=procedure_hub_budget_settings.input_token_budget,
            safety_margin_tokens=procedure_hub_budget_settings.safety_margin_tokens,
        ),
    )
    procedure_hub_summary_module = GlobalProcedureHubSummaryModule(
        procedure_hub_summary_predictor
    )
    procedure_hub_merge_predictor, procedure_hub_merge_budget = (
        predictors.create(
            GlobalProcedureHubSummaryMergeModule,
            procedure_hub_summary_inference,
            GlobalProcedureHubSummaryMergeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            procedure_hub_summary_inference,
            input_token_budget=procedure_hub_budget_settings.input_token_budget,
            safety_margin_tokens=procedure_hub_budget_settings.safety_margin_tokens,
        ),
    )
    procedure_hub_merge_module = GlobalProcedureHubSummaryMergeModule(
        procedure_hub_merge_predictor
    )
    procedure_hub_judge_predictor, procedure_hub_judge_budget = (
        predictors.create(
            GlobalProcedureHubJudgeModule,
            global_semantic.global_procedure_hubs.judge,
            GlobalProcedureHubJudgeSignature,
        ),
        build_token_budget(
            settings.local_models,
            tokenizers,
            global_semantic.global_procedure_hubs.judge,
            input_token_budget=global_semantic.global_procedure_hubs.judge_token_budget,
            safety_margin_tokens=procedure_hub_budget_settings.safety_margin_tokens,
        ),
    )
    procedure_hub_judge = GlobalProcedureHubJudgeModule(
        procedure_hub_judge_predictor
    )
    procedure_hub = GlobalProcedureHubNode(
        procedure_repository,
        procedure_hub_module,
        procedure_hub_judge,
        local_models,
        local_models,
        global_semantic.global_procedure_hubs,
        reranker_token_counter=reranker_token_counter,
        reranker_overhead_tokens=reranker_overhead_tokens,
        judge_budget=procedure_hub_judge_budget,
        summary_module=procedure_hub_summary_module,
        merge_module=procedure_hub_merge_module,
        final_budget=procedure_hub_final_budget,
        summary_budget=procedure_hub_summary_budget,
        merge_budget=procedure_hub_merge_budget,
    )

    return GlobalSemanticGraph(
        entity_hub,
        GlobalEntityHubPersistenceNode(entity_repository),
        event_hub,
        GlobalEventHubPersistenceNode(event_repository),
        predicate_hub,
        GlobalPredicateHubPersistenceNode(predicate_repository),
        triplet_projection,
        triplet_hub,
        triplet_hub_persistence,
        statement_hub,
        GlobalStatementHubPersistenceNode(statement_repository),
        procedure_hub,
        GlobalProcedureHubPersistenceNode(procedure_repository),
    )
