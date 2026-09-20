"""Dependency composition for the global semantic graph."""

from kms2.composition.predictors import PredictorFactory
from kms2.config.settings import Settings
from kms2.database.client import DatabaseClient
from kms2.database.global_semantic.entity_hub_repository import (
    GlobalEntityHubRepository,
)
from kms2.database.global_semantic.event_hub_repository import (
    GlobalEventHubRepository,
)
from kms2.database.global_semantic.predicate_hub_repository import (
    GlobalPredicateHubRepository,
)
from kms2.database.global_semantic.procedure_hub_repository import (
    GlobalProcedureHubRepository,
)
from kms2.database.global_semantic.statement_hub_repository import (
    GlobalStatementHubRepository,
)
from kms2.langgraph.global_semantic.graph import GlobalSemanticGraph
from kms2.local_models import LocalModelRuntime
from kms2.module.global_semantic.global_entity_hub import (
    GlobalEntityHubModule,
    GlobalEntityHubSignature,
)
from kms2.module.global_semantic.global_entity_hub_judge import (
    GlobalEntityHubJudgeModule,
    GlobalEntityHubJudgeSignature,
)
from kms2.module.global_semantic.global_event_hub import (
    GlobalEventHubModule,
    GlobalEventHubSignature,
)
from kms2.module.global_semantic.global_event_hub_judge import (
    GlobalEventHubJudgeModule,
    GlobalEventHubJudgeSignature,
)
from kms2.module.global_semantic.global_predicate_hub import (
    GlobalPredicateHubModule,
    GlobalPredicateHubSignature,
)
from kms2.module.global_semantic.global_predicate_hub_judge import (
    GlobalPredicateHubJudgeModule,
    GlobalPredicateHubJudgeSignature,
)
from kms2.module.global_semantic.global_procedure_hub import (
    GlobalProcedureHubModule,
    GlobalProcedureHubSignature,
)
from kms2.module.global_semantic.global_procedure_hub_judge import (
    GlobalProcedureHubJudgeModule,
    GlobalProcedureHubJudgeSignature,
)
from kms2.module.global_semantic.global_statement_hub import (
    GlobalStatementHubModule,
    GlobalStatementHubSignature,
)
from kms2.module.global_semantic.global_statement_hub_judge import (
    GlobalStatementHubJudgeModule,
    GlobalStatementHubJudgeSignature,
)
from kms2.node.global_semantic.entity_hub import GlobalEntityHubNode
from kms2.node.global_semantic.entity_hub_persistence import (
    GlobalEntityHubPersistenceNode,
)
from kms2.node.global_semantic.event_hub import GlobalEventHubNode
from kms2.node.global_semantic.event_hub_persistence import (
    GlobalEventHubPersistenceNode,
)
from kms2.node.global_semantic.predicate_hub import GlobalPredicateHubNode
from kms2.node.global_semantic.predicate_hub_persistence import (
    GlobalPredicateHubPersistenceNode,
)
from kms2.node.global_semantic.procedure_hub import GlobalProcedureHubNode
from kms2.node.global_semantic.procedure_hub_persistence import (
    GlobalProcedureHubPersistenceNode,
)
from kms2.node.global_semantic.statement_hub import GlobalStatementHubNode
from kms2.node.global_semantic.statement_hub_persistence import (
    GlobalStatementHubPersistenceNode,
)
from kms2.train.recorder import Recorder


def build_global_semantic_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    recorder: Recorder | None = None,
) -> GlobalSemanticGraph:
    """Compose the global semantic graph from shared runtime resources."""
    global_semantic = settings.global_semantic
    predictors = PredictorFactory(local_models, recorder)

    entity_repository = GlobalEntityHubRepository(database.session)
    entity_hub_module = GlobalEntityHubModule(
        predictors.create(
            GlobalEntityHubModule,
            global_semantic.global_entity_hubs.inference,
            GlobalEntityHubSignature,
        )
    )
    entity_hub_judge = GlobalEntityHubJudgeModule(
        predictors.create(
            GlobalEntityHubJudgeModule,
            global_semantic.global_entity_hubs.judge,
            GlobalEntityHubJudgeSignature,
        )
    )
    entity_hub = GlobalEntityHubNode(
        entity_repository,
        entity_hub_module,
        entity_hub_judge,
        local_models,
        local_models,
        global_semantic.global_entity_hubs,
    )

    event_repository = GlobalEventHubRepository(database.session)
    event_hub_module = GlobalEventHubModule(
        predictors.create(
            GlobalEventHubModule,
            global_semantic.global_event_hubs.inference,
            GlobalEventHubSignature,
        )
    )
    event_hub_judge = GlobalEventHubJudgeModule(
        predictors.create(
            GlobalEventHubJudgeModule,
            global_semantic.global_event_hubs.judge,
            GlobalEventHubJudgeSignature,
        )
    )
    event_hub = GlobalEventHubNode(
        event_repository,
        event_hub_module,
        event_hub_judge,
        local_models,
        local_models,
        global_semantic.global_event_hubs,
    )

    predicate_repository = GlobalPredicateHubRepository(database.session)
    predicate_hub_module = GlobalPredicateHubModule(
        predictors.create(
            GlobalPredicateHubModule,
            global_semantic.global_predicate_hubs.inference,
            GlobalPredicateHubSignature,
        )
    )
    predicate_hub_judge = GlobalPredicateHubJudgeModule(
        predictors.create(
            GlobalPredicateHubJudgeModule,
            global_semantic.global_predicate_hubs.judge,
            GlobalPredicateHubJudgeSignature,
        )
    )
    predicate_hub = GlobalPredicateHubNode(
        predicate_repository,
        predicate_hub_module,
        predicate_hub_judge,
        local_models,
        local_models,
        global_semantic.global_predicate_hubs,
    )

    statement_repository = GlobalStatementHubRepository(database.session)
    statement_hub_module = GlobalStatementHubModule(
        predictors.create(
            GlobalStatementHubModule,
            global_semantic.global_statement_hubs.inference,
            GlobalStatementHubSignature,
        )
    )
    statement_hub_judge = GlobalStatementHubJudgeModule(
        predictors.create(
            GlobalStatementHubJudgeModule,
            global_semantic.global_statement_hubs.judge,
            GlobalStatementHubJudgeSignature,
        )
    )
    statement_hub = GlobalStatementHubNode(
        statement_repository,
        statement_hub_module,
        statement_hub_judge,
        local_models,
        local_models,
        global_semantic.global_statement_hubs,
    )

    procedure_repository = GlobalProcedureHubRepository(database.session)
    procedure_hub_module = GlobalProcedureHubModule(
        predictors.create(
            GlobalProcedureHubModule,
            global_semantic.global_procedure_hubs.inference,
            GlobalProcedureHubSignature,
        )
    )
    procedure_hub_judge = GlobalProcedureHubJudgeModule(
        predictors.create(
            GlobalProcedureHubJudgeModule,
            global_semantic.global_procedure_hubs.judge,
            GlobalProcedureHubJudgeSignature,
        )
    )
    procedure_hub = GlobalProcedureHubNode(
        procedure_repository,
        procedure_hub_module,
        procedure_hub_judge,
        local_models,
        local_models,
        global_semantic.global_procedure_hubs,
    )

    return GlobalSemanticGraph(
        entity_hub,
        GlobalEntityHubPersistenceNode(entity_repository),
        event_hub,
        GlobalEventHubPersistenceNode(event_repository),
        predicate_hub,
        GlobalPredicateHubPersistenceNode(predicate_repository),
        statement_hub,
        GlobalStatementHubPersistenceNode(statement_repository),
        procedure_hub,
        GlobalProcedureHubPersistenceNode(procedure_repository),
    )


__all__ = ['build_global_semantic_graph']
