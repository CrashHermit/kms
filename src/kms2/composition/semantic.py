"""Dependency composition for the KMS2 semantic graph."""

from kms2.composition.predictors import PredictorFactory
from kms2.config.settings import Settings
from kms2.database.client import DatabaseClient
from kms2.database.semantic.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.database.semantic.source_event_repository import SourceEventRepository
from kms2.database.semantic.source_predicate_repository import (
    SourcePredicateRepository,
)
from kms2.database.semantic.source_procedure_repository import (
    SourceProcedureRepository,
)
from kms2.database.semantic.source_statement_repository import (
    SourceStatementRepository,
)
from kms2.database.semantic.source_triplet_repository import (
    SourceTripletRepository,
)
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.langgraph.semantic.graph import SemanticGraph
from kms2.local_models import LocalModelRuntime
from kms2.module.semantic.fact_extraction import (
    FactExtractionSignature,
    FactExtractorModule,
)
from kms2.module.semantic.source_entity_description import (
    SourceEntityDescriptionModule,
    SourceEntityDescriptionSignature,
)
from kms2.module.semantic.source_entity_hub import (
    SourceEntityHubModule,
    SourceEntityHubSignature,
)
from kms2.module.semantic.source_entity_hub_judge import (
    SourceEntityHubJudgeModule,
    SourceEntityHubJudgeSignature,
)
from kms2.module.semantic.source_event_description import (
    SourceEventDescriptionModule,
    SourceEventDescriptionSignature,
)
from kms2.module.semantic.source_event_hub import (
    SourceEventHubModule,
    SourceEventHubSignature,
)
from kms2.module.semantic.source_event_hub_judge import (
    SourceEventHubJudgeModule,
    SourceEventHubJudgeSignature,
)
from kms2.module.semantic.source_predicate_description import (
    SourcePredicateDescriptionModule,
    SourcePredicateDescriptionSignature,
)
from kms2.module.semantic.source_predicate_hub import (
    SourcePredicateHubModule,
    SourcePredicateHubSignature,
)
from kms2.module.semantic.source_predicate_hub_judge import (
    SourcePredicateHubJudgeModule,
    SourcePredicateHubJudgeSignature,
)
from kms2.module.semantic.source_procedure_description import (
    SourceProcedureDescriptionModule,
    SourceProcedureDescriptionSignature,
)
from kms2.module.semantic.source_procedure_hub import (
    SourceProcedureHubModule,
    SourceProcedureHubSignature,
)
from kms2.module.semantic.source_procedure_hub_judge import (
    SourceProcedureHubJudgeModule,
    SourceProcedureHubJudgeSignature,
)
from kms2.module.semantic.source_statement_description import (
    SourceStatementDescriptionModule,
    SourceStatementDescriptionSignature,
)
from kms2.module.semantic.source_statement_hub import (
    SourceStatementHubModule,
    SourceStatementHubSignature,
)
from kms2.module.semantic.source_statement_hub_judge import (
    SourceStatementHubJudgeModule,
    SourceStatementHubJudgeSignature,
)
from kms2.module.semantic.source_triplet_hub import (
    SourceTripletHubModule,
    SourceTripletHubSignature,
)
from kms2.module.semantic.triplet_decomposition import (
    TripletDecomposerModule,
    TripletDecompositionSignature,
)
from kms2.node.semantic.fact_extraction import FactExtractionNode
from kms2.node.semantic.source_entity_description import (
    SourceEntityDescriptionNode,
)
from kms2.node.semantic.source_entity_description_load import (
    SourceEntityDescriptionLoadNode,
)
from kms2.node.semantic.source_entity_embedding import SourceEntityEmbeddingNode
from kms2.node.semantic.source_entity_hub import SourceEntityHubNode
from kms2.node.semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
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
from kms2.node.semantic.source_event_hub import SourceEventHubNode
from kms2.node.semantic.source_event_hub_persistence import (
    SourceEventHubPersistenceNode,
)
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
from kms2.node.semantic.source_predicate_hub import SourcePredicateHubNode
from kms2.node.semantic.source_predicate_hub_persistence import (
    SourcePredicateHubPersistenceNode,
)
from kms2.node.semantic.source_predicate_persistence import (
    SourcePredicatePersistenceNode,
)
from kms2.node.semantic.source_procedure_description import (
    SourceProcedureDescriptionNode,
)
from kms2.node.semantic.source_procedure_description_load import (
    SourceProcedureDescriptionLoadNode,
)
from kms2.node.semantic.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms2.node.semantic.source_procedure_hub import SourceProcedureHubNode
from kms2.node.semantic.source_procedure_hub_persistence import (
    SourceProcedureHubPersistenceNode,
)
from kms2.node.semantic.source_procedure_persistence import (
    SourceProcedurePersistenceNode,
)
from kms2.node.semantic.source_statement_description import (
    SourceStatementDescriptionNode,
)
from kms2.node.semantic.source_statement_description_load import (
    SourceStatementDescriptionLoadNode,
)
from kms2.node.semantic.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms2.node.semantic.source_statement_hub import SourceStatementHubNode
from kms2.node.semantic.source_statement_hub_persistence import (
    SourceStatementHubPersistenceNode,
)
from kms2.node.semantic.source_statement_persistence import (
    SourceStatementPersistenceNode,
)
from kms2.node.semantic.source_triplet_hub import SourceTripletHubNode
from kms2.node.semantic.source_triplet_hub_persistence import (
    SourceTripletHubPersistenceNode,
)
from kms2.node.semantic.triplet_decomposition import TripletDecompositionNode
from kms2.node.semantic.triplet_load import TripletSourceLoadNode
from kms2.node.semantic.triplet_persistence import TripletPersistenceNode
from kms2.train.recorder import Recorder


def build_semantic_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    recorder: Recorder | None = None,
) -> SemanticGraph:
    """Compose the complete semantic graph from shared runtime resources."""
    semantic = settings.semantic
    predictors = PredictorFactory(local_models, recorder)
    source_block_repository = SourceBlockRepository(database.session)
    source_entity_repository = SourceEntityRepository(database.session)
    source_event_repository = SourceEventRepository(database.session)
    source_predicate_repository = SourcePredicateRepository(database.session)
    source_procedure_repository = SourceProcedureRepository(database.session)
    source_statement_repository = SourceStatementRepository(database.session)
    source_triplet_repository = SourceTripletRepository(database.session)
    fact_extractor = FactExtractorModule(
        predictors.create(
            semantic.fact_extraction,
            FactExtractionSignature,
        )
    )
    triplet_decomposer = TripletDecomposerModule(
        predictors.create(
            semantic.triplet_decomposition,
            TripletDecompositionSignature,
        )
    )
    source_entity_hub_module = SourceEntityHubModule(
        predictors.create(
            semantic.source_entity_hubs.inference,
            SourceEntityHubSignature,
        )
    )
    source_entity_hub_judge = SourceEntityHubJudgeModule(
        predictors.create(
            semantic.source_entity_hubs.judge,
            SourceEntityHubJudgeSignature,
        )
    )
    source_event_hub_module = SourceEventHubModule(
        predictors.create(
            semantic.source_event_hubs.inference,
            SourceEventHubSignature,
        )
    )
    source_event_hub_judge = SourceEventHubJudgeModule(
        predictors.create(
            semantic.source_event_hubs.judge,
            SourceEventHubJudgeSignature,
        )
    )
    source_predicate_hub_module = SourcePredicateHubModule(
        predictors.create(
            semantic.source_predicate_hubs.inference,
            SourcePredicateHubSignature,
        )
    )
    source_predicate_hub_judge = SourcePredicateHubJudgeModule(
        predictors.create(
            semantic.source_predicate_hubs.judge,
            SourcePredicateHubJudgeSignature,
        )
    )
    source_statement_hub_module = SourceStatementHubModule(
        predictors.create(
            semantic.source_statement_hubs.inference,
            SourceStatementHubSignature,
        )
    )
    source_statement_hub_judge = SourceStatementHubJudgeModule(
        predictors.create(
            semantic.source_statement_hubs.judge,
            SourceStatementHubJudgeSignature,
        )
    )
    source_procedure_hub_module = SourceProcedureHubModule(
        predictors.create(
            semantic.source_procedure_hubs.inference,
            SourceProcedureHubSignature,
        )
    )
    source_procedure_hub_judge = SourceProcedureHubJudgeModule(
        predictors.create(
            semantic.source_procedure_hubs.judge,
            SourceProcedureHubJudgeSignature,
        )
    )
    source_triplet_hub_module = SourceTripletHubModule(
        predictors.create(
            semantic.source_triplet_hubs.inference,
            SourceTripletHubSignature,
        )
    )

    return SemanticGraph(
        triplet_source_load=TripletSourceLoadNode(source_block_repository),
        fact_extraction=FactExtractionNode(
            fact_extractor,
            semantic.context_window,
        ),
        triplet_decomposition=TripletDecompositionNode(triplet_decomposer),
        triplet_persistence=TripletPersistenceNode(source_triplet_repository),
        source_entity_description_load=SourceEntityDescriptionLoadNode(
            source_block_repository,
            source_entity_repository,
            semantic.source_entity_description.context_window,
        ),
        source_entity_description=SourceEntityDescriptionNode(
            SourceEntityDescriptionModule(
                predictors.create(
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
        ),
        source_event_description=SourceEventDescriptionNode(
            SourceEventDescriptionModule(
                predictors.create(
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
        ),
        source_predicate_description=SourcePredicateDescriptionNode(
            SourcePredicateDescriptionModule(
                predictors.create(
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
        ),
        source_statement_description=SourceStatementDescriptionNode(
            SourceStatementDescriptionModule(
                predictors.create(
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
        ),
        source_procedure_description=SourceProcedureDescriptionNode(
            SourceProcedureDescriptionModule(
                predictors.create(
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
        ),
        source_event_hub=SourceEventHubNode(
            source_event_repository,
            source_event_hub_module,
            source_event_hub_judge,
            local_models,
            local_models,
            semantic.source_event_hubs,
        ),
        source_predicate_hub=SourcePredicateHubNode(
            source_predicate_repository,
            source_predicate_hub_module,
            source_predicate_hub_judge,
            local_models,
            local_models,
            semantic.source_predicate_hubs,
        ),
        source_triplet_hub=SourceTripletHubNode(
            source_triplet_repository,
            source_triplet_hub_module,
            local_models,
        ),
        source_statement_hub=SourceStatementHubNode(
            source_statement_repository,
            source_statement_hub_module,
            source_statement_hub_judge,
            local_models,
            local_models,
            semantic.source_statement_hubs,
        ),
        source_procedure_hub=SourceProcedureHubNode(
            source_procedure_repository,
            source_procedure_hub_module,
            source_procedure_hub_judge,
            local_models,
            local_models,
            semantic.source_procedure_hubs,
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
