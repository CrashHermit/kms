"""Composition for researched source snapshot admission."""

from kms2.composition.predictors import PredictorFactory
from kms2.composition.source_processing import build_source_processing_graph
from kms2.config.settings import Settings
from kms2.core.embedding import EmbeddingClient
from kms2.database.client import DatabaseClient
from kms2.database.source.source_snapshot_repository import (
    SourceSnapshotRepository,
)
from kms2.inference.token_counting import InferenceTokenizers
from kms2.langgraph.source_admission.graph import SourceAdmissionGraph
from kms2.local_models.runtime import LocalModelRuntime
from kms2.node.source_admission.finalization import (
    SourceAdmissionFinalizationNode,
)
from kms2.node.source_admission.preparation import (
    SourceAdmissionPreparationNode,
)
from kms2.node.source_admission.processing import SourceAdmissionProcessingNode
from kms2.service.source_admission import SourceAdmissionService
from kms2.train.recorder import Recorder


def build_source_admission_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    embedding_client: EmbeddingClient,
    tokenizers: InferenceTokenizers,
    predictors: PredictorFactory,
    recorder: Recorder | None = None,
) -> SourceAdmissionGraph:
    """Compose capture, shared processing, and staged snapshot finalization."""
    research = settings.research
    service = SourceAdmissionService(
        SourceSnapshotRepository(database.session),
        tokenizers.for_inference(research.inference),
        research.source_directory,
        research.passage_token_budget,
    )
    processing_graph = build_source_processing_graph(
        settings,
        local_models,
        database,
        embedding_client=embedding_client,
        tokenizers=tokenizers,
        recorder=recorder,
        predictors=predictors,
    ).build_graph(persist=False)
    return SourceAdmissionGraph(
        SourceAdmissionPreparationNode(service),
        SourceAdmissionProcessingNode(processing_graph),
        SourceAdmissionFinalizationNode(service),
    )
