"""Composition for grounded knowledge admission and identity preparation."""

from kms2.composition.predictors import PredictorFactory, build_token_budget
from kms2.config.settings import Settings
from kms2.core.embedding import EmbeddingClient
from kms2.core.reranking import RerankerClient
from kms2.database.client import DatabaseClient
from kms2.database.knowledge.repository import KnowledgeRepository
from kms2.inference.token_counting import InferenceTokenizers
from kms2.langgraph.knowledge.graph import KnowledgeAdmissionGraph
from kms2.module.knowledge.equivalence import (
    PropositionEquivalenceModule,
    PropositionEquivalenceSignature,
)
from kms2.module.knowledge.examples.equivalence import (
    proposition_equivalence_examples,
)
from kms2.module.knowledge.examples.extraction import (
    knowledge_extraction_examples,
)
from kms2.module.knowledge.extraction import (
    KnowledgeExtractorModule,
    KnowledgeExtractorSignature,
)
from kms2.node.knowledge.extraction import KnowledgeExtractionNode
from kms2.node.knowledge.identity import KnowledgeIdentityNode
from kms2.service.knowledge import KnowledgeService


def build_knowledge_admission_graph(
    settings: Settings,
    database: DatabaseClient,
    *,
    predictors: PredictorFactory,
    embedding_client: EmbeddingClient,
    reranker_client: RerankerClient,
    tokenizers: InferenceTokenizers,
) -> KnowledgeAdmissionGraph:
    """Compose inference and pure preparation over shared runtime resources."""
    knowledge_settings = settings.knowledge
    repository = KnowledgeRepository(database.session)
    knowledge = KnowledgeService(repository)

    extraction_settings = knowledge_settings.extraction
    extraction_module = KnowledgeExtractorModule(
        predictors.create(
            KnowledgeExtractorModule,
            extraction_settings.inference,
            KnowledgeExtractorSignature,
            examples=knowledge_extraction_examples(),
        )
    )
    extraction_budget = build_token_budget(
        settings,
        tokenizers,
        extraction_settings.inference,
        input_token_budget=extraction_settings.input_token_budget,
        safety_margin_tokens=extraction_settings.safety_margin_tokens,
    )

    identity_settings = knowledge_settings.identity
    equivalence_module = PropositionEquivalenceModule(
        predictors.create(
            PropositionEquivalenceModule,
            identity_settings.inference,
            PropositionEquivalenceSignature,
            examples=proposition_equivalence_examples(),
        )
    )
    identity_budget = build_token_budget(
        settings,
        tokenizers,
        identity_settings.inference,
        input_token_budget=identity_settings.input_token_budget,
        safety_margin_tokens=identity_settings.safety_margin_tokens,
    )
    return KnowledgeAdmissionGraph(
        KnowledgeExtractionNode(extraction_module, extraction_budget),
        KnowledgeIdentityNode(
            knowledge,
            equivalence_module,
            embedding_client,
            reranker_client,
            identity_budget,
            candidate_limit=identity_settings.candidate_limit,
            result_limit=identity_settings.result_limit,
        ),
    )
