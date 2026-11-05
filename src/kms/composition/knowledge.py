"""Composition for grounded knowledge admission and identity preparation."""

from kms.composition.predictors import PredictorFactory, build_token_budget
from kms.composition.tokenization import LanguageModelTokenizers
from kms.config.settings import Settings
from kms.core.embedding import Embedder
from kms.core.reranking import Reranker
from kms.database.client import DatabaseClient
from kms.database.knowledge.repository import KnowledgeRepository
from kms.langgraph.knowledge.graph import KnowledgeAdmissionGraph
from kms.module.knowledge.equivalence import (
    PropositionEquivalenceModule,
    PropositionEquivalenceSignature,
)
from kms.module.knowledge.examples.equivalence import (
    proposition_equivalence_examples,
)
from kms.module.knowledge.examples.extraction import (
    knowledge_extraction_examples,
)
from kms.module.knowledge.extraction import (
    KnowledgeExtractorModule,
    KnowledgeExtractorSignature,
)
from kms.node.knowledge.extraction import KnowledgeExtractionNode
from kms.node.knowledge.identity import KnowledgeIdentityNode
from kms.service.knowledge import KnowledgeService


def build_knowledge_admission_graph(
    settings: Settings,
    database: DatabaseClient,
    *,
    predictors: PredictorFactory,
    embedder: Embedder,
    reranker: Reranker,
    tokenizers: LanguageModelTokenizers,
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
            embedder,
            reranker,
            identity_budget,
            candidate_limit=identity_settings.candidate_limit,
            result_limit=identity_settings.result_limit,
        ),
    )
