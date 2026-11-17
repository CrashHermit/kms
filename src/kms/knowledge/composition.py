"""Compose transactional knowledge persistence and identity auditing."""

from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.inference.embedding.resources import EmbeddingResources
from kms.inference.predictors import PredictorFactory
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.knowledge.commit import KnowledgeCommitter
from kms.knowledge.identity.composition import build_identity_auditor
from kms.knowledge.storage import KnowledgeRepository


def build_committer(
    settings: Settings,
    database: DatabaseClient,
    *,
    embeddings: EmbeddingResources,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> KnowledgeCommitter:
    repository = KnowledgeRepository(database.session)
    auditor = build_identity_auditor(
        settings,
        repository,
        embeddings.embedder,
        embeddings.model.model,
        predictors=predictors,
        tokenizers=tokenizers,
    )
    return KnowledgeCommitter(
        repository,
        embeddings.embedder,
        embeddings.model.model,
        auditor=auditor,
    )
