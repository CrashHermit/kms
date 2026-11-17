"""Compose the identity auditor with inference and embedding resources."""

from kms.config.settings import Settings
from kms.core.embedding import Embedder
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.knowledge.identity.audit import IdentityAuditor
from kms.knowledge.identity.inference import IdentityModule, IdentitySignature
from kms.knowledge.storage import KnowledgeRepository


def build_identity_auditor(
    settings: Settings,
    repository: KnowledgeRepository,
    embedder: Embedder,
    embedding_model: str,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> IdentityAuditor:
    """Build an auditor with matching selection and native prompt budgets."""
    stage = settings.knowledge_identity
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=IdentitySignature,
    )
    predictor = predictors.create(
        IdentityModule,
        stage.inference,
        IdentitySignature,
        budget=budget,
    )
    return IdentityAuditor(
        repository,
        embedder,
        embedding_model,
        IdentityModule(predictor),
        budget,
        candidate_limit=stage.candidate_limit,
    )
