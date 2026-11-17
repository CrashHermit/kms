"""Compose bounded knowledge reasoning."""

from kms.config.settings import Settings
from kms.database.client import DatabaseClient
from kms.inference.embedding.resources import EmbeddingResources
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.knowledge.commit import KnowledgeCommitNode
from kms.knowledge.composition import build_committer
from kms.knowledge.reasoning.examples import reasoning_examples
from kms.knowledge.reasoning.graph import KnowledgeReasoningGraph
from kms.knowledge.reasoning.inference import (
    KnowledgeReasoningModule,
    KnowledgeReasoningSignature,
)
from kms.knowledge.reasoning.node import KnowledgeReasoningNode
from kms.knowledge.retrieval.composition import build_search
from kms.knowledge.retrieval.node import KnowledgeSearchNode
from kms.retrieval.selection.composition import build_search_selector


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> KnowledgeReasoningNode:
    examples = reasoning_examples()
    stage = settings.knowledge_reasoning
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=KnowledgeReasoningSignature,
        examples=examples,
    )
    return KnowledgeReasoningNode(
        KnowledgeReasoningModule(
            predictors.create(
                KnowledgeReasoningModule,
                stage.inference,
                KnowledgeReasoningSignature,
                examples=examples,
                budget=budget,
            )
        ),
        budget,
        candidate_limit=stage.candidate_limit,
        argument_limit=stage.argument_limit,
    )


def build_graph(
    settings: Settings,
    database: DatabaseClient,
    *,
    embeddings: EmbeddingResources,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> KnowledgeReasoningGraph:
    """Build the standalone workflow using invocation-owned resources."""

    selection = build_search_selector(
        settings, predictors=predictors, tokenizers=tokenizers
    )
    search = KnowledgeSearchNode(
        build_search(
            settings, database, embeddings=embeddings, selection=selection
        )
    )
    reasoning = build_node(
        settings, predictors=predictors, tokenizers=tokenizers
    )
    commit = KnowledgeCommitNode(
        build_committer(
            settings,
            database,
            embeddings=embeddings,
            predictors=predictors,
            tokenizers=tokenizers,
        )
    )
    return KnowledgeReasoningGraph(search, reasoning, commit)
