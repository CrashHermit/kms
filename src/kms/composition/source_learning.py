"""Dependency composition for the two-pass source-learning graph."""

from kms.composition.predictors import PredictorFactory, build_token_budget
from kms.composition.tokenization import LanguageModelTokenizers
from kms.config.settings import Settings
from kms.core.media_budget import ImagePayloadBudget
from kms.database.client import DatabaseClient
from kms.database.source.source_block_repository import SourceBlockRepository
from kms.database.source_learning.repository import SourceLearningRepository
from kms.langgraph.source_learning.graph import SourceLearningGraph
from kms.local_models.runtime import LocalModelRuntime
from kms.module.source_learning.atomic import (
    SourceAtomicFlashcardModule,
    SourceAtomicFlashcardSignature,
)
from kms.module.source_learning.coherent import (
    SourceCoherentFlashcardModule,
    SourceCoherentFlashcardSignature,
)
from kms.node.source_learning.atomic import SourceAtomicFlashcardNode
from kms.node.source_learning.atomic_persistence import (
    SourceAtomicFlashcardPersistenceNode,
)
from kms.node.source_learning.cleanup import SourceLearningCleanupNode
from kms.node.source_learning.coherent import SourceCoherentFlashcardNode
from kms.node.source_learning.coherent_persistence import (
    SourceCoherentFlashcardPersistenceNode,
)
from kms.train.recorder import Recorder


def build_source_learning_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    tokenizers: LanguageModelTokenizers,
    recorder: Recorder | None = None,
) -> SourceLearningGraph:
    """Compose predictors, budgets, processing nodes, and persistence nodes."""
    learning = settings.source_learning
    predictors = PredictorFactory(settings, local_models, recorder)
    repository = SourceLearningRepository(database.session)
    source_block_repository = SourceBlockRepository(database.session)
    atomic_module = SourceAtomicFlashcardModule(
        predictors.create(
            SourceAtomicFlashcardModule,
            learning.atomic_flashcards,
            SourceAtomicFlashcardSignature,
        )
    )
    coherent_module = SourceCoherentFlashcardModule(
        predictors.create(
            SourceCoherentFlashcardModule,
            learning.coherent_flashcards,
            SourceCoherentFlashcardSignature,
        )
    )
    image_budget = ImagePayloadBudget(**learning.image_payload.model_dump())
    atomic_budget = build_token_budget(
        settings,
        tokenizers,
        learning.atomic_flashcards,
        input_token_budget=learning.input_token_budget,
        safety_margin_tokens=learning.safety_margin_tokens,
    )
    coherent_budget = build_token_budget(
        settings,
        tokenizers,
        learning.coherent_flashcards,
        input_token_budget=learning.input_token_budget,
        safety_margin_tokens=learning.safety_margin_tokens,
    )
    return SourceLearningGraph(
        SourceLearningCleanupNode(repository),
        SourceAtomicFlashcardNode(
            repository,
            atomic_module,
            learning.context_window,
            source_block_repository,
            budget=atomic_budget,
            image_budget=image_budget,
        ),
        SourceAtomicFlashcardPersistenceNode(repository),
        SourceCoherentFlashcardNode(
            coherent_module,
            budget=coherent_budget,
            image_budget=image_budget,
        ),
        SourceCoherentFlashcardPersistenceNode(repository),
    )
