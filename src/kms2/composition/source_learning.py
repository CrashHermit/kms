"""Dependency composition for the two-pass source-learning graph."""

from kms2.composition.predictors import PredictorFactory, build_token_budget
from kms2.config.settings import Settings
from kms2.database.client import DatabaseClient
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.inference.token_counting import InferenceTokenizers
from kms2.langgraph.source_learning.graph import SourceLearningGraph
from kms2.local_models.runtime import LocalModelRuntime
from kms2.module.source_learning.atomic import (
    SourceAtomicFlashcardModule,
    SourceAtomicFlashcardSignature,
)
from kms2.module.source_learning.coherent import (
    SourceCoherentFlashcardModule,
    SourceCoherentFlashcardSignature,
)
from kms2.node.source_learning.atomic import SourceAtomicFlashcardNode
from kms2.node.source_learning.atomic_persistence import (
    SourceAtomicFlashcardPersistenceNode,
)
from kms2.node.source_learning.coherent import SourceCoherentFlashcardNode
from kms2.node.source_learning.coherent_persistence import (
    SourceCoherentFlashcardPersistenceNode,
)
from kms2.train.recorder import Recorder


def build_source_learning_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    tokenizers: InferenceTokenizers,
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
        repository,
        SourceAtomicFlashcardNode(
            repository,
            atomic_module,
            learning.context_window,
            source_block_repository,
            budget=atomic_budget,
        ),
        SourceAtomicFlashcardPersistenceNode(repository),
        SourceCoherentFlashcardNode(
            coherent_module,
            budget=coherent_budget,
        ),
        SourceCoherentFlashcardPersistenceNode(repository),
    )
