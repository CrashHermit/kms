"""Dependency composition for the typed source-learning graph."""

from kms2.composition.predictors import PredictorFactory
from kms2.config.settings import Settings
from kms2.database.client import DatabaseClient
from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.graph import SourceLearningGraph
from kms2.local_models.runtime import LocalModelRuntime
from kms2.module.source_learning.entity import (
    SourceEntityFlashcardModule,
    SourceEntityFlashcardSignature,
    SourceEntityLearningFactModule,
    SourceEntityLearningFactSignature,
)
from kms2.module.source_learning.event import (
    SourceEventFlashcardModule,
    SourceEventFlashcardSignature,
    SourceEventLearningFactModule,
    SourceEventLearningFactSignature,
)
from kms2.module.source_learning.predicate import (
    SourcePredicateFlashcardModule,
    SourcePredicateFlashcardSignature,
    SourcePredicateLearningFactModule,
    SourcePredicateLearningFactSignature,
)
from kms2.module.source_learning.triplet import (
    SourceTripletFlashcardModule,
    SourceTripletFlashcardSignature,
    SourceTripletLearningFactModule,
    SourceTripletLearningFactSignature,
)
from kms2.node.source_learning.entity import (
    SourceEntityFlashcardNode,
    SourceEntityLearningFactNode,
)
from kms2.node.source_learning.event import (
    SourceEventFlashcardNode,
    SourceEventLearningFactNode,
)
from kms2.node.source_learning.predicate import (
    SourcePredicateFlashcardNode,
    SourcePredicateLearningFactNode,
)
from kms2.node.source_learning.triplet import (
    SourceTripletFlashcardNode,
    SourceTripletLearningFactNode,
)
from kms2.train.recorder import Recorder


def build_source_learning_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    recorder: Recorder | None = None,
) -> SourceLearningGraph:
    """Compose all eight typed source-learning modules and nodes."""
    learning = settings.source_learning
    predictors = PredictorFactory(local_models, recorder)
    repository = SourceLearningRepository(database.session)

    entity_learning_fact = SourceEntityLearningFactModule(
        predictors.create(
            SourceEntityLearningFactModule,
            learning.source_entity_learning_fact,
            SourceEntityLearningFactSignature,
        )
    )
    entity_flashcard = SourceEntityFlashcardModule(
        predictors.create(
            SourceEntityFlashcardModule,
            learning.source_entity_flashcard,
            SourceEntityFlashcardSignature,
        )
    )
    event_learning_fact = SourceEventLearningFactModule(
        predictors.create(
            SourceEventLearningFactModule,
            learning.source_event_learning_fact,
            SourceEventLearningFactSignature,
        )
    )
    event_flashcard = SourceEventFlashcardModule(
        predictors.create(
            SourceEventFlashcardModule,
            learning.source_event_flashcard,
            SourceEventFlashcardSignature,
        )
    )
    predicate_learning_fact = SourcePredicateLearningFactModule(
        predictors.create(
            SourcePredicateLearningFactModule,
            learning.source_predicate_learning_fact,
            SourcePredicateLearningFactSignature,
        )
    )
    predicate_flashcard = SourcePredicateFlashcardModule(
        predictors.create(
            SourcePredicateFlashcardModule,
            learning.source_predicate_flashcard,
            SourcePredicateFlashcardSignature,
        )
    )
    triplet_learning_fact = SourceTripletLearningFactModule(
        predictors.create(
            SourceTripletLearningFactModule,
            learning.source_triplet_learning_fact,
            SourceTripletLearningFactSignature,
        )
    )
    triplet_flashcard = SourceTripletFlashcardModule(
        predictors.create(
            SourceTripletFlashcardModule,
            learning.source_triplet_flashcard,
            SourceTripletFlashcardSignature,
        )
    )

    return SourceLearningGraph(
        repository,
        SourceEntityLearningFactNode(repository, entity_learning_fact),
        SourceEntityFlashcardNode(repository, entity_flashcard),
        SourceEventLearningFactNode(repository, event_learning_fact),
        SourceEventFlashcardNode(repository, event_flashcard),
        SourcePredicateLearningFactNode(repository, predicate_learning_fact),
        SourcePredicateFlashcardNode(repository, predicate_flashcard),
        SourceTripletLearningFactNode(repository, triplet_learning_fact),
        SourceTripletFlashcardNode(repository, triplet_flashcard),
    )
