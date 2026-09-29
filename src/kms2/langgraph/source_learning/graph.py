"""Complete typed source-learning LangGraph assembly."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.entity import (
    add_source_entity_learning_phase,
)
from kms2.langgraph.source_learning.event import add_source_event_learning_phase
from kms2.langgraph.source_learning.predicate import (
    add_source_predicate_learning_phase,
)
from kms2.langgraph.source_learning.state import SourceLearningState
from kms2.langgraph.source_learning.triplet import (
    add_source_triplet_learning_phase,
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


class SourceLearningCleanupNode:
    """Clear generated source-learning records before regeneration."""

    def __init__(self, repository: SourceLearningRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceLearningState) -> dict:
        """Delete only generated artifacts for the requested source."""
        await self._repository.clear_source_learning(state.source_uuid)
        return {}


class SourceLearningGraph:
    """Compile entity, event, predicate, and triplet learning phases."""

    def __init__(
        self,
        repository: SourceLearningRepository,
        entity_learning_fact: SourceEntityLearningFactNode,
        entity_flashcard: SourceEntityFlashcardNode,
        event_learning_fact: SourceEventLearningFactNode,
        event_flashcard: SourceEventFlashcardNode,
        predicate_learning_fact: SourcePredicateLearningFactNode,
        predicate_flashcard: SourcePredicateFlashcardNode,
        triplet_learning_fact: SourceTripletLearningFactNode,
        triplet_flashcard: SourceTripletFlashcardNode,
    ) -> None:
        self._cleanup = SourceLearningCleanupNode(repository)
        self._entity_learning_fact = entity_learning_fact
        self._entity_flashcard = entity_flashcard
        self._event_learning_fact = event_learning_fact
        self._event_flashcard = event_flashcard
        self._predicate_learning_fact = predicate_learning_fact
        self._predicate_flashcard = predicate_flashcard
        self._triplet_learning_fact = triplet_learning_fact
        self._triplet_flashcard = triplet_flashcard
        self.graph = StateGraph(SourceLearningState)

    def build_graph(self) -> CompiledStateGraph:
        """Compile the explicit source-learning phase sequence."""
        self.graph.add_node('source_learning_cleanup', self._cleanup.run)
        add_source_entity_learning_phase(
            self.graph, self._entity_learning_fact, self._entity_flashcard
        )
        add_source_event_learning_phase(
            self.graph, self._event_learning_fact, self._event_flashcard
        )
        add_source_predicate_learning_phase(
            self.graph, self._predicate_learning_fact, self._predicate_flashcard
        )
        add_source_triplet_learning_phase(
            self.graph, self._triplet_learning_fact, self._triplet_flashcard
        )
        self.graph.add_edge(START, 'source_learning_cleanup')
        self.graph.add_edge(
            'source_learning_cleanup', 'source_entity_learning_fact_load'
        )
        self.graph.add_edge(
            'source_entity_flashcard_persistence',
            'source_event_learning_fact_load',
        )
        self.graph.add_edge(
            'source_event_flashcard_persistence',
            'source_predicate_learning_fact_load',
        )
        self.graph.add_edge(
            'source_predicate_flashcard_persistence',
            'source_triplet_learning_fact_load',
        )
        self.graph.add_edge('source_triplet_flashcard_persistence', END)
        return self.graph.compile()
