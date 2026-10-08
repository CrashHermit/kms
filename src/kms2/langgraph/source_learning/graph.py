"""Complete two-pass source-learning LangGraph assembly."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.atomic import (
    add_source_atomic_flashcard_phase,
)
from kms2.langgraph.source_learning.coherent import (
    add_source_coherent_flashcard_phase,
)
from kms2.langgraph.source_learning.state import SourceLearningState
from kms2.node.source_learning.atomic import SourceAtomicFlashcardNode
from kms2.node.source_learning.atomic_persistence import (
    SourceAtomicFlashcardPersistenceNode,
)
from kms2.node.source_learning.coherent import SourceCoherentFlashcardNode
from kms2.node.source_learning.coherent_persistence import (
    SourceCoherentFlashcardPersistenceNode,
)


class SourceLearningCleanupNode:
    """Clear generated source-learning cards before regeneration."""

    def __init__(self, repository: SourceLearningRepository) -> None:
        self._repository = repository

    async def run(self, state: SourceLearningState) -> dict[str, object]:
        """Delete only generated artifacts for the requested source."""
        await self._repository.clear_source_learning(state.source_uuid)
        return {}


class SourceLearningGraph:
    """Compile generation phases with separately injected persistence nodes.

    Processing prepares card occurrences; persistence nodes own card writes
    and persisted counts. Atomic cards are written before coherent preparation
    so coherent persistence can link the existing parent identities.
    """

    def __init__(
        self,
        repository: SourceLearningRepository,
        atomic: SourceAtomicFlashcardNode,
        atomic_persistence: SourceAtomicFlashcardPersistenceNode,
        coherent: SourceCoherentFlashcardNode,
        coherent_persistence: SourceCoherentFlashcardPersistenceNode,
    ) -> None:
        self._cleanup = SourceLearningCleanupNode(repository)
        self._atomic = atomic
        self._atomic_persistence = atomic_persistence
        self._coherent = coherent
        self._coherent_persistence = coherent_persistence
        self.graph = StateGraph(SourceLearningState)

    def build_graph(self) -> CompiledStateGraph:
        """Compile cleanup, atomic generation, and coherent generation."""
        self.graph.add_node('source_learning_cleanup', self._cleanup.run)
        add_source_atomic_flashcard_phase(
            self.graph, self._atomic, self._atomic_persistence
        )
        add_source_coherent_flashcard_phase(
            self.graph, self._coherent, self._coherent_persistence
        )
        self.graph.add_edge(START, 'source_learning_cleanup')
        self.graph.add_edge(
            'source_learning_cleanup',
            'source_atomic_flashcard_load',
        )
        self.graph.add_edge(
            'source_atomic_flashcard_persistence',
            'source_coherent_flashcard_prepare',
        )
        self.graph.add_edge(
            'source_coherent_flashcard_persistence',
            END,
        )
        return self.graph.compile()
