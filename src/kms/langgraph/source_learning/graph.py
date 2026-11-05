"""Complete two-pass source-learning LangGraph assembly."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms.langgraph.source_learning.atomic import (
    add_source_atomic_flashcard_phase,
)
from kms.langgraph.source_learning.coherent import (
    add_source_coherent_flashcard_phase,
)
from kms.langgraph.source_learning.state import SourceLearningState
from kms.node.source_learning.atomic import SourceAtomicFlashcardNode
from kms.node.source_learning.atomic_persistence import (
    SourceAtomicFlashcardPersistenceNode,
)
from kms.node.source_learning.cleanup import SourceLearningCleanupNode
from kms.node.source_learning.coherent import SourceCoherentFlashcardNode
from kms.node.source_learning.coherent_persistence import (
    SourceCoherentFlashcardPersistenceNode,
)


class SourceLearningGraph:
    """Compile and execute cleanup and the two flashcard generation passes."""

    def __init__(
        self,
        cleanup: SourceLearningCleanupNode,
        atomic: SourceAtomicFlashcardNode,
        atomic_persistence: SourceAtomicFlashcardPersistenceNode,
        coherent: SourceCoherentFlashcardNode,
        coherent_persistence: SourceCoherentFlashcardPersistenceNode,
    ) -> None:
        graph = StateGraph(SourceLearningState)
        graph.add_node('source_learning_cleanup', cleanup.run)
        add_source_atomic_flashcard_phase(graph, atomic, atomic_persistence)
        add_source_coherent_flashcard_phase(
            graph, coherent, coherent_persistence
        )
        graph.add_edge(START, 'source_learning_cleanup')
        graph.add_edge(
            'source_learning_cleanup',
            'source_atomic_flashcard_load',
        )
        graph.add_edge(
            'source_atomic_flashcard_persistence',
            'source_coherent_flashcard_prepare',
        )
        graph.add_edge(
            'source_coherent_flashcard_persistence',
            END,
        )
        self._graph: CompiledStateGraph = graph.compile()

    async def run(self, state: SourceLearningState) -> SourceLearningState:
        result = await self._graph.ainvoke(state)
        return SourceLearningState.model_validate(result)
