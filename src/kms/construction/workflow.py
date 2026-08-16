"""Build the document-construction workflow graph."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from kms.construction import (
    canonicalizer,
    corrector,
    extractor,
    formatter,
    instruction_finder,
    name_hubs,
    ocr,
    pedagogical_component_finder,
    procedure_creator,
    seam_merger,
    splitter,
    statement_procedure_builder,
    triplet_extractor,
    triplet_hubs,
)
from kms.core import llm, recording, serve, state
from kms.graph import persister

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class CanonicalizerNode:
    """Runs source-local semantic and lexical canonicalization stages."""

    def __init__(self, session_factory, language_model) -> None:
        self._session_factory = session_factory
        self._language_model = language_model

    async def run(self, current_state: state.State) -> dict:
        """Canonicalize the current source and rebuild its assertion hubs."""
        if not self._session_factory:
            return {}
        source = current_state.get('source')
        if not source:
            raise ValueError('canonicalization requires the current source')
        entity_result = await canonicalizer.assign_source(
            'entity',
            source,
            language_model=self._language_model,
            session_factory=self._session_factory,
        )
        predicate_result = await canonicalizer.assign_source(
            'predicate',
            source,
            language_model=self._language_model,
            session_factory=self._session_factory,
        )
        entity_name_result = await name_hubs.rebuild(
            'entity',
            source,
            language_model=self._language_model,
            session_factory=self._session_factory,
        )
        predicate_name_result = await name_hubs.rebuild(
            'predicate',
            source,
            language_model=self._language_model,
            session_factory=self._session_factory,
        )
        triplet_hub_result = await triplet_hubs.rebuild(
            language_model=self._language_model,
            session_factory=self._session_factory,
            source=source,
        )
        return {
            'entity_assigned': entity_result['assigned'],
            'predicate_assigned': predicate_result['assigned'],
            'entity_hubs_created': entity_result['new_hubs'],
            'predicate_hubs_created': predicate_result['new_hubs'],
            'entity_name_hubs_created': entity_name_result['name_hubs'],
            'predicate_name_hubs_created': predicate_name_result['name_hubs'],
            'triplet_hubs_created': triplet_hub_result['triplet_hubs'],
        }


class ProcedureCreatorNode:
    """Creates learnable procedures from the persisted graph."""

    def __init__(self, session_factory, language_model) -> None:
        self._session_factory = session_factory
        self._language_model = language_model

    async def run(self, current_state: state.State) -> dict:
        """Create procedures when the graph database is configured."""
        if not self._session_factory:
            return {}
        created = await procedure_creator.create_procedures(
            self._session_factory,
            language_model=self._language_model,
        )
        return {'procedures_created': created}


def _build_modules(
    recorder: recording.Recorder | None,
) -> dict[str, object]:
    return {
        'corrector': corrector.Corrector(
            language_model=llm.module_lm('corrector'), recorder=recorder
        ),
        'formatter': formatter.Formatter(
            language_model=llm.module_lm('formatter'), recorder=recorder
        ),
        'extractor': extractor.Extractor(
            language_model=llm.module_lm('extractor'), recorder=recorder
        ),
        'seam_merger': seam_merger.SeamMerger(
            language_model=llm.module_lm('seam_merger'), recorder=recorder
        ),
        'seam_rewriter': seam_merger.SeamRewriter(
            language_model=llm.module_lm('seam_rewriter'), recorder=recorder
        ),
        'splitter': splitter.Splitter(
            language_model=llm.module_lm('splitter'), recorder=recorder
        ),
        'instruction_finder': instruction_finder.InstructionFinder(
            language_model=llm.module_lm('instruction_finder'),
            recorder=recorder,
        ),
        'component_finder': (
            pedagogical_component_finder.PedagogicalComponentFinder(
                language_model=llm.module_lm('pedagogical_component_finder'),
                recorder=recorder,
            )
        ),
        'role_typer': statement_procedure_builder.RoleTyper(
            language_model=llm.module_lm('role_typer'), recorder=recorder
        ),
        'statement_partitioner': (
            statement_procedure_builder.StatementPartitioner(
                language_model=llm.module_lm('statement_partitioner'),
                recorder=recorder,
            )
        ),
        'procedure_partitioner': (
            statement_procedure_builder.ProcedurePartitioner(
                language_model=llm.module_lm('procedure_partitioner'),
                recorder=recorder,
            )
        ),
        'fact_extractor': triplet_extractor._FactExtractor(
            language_model=llm.module_lm('atomic_fact_extractor'),
            recorder=recorder,
        ),
        'triplet_extractor': triplet_extractor._TripletDecomposer(
            language_model=llm.module_lm('triplet_extractor'),
            recorder=recorder,
        ),
        'component_enrichment': canonicalizer.ComponentEnricher(
            language_model=llm.module_lm('component_enrichment'),
            recorder=recorder,
        ),
    }


def build_workflow(
    *,
    recorder: recording.Recorder | None = None,
    neo4j_session_factory: Callable | None = None,
    neo4j_configured: bool = False,
    model_manager: serve.RouterManager | None = None,
) -> 'CompiledStateGraph':
    """Build and compile the document-construction workflow.

    Args:
        recorder: Optional recorder for stage-level LLM examples.
        neo4j_session_factory: Factory for graph database sessions.
        neo4j_configured: Whether graph persistence is enabled.
        model_manager: Optional manager for stage-specific model switching.

    Returns:
        The compiled LangGraph workflow.
    """
    modules = _build_modules(recorder)
    corrector_module = modules['corrector']
    formatter_module = modules['formatter']
    extractor_module = modules['extractor']
    seam_module = modules['seam_merger']
    seam_rewriter_module = modules['seam_rewriter']
    splitter_module = modules['splitter']
    instruction_finder_module = modules['instruction_finder']
    component_finder_module = modules['component_finder']
    role_typer_module = modules['role_typer']
    statement_partitioner_module = modules['statement_partitioner']
    procedure_partitioner_module = modules['procedure_partitioner']
    fact_module = modules['fact_extractor']
    triplet_module = modules['triplet_extractor']
    component_enrichment_module = modules['component_enrichment']

    corrector_node = corrector.CorrectorNode(module=corrector_module)
    formatter_node = formatter.FormatterNode(module=formatter_module)
    extractor_node = extractor.ExtractorNode(module=extractor_module)
    seam_node = seam_merger.SeamMergerNode(
        module=seam_module, rewriter=seam_rewriter_module
    )
    splitter_node = splitter.SplitterNode(module=splitter_module)
    instruction_finder_node = instruction_finder.InstructionFinderNode(
        module=instruction_finder_module
    )
    component_finder_node = (
        pedagogical_component_finder.PedagogicalComponentFinderNode(
            module=component_finder_module
        )
    )
    statement_procedure_builder_node = (
        statement_procedure_builder.StatementProcedureBuilderNode(
            role_module=role_typer_module,
            statement_partitioner=statement_partitioner_module,
            procedure_partitioner=procedure_partitioner_module,
        )
    )
    triplet_extractor_node = triplet_extractor.TripletNode(
        fact_module=fact_module,
        triplet_module=triplet_module,
    )
    component_enrichment_node = canonicalizer.ComponentEnrichmentNode(
        module=component_enrichment_module
    )
    node_persister_node = persister.IngestionPersisterNode(
        session_factory=neo4j_session_factory,
        neo4j_configured=neo4j_configured,
    )
    canonicalizer_node = CanonicalizerNode(
        session_factory=neo4j_session_factory,
        language_model=llm.module_lm('canonicalizer'),
    )
    procedure_creator_node = ProcedureCreatorNode(
        session_factory=neo4j_session_factory,
        language_model=llm.module_lm('procedure_creator'),
    )

    graph = StateGraph(state.State)
    graph.add_node('ocr', ocr.OCRNode().run)
    graph.add_node('corrector_worker', corrector_node.worker)
    graph.add_node('corrector_collect', corrector_node.collect)
    graph.add_node('formatter_worker', formatter_node.worker)
    graph.add_node('formatter_collect', formatter_node.collect)
    graph.add_node('extractor_worker', extractor_node.worker)
    graph.add_node('extractor_collect', extractor_node.collect)
    graph.add_node('seam_even_worker', seam_node.even_worker)
    graph.add_node('seam_even_collect', seam_node.even_collect)
    graph.add_node('seam_odd_worker', seam_node.odd_worker)
    graph.add_node('seam_odd_collect', seam_node.odd_collect)
    graph.add_node('splitter', splitter_node.run)
    graph.add_node('instruction_finder', instruction_finder_node.run)
    graph.add_node('pedagogical_component_finder', component_finder_node.run)
    graph.add_node(
        'statement_procedure_builder', statement_procedure_builder_node.run
    )
    graph.add_node('triplet_extraction', triplet_extractor_node.run)
    graph.add_node('component_enrichment', component_enrichment_node.run)
    graph.add_node('ingestion_persister', node_persister_node.run)
    graph.add_node('canonicalizer', canonicalizer_node.run)
    graph.add_node('procedure_creator', procedure_creator_node.run)

    formatter_entry = 'corrector_collect'
    procedure_creator_entry = 'canonicalizer'
    if model_manager:
        graph.add_node(
            'switch_to_corrector',
            serve.SwitchNode(model_manager, 'corrector').run,
        )
        graph.add_node(
            'switch_to_modules',
            serve.SwitchNode(model_manager, 'formatter').run,
        )
        graph.add_node(
            'switch_to_procedure_creator',
            serve.SwitchNode(model_manager, 'procedure_creator').run,
        )
        graph.add_edge(START, 'switch_to_corrector')
        graph.add_edge('switch_to_corrector', 'ocr')
        graph.add_edge('corrector_collect', 'switch_to_modules')
        graph.add_edge('canonicalizer', 'switch_to_procedure_creator')
        formatter_entry = 'switch_to_modules'
        procedure_creator_entry = 'switch_to_procedure_creator'

    if not model_manager:
        graph.add_edge(START, 'ocr')
    graph.add_conditional_edges(
        'ocr',
        corrector_node.dispatch,
        ['corrector_worker', 'corrector_collect'],
    )
    graph.add_edge('corrector_worker', 'corrector_collect')
    graph.add_conditional_edges(
        formatter_entry,
        formatter_node.dispatch,
        ['formatter_worker', 'formatter_collect'],
    )
    graph.add_edge('formatter_worker', 'formatter_collect')
    graph.add_conditional_edges(
        'formatter_collect',
        extractor_node.dispatch,
        ['extractor_worker', 'extractor_collect'],
    )
    graph.add_edge('extractor_worker', 'extractor_collect')
    graph.add_conditional_edges(
        'extractor_collect',
        seam_node.dispatch_even,
        ['seam_even_worker', 'seam_even_collect'],
    )
    graph.add_edge('seam_even_worker', 'seam_even_collect')
    graph.add_conditional_edges(
        'seam_even_collect',
        seam_node.dispatch_odd,
        ['seam_odd_worker', 'seam_odd_collect'],
    )
    graph.add_edge('seam_odd_worker', 'seam_odd_collect')
    graph.add_edge('seam_odd_collect', 'splitter')
    graph.add_edge('splitter', 'instruction_finder')
    graph.add_edge('instruction_finder', 'pedagogical_component_finder')
    graph.add_edge(
        'pedagogical_component_finder', 'statement_procedure_builder'
    )
    graph.add_edge('statement_procedure_builder', 'triplet_extraction')
    graph.add_edge('triplet_extraction', 'component_enrichment')
    graph.add_edge('component_enrichment', 'ingestion_persister')
    graph.add_edge('ingestion_persister', 'canonicalizer')
    graph.add_edge(procedure_creator_entry, 'procedure_creator')
    graph.add_edge('procedure_creator', END)
    return graph.compile()
