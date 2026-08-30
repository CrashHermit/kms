"""Build the document-construction workflow graph."""

import inspect
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from kms import config
from kms.construction import (
    block_corrector,
    entity_enrichment,
    formatter,
    governance_judge,
    governance_walker,
    image_enricher,
    image_seam_merger,
    instruction_finder,
    local_entity_hubs,
    local_predicate_hubs,
    local_procedure_hubs,
    local_statement_hubs,
    ocr,
    pedagogical_component_finder,
    pedagogical_window_readiness,
    predicate_enrichment,
    procedure_enrichment,
    splitter,
    statement_enrichment,
    statement_procedure_builder,
    text_seam_merger,
    triplet_extractor,
    triplet_hubs,
)
from kms.core import llm, recording, state
from kms.graph import projectors

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class TripletHubNode:
    def __init__(self, language_model) -> None:
        self._language_model = language_model

    async def run(self, current_state: state.State) -> dict:
        bundle = state.to_construction_bundle(current_state)
        source = bundle.source.key
        if not source:
            return {}
        result = await triplet_hubs.build_source(
            language_model=self._language_model,
            source=source,
            triplets=bundle.triplets,
            entity_assignments=bundle.entity_hub_assignments,
            predicate_assignments=bundle.predicate_hub_assignments,
            entity_hubs=bundle.entity_hub_records,
            predicate_hubs=bundle.predicate_hub_records,
        )
        bundle.triplet_hubs = result.get('hubs', [])
        return {
            'triplet_hubs': bundle.triplet_hubs,
            'construction_bundle': bundle,
        }


def _build_modules(
    recorder: recording.Recorder | None,
) -> dict[str, object]:
    return {
        'block_corrector': block_corrector.BlockCorrector(
            language_model=llm.module_lm('corrector'), recorder=recorder
        ),
        'formatter': formatter.Formatter(
            language_model=llm.module_lm('formatter'), recorder=recorder
        ),
        'text_seam_merger': text_seam_merger.TextSeamMerger(
            language_model=llm.module_lm('text_seam_merger'),
            recorder=recorder,
        ),
        'text_seam_rewriter': text_seam_merger.TextSeamRewriter(
            language_model=llm.module_lm('text_seam_rewriter'),
            recorder=recorder,
        ),
        'image_seam_merger': image_seam_merger.ImageSeamMerger(
            language_model=llm.module_lm('image_seam_merger'),
            recorder=recorder,
        ),
        'image_enricher': image_enricher.ImageEnricher(
            language_model=llm.module_lm('image_enricher'),
            recorder=recorder,
        ),
        'splitter': splitter.Splitter(
            language_model=llm.module_lm('splitter'), recorder=recorder
        ),
        'exercise_strip_router': splitter.ExerciseStripRouter(
            language_model=llm.module_lm('exercise_strip_router'),
            recorder=recorder,
        ),
        'instruction_router': instruction_finder.InstructionRouter(
            language_model=llm.module_lm('instruction_router'),
            recorder=recorder,
        ),
        'instruction_grower': instruction_finder.InstructionGrower(
            language_model=llm.module_lm('instruction_grower'),
            recorder=recorder,
        ),
        'component_finder': (
            pedagogical_component_finder.PedagogicalComponentFinder(
                language_model=llm.module_lm('pedagogical_component_finder'),
                recorder=recorder,
            )
        ),
        'pedagogical_window_readiness_router': (
            pedagogical_window_readiness.PedagogicalWindowReadinessRouter(
                language_model=llm.module_lm('pedagogical_component_finder'),
                recorder=recorder,
            )
        ),
        'governance_judge': governance_judge.GovernanceJudge(
            language_model=llm.module_lm('governance_judge'),
            recorder=recorder,
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
        'entity_enrichment': entity_enrichment.EntityEnricher(
            language_model=llm.module_lm('entity_enrichment'),
            recorder=recorder,
        ),
        'predicate_enrichment': predicate_enrichment.PredicateEnricher(
            language_model=llm.module_lm('predicate_enrichment'),
            recorder=recorder,
        ),
        'statement_enrichment': statement_enrichment.StatementEnricher(
            language_model=llm.module_lm('statement_enrichment'),
            recorder=recorder,
        ),
        'procedure_enrichment': procedure_enrichment.ProcedureEnricher(
            language_model=llm.module_lm('procedure_enrichment'),
            recorder=recorder,
        ),
        'statement_hub_builder': local_statement_hubs.LocalStatementHubSynthesizer(
            language_model=llm.module_lm('statement_hub_builder'),
            recorder=recorder,
        ),
        'statement_hub_adjudicator': local_statement_hubs.LocalStatementHubAdjudicator(
            language_model=llm.module_lm('statement_hub_builder'),
            recorder=recorder,
        ),
        'procedure_hub_builder': local_procedure_hubs.LocalProcedureHubSynthesizer(
            language_model=llm.module_lm('procedure_hub_builder'),
            recorder=recorder,
        ),
        'procedure_hub_adjudicator': local_procedure_hubs.LocalProcedureHubAdjudicator(
            language_model=llm.module_lm('procedure_hub_builder'),
            recorder=recorder,
        ),
        'entity_hub_adjudicator': local_entity_hubs.EntityHubAdjudicator(
            language_model=llm.module_lm('entity_hub_builder'),
            recorder=recorder,
        ),
        'entity_hub_builder': local_entity_hubs.EntityHubSynthesizer(
            language_model=llm.module_lm('entity_hub_builder'),
            recorder=recorder,
        ),
        'predicate_hub_adjudicator': local_predicate_hubs.PredicateHubAdjudicator(
            language_model=llm.module_lm('predicate_hub_builder'),
        ),
        'predicate_hub_builder': local_predicate_hubs.PredicateHubSynthesizer(
            language_model=llm.module_lm('predicate_hub_builder'),
        ),
        'triplet_hub_builder': llm.module_lm('triplet_hub_builder'),
    }


async def _timed_stage(
    name: str,
    node,
    current_state: state.State,
    recorder: recording.Recorder | None,
) -> dict:
    """Runs one workflow stage with console and recording progress."""
    started = time.perf_counter()
    logger.info('pipeline stage started: %s', name)
    try:
        result = node(current_state)
        if inspect.isawaitable(result):
            result = await result
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            'pipeline stage failed: %s (%s ms)',
            name,
            duration_ms,
        )
        if recorder is not None:
            recorder.record_progress(
                name,
                status='failed',
                duration_ms=duration_ms,
            )
        raise
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info('pipeline stage completed: %s (%s ms)', name, duration_ms)
    if recorder is not None:
        recorder.record_progress(
            name,
            status='completed',
            duration_ms=duration_ms,
            output_keys=list(result),
        )
    return result


def _timed_node(
    name: str,
    node,
    recorder: recording.Recorder | None,
):
    """Returns a graph node wrapper with stage timing."""

    async def run(current_state: state.State) -> dict:
        return await _timed_stage(name, node, current_state, recorder)

    return run


def build_workflow(
    *,
    recorder: recording.Recorder | None = None,
    neo4j_session_factory: Callable | None = None,
    neo4j_configured: bool = False,
) -> 'CompiledStateGraph':
    """Build and compile the document-construction workflow.

    Args:
        recorder: Optional recorder for stage-level LLM examples.
        neo4j_session_factory: Factory for graph database sessions.
        neo4j_configured: Whether graph persistence is enabled.

    Returns:
        The compiled LangGraph workflow.
    """
    modules = _build_modules(recorder)
    block_corrector_module = modules['block_corrector']
    formatter_module = modules['formatter']
    text_seam_module = modules['text_seam_merger']
    text_seam_rewriter_module = modules['text_seam_rewriter']
    image_seam_module = modules['image_seam_merger']
    image_enricher_module = modules['image_enricher']
    splitter_module = modules['splitter']
    exercise_strip_router_module = modules['exercise_strip_router']
    instruction_router_module = modules['instruction_router']
    instruction_grower_module = modules['instruction_grower']
    component_finder_module = modules['component_finder']
    readiness_module = modules['pedagogical_window_readiness_router']
    role_typer_module = modules['role_typer']
    statement_partitioner_module = modules['statement_partitioner']
    procedure_partitioner_module = modules['procedure_partitioner']
    fact_module = modules['fact_extractor']
    triplet_module = modules['triplet_extractor']
    entity_enrichment_module = modules['entity_enrichment']
    predicate_enrichment_module = modules['predicate_enrichment']
    statement_enrichment_module = modules['statement_enrichment']
    procedure_enrichment_module = modules['procedure_enrichment']
    statement_hub_module = modules['statement_hub_builder']
    statement_hub_adjudicator = modules['statement_hub_adjudicator']
    procedure_hub_module = modules['procedure_hub_builder']
    procedure_hub_adjudicator = modules['procedure_hub_adjudicator']

    block_corrector_node = block_corrector.BlockCorrectorNode(
        corrector=block_corrector_module,
    )
    formatter_node = formatter.FormatterNode(module=formatter_module)
    text_seam_node = text_seam_merger.TextSeamMergerNode(
        module=text_seam_module,
        rewriter=text_seam_rewriter_module,
    )
    image_seam_node = image_seam_merger.ImageSeamMergerNode(
        merger=image_seam_module
    )
    image_enrichment_node = image_enricher.ImageEnrichmentNode(
        enricher=image_enricher_module,
    )
    splitter_node = splitter.SplitterNode(
        module=splitter_module,
        router=exercise_strip_router_module,
    )
    instruction_finder_node = instruction_finder.InstructionFinderNode(
        router=instruction_router_module,
        grower=instruction_grower_module,
    )
    component_finder_node = (
        pedagogical_component_finder.PedagogicalComponentFinderNode(
            module=component_finder_module,
            readiness_module=readiness_module,
        )
    )
    governance_config = config.get_settings().stages.governance
    governance_walker_node = governance_walker.GovernanceStatementWalkerNode(
        judge=modules['governance_judge'],
        backward_budget=governance_config.backward_context_budget,
        forward_budget=governance_config.forward_context_budget,
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
    entity_enrichment_node = entity_enrichment.EntityEnrichmentNode(
        enricher=entity_enrichment_module
    )
    predicate_enrichment_node = predicate_enrichment.PredicateEnrichmentNode(
        enricher=predicate_enrichment_module
    )
    final_projector_node = projectors.FinalProjectorNode(
        session_factory=neo4j_session_factory,
        neo4j_configured=neo4j_configured,
    )
    entity_hub_node = local_entity_hubs.EntityHubNode(
        modules['entity_hub_adjudicator'],
        modules['entity_hub_builder'],
    )
    predicate_hub_node = local_predicate_hubs.PredicateHubNode(
        modules['predicate_hub_adjudicator'],
        modules['predicate_hub_builder'],
    )
    triplet_hub_node = TripletHubNode(modules['triplet_hub_builder'])
    statement_enrichment_node = statement_enrichment.StatementEnrichmentNode(
        enricher=statement_enrichment_module,
    )
    procedure_enrichment_node = procedure_enrichment.ProcedureEnrichmentNode(
        enricher=procedure_enrichment_module,
    )

    statement_hub_node = local_statement_hubs.LocalStatementHubNode(
        adjudicator=statement_hub_adjudicator,
        synthesizer=statement_hub_module,
    )
    procedure_hub_node = local_procedure_hubs.LocalProcedureHubNode(
        adjudicator=procedure_hub_adjudicator,
        synthesizer=procedure_hub_module,
    )
    graph = StateGraph(state.State)
    graph.add_node('ocr', _timed_node('ocr', ocr.OCRNode().run, recorder))
    graph.add_node('block_corrector_worker', block_corrector_node.worker)
    graph.add_node('block_corrector_collect', block_corrector_node.collect)
    graph.add_node('formatter_worker', formatter_node.worker)
    graph.add_node('formatter_collect', formatter_node.collect)
    graph.add_node('text_seam_even_worker', text_seam_node.even_worker)
    graph.add_node('text_seam_even_collect', text_seam_node.even_collect)
    graph.add_node(
        'text_seam_odd_dispatch',
        lambda current_state: {},
    )
    graph.add_node('text_seam_odd_worker', text_seam_node.odd_worker)
    graph.add_node('text_seam_odd_collect', text_seam_node.odd_collect)
    graph.add_node(
        'image_seam_even_dispatch',
        lambda current_state: {},
    )
    graph.add_node('image_seam_even_worker', image_seam_node.even_worker)
    graph.add_node('image_seam_even_collect', image_seam_node.even_collect)
    graph.add_node(
        'image_seam_odd_dispatch',
        lambda current_state: {},
    )
    graph.add_node('image_seam_odd_worker', image_seam_node.odd_worker)
    graph.add_node('image_seam_odd_collect', image_seam_node.odd_collect)
    graph.add_node(
        'image_enrichment',
        _timed_node('image_enrichment', image_enrichment_node.run, recorder),
    )
    graph.add_node(
        'splitter',
        _timed_node('splitter', splitter_node.run, recorder),
    )
    graph.add_node(
        'instruction_finder',
        _timed_node(
            'instruction_finder', instruction_finder_node.run, recorder
        ),
    )
    graph.add_node(
        'governance_walker',
        _timed_node('governance_walker', governance_walker_node.run, recorder),
    )
    graph.add_node(
        'pedagogical_component_finder',
        _timed_node(
            'pedagogical_component_finder',
            component_finder_node.run,
            recorder,
        ),
    )
    graph.add_node(
        'statement_procedure_builder',
        _timed_node(
            'statement_procedure_builder',
            statement_procedure_builder_node.run,
            recorder,
        ),
    )
    graph.add_node(
        'triplet_extraction',
        _timed_node('triplet_extraction', triplet_extractor_node.run, recorder),
    )
    graph.add_node(
        'entity_enrichment',
        _timed_node('entity_enrichment', entity_enrichment_node.run, recorder),
    )
    graph.add_node(
        'predicate_enrichment',
        _timed_node(
            'predicate_enrichment', predicate_enrichment_node.run, recorder
        ),
    )
    graph.add_node(
        'entity_hub_builder',
        _timed_node('entity_hub_builder', entity_hub_node.run, recorder),
    )
    graph.add_node(
        'predicate_hub_builder',
        _timed_node('predicate_hub_builder', predicate_hub_node.run, recorder),
    )
    graph.add_node(
        'triplet_hub_builder',
        _timed_node('triplet_hub_builder', triplet_hub_node.run, recorder),
    )
    graph.add_node(
        'statement_enrichment',
        _timed_node(
            'statement_enrichment', statement_enrichment_node.run, recorder
        ),
    )
    graph.add_node(
        'procedure_enrichment',
        _timed_node(
            'procedure_enrichment', procedure_enrichment_node.run, recorder
        ),
    )
    graph.add_node(
        'statement_hub_builder',
        _timed_node('statement_hub_builder', statement_hub_node.run, recorder),
    )
    graph.add_node(
        'procedure_hub_builder',
        _timed_node('procedure_hub_builder', procedure_hub_node.run, recorder),
    )
    graph.add_node(
        'final_projector',
        _timed_node('final_projector', final_projector_node.run, recorder),
    )
    graph.add_edge(START, 'ocr')
    block_corrector_entry = 'ocr'
    formatter_entry = 'block_corrector_collect'
    text_seam_even_entry = 'formatter_collect'
    text_seam_odd_entry = 'text_seam_odd_dispatch'
    image_seam_even_entry = 'image_seam_even_dispatch'
    image_seam_odd_entry = 'image_seam_odd_dispatch'
    splitter_entry = 'splitter'
    instruction_entry = 'instruction_finder'
    component_entry = 'pedagogical_component_finder'
    statement_builder_entry = 'statement_procedure_builder'
    triplet_entry = 'triplet_extraction'
    entity_enrichment_entry = 'entity_enrichment'
    predicate_enrichment_entry = 'predicate_enrichment'
    entity_hub_entry = 'entity_hub_builder'
    predicate_hub_entry = 'predicate_hub_builder'
    triplet_hub_entry = 'triplet_hub_builder'
    statement_enrichment_entry = 'statement_enrichment'
    procedure_enrichment_entry = 'procedure_enrichment'
    statement_hub_entry = 'statement_hub_builder'
    procedure_hub_entry = 'procedure_hub_builder'
    procedure_hub_exit = 'procedure_hub_builder'
    graph.add_conditional_edges(
        block_corrector_entry,
        block_corrector_node.dispatch,
        ['block_corrector_worker', 'block_corrector_collect'],
    )
    graph.add_edge('block_corrector_worker', 'block_corrector_collect')
    graph.add_conditional_edges(
        formatter_entry,
        formatter_node.dispatch,
        ['formatter_worker', 'formatter_collect'],
    )
    graph.add_edge('formatter_worker', 'formatter_collect')
    graph.add_conditional_edges(
        text_seam_even_entry,
        text_seam_node.dispatch_even,
        ['text_seam_even_worker', 'text_seam_even_collect'],
    )
    graph.add_edge('text_seam_even_worker', 'text_seam_even_collect')
    graph.add_edge('text_seam_even_collect', text_seam_odd_entry)
    graph.add_conditional_edges(
        text_seam_odd_entry,
        text_seam_node.dispatch_odd,
        ['text_seam_odd_worker', 'text_seam_odd_collect'],
    )
    graph.add_edge('text_seam_odd_worker', 'text_seam_odd_collect')
    graph.add_edge('text_seam_odd_collect', image_seam_even_entry)
    graph.add_conditional_edges(
        image_seam_even_entry,
        image_seam_node.dispatch_even,
        ['image_seam_even_worker', 'image_seam_even_collect'],
    )
    graph.add_edge('image_seam_even_worker', 'image_seam_even_collect')
    graph.add_edge('image_seam_even_collect', image_seam_odd_entry)
    graph.add_conditional_edges(
        image_seam_odd_entry,
        image_seam_node.dispatch_odd,
        ['image_seam_odd_worker', 'image_seam_odd_collect'],
    )
    graph.add_edge('image_seam_odd_worker', 'image_seam_odd_collect')
    graph.add_edge('image_seam_odd_collect', 'image_enrichment')
    graph.add_edge('image_enrichment', splitter_entry)
    graph.add_edge('splitter', instruction_entry)
    graph.add_edge('instruction_finder', component_entry)
    graph.add_edge('pedagogical_component_finder', statement_builder_entry)
    graph.add_edge('statement_procedure_builder', 'governance_walker')
    graph.add_edge('governance_walker', triplet_entry)
    graph.add_edge('triplet_extraction', entity_enrichment_entry)
    graph.add_edge('entity_enrichment', predicate_enrichment_entry)
    graph.add_edge('predicate_enrichment', entity_hub_entry)
    graph.add_edge('entity_hub_builder', predicate_hub_entry)
    graph.add_edge('predicate_hub_builder', triplet_hub_entry)
    graph.add_edge('triplet_hub_builder', statement_enrichment_entry)
    graph.add_edge('statement_enrichment', procedure_enrichment_entry)
    graph.add_edge('procedure_enrichment', statement_hub_entry)
    graph.add_edge('statement_hub_builder', procedure_hub_entry)
    graph.add_edge(procedure_hub_exit, 'final_projector')
    graph.add_edge('final_projector', END)
    return graph.compile()
