"""Build the document-construction workflow graph."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from kms import config
from kms.construction import (
    block_corrector,
    entity_enrichment,
    entity_hubs,
    formatter,
    governance_judge,
    governance_walker,
    instruction_finder,
    ocr,
    pedagogical_component_finder,
    predicate_enrichment,
    predicate_hubs,
    procedure_enrichment,
    procedure_hubs,
    seam_merger,
    splitter,
    statement_enrichment,
    statement_hubs,
    statement_procedure_builder,
    triplet_extractor,
    triplet_hubs,
)
from kms.core import llm, recording, state
from kms.graph import projectors

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
        'seam_merger': seam_merger.SeamMerger(
            language_model=llm.module_lm('seam_merger'), recorder=recorder
        ),
        'seam_rewriter': seam_merger.SeamRewriter(
            language_model=llm.module_lm('seam_rewriter'), recorder=recorder
        ),
        'splitter': splitter.Splitter(
            language_model=llm.module_lm('splitter'), recorder=recorder
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
        'statement_hub_builder': statement_hubs.StatementHubSynthesizer(
            language_model=llm.module_lm('statement_hub_builder'),
            recorder=recorder,
        ),
        'statement_hub_adjudicator': statement_hubs.StatementHubAdjudicator(
            language_model=llm.module_lm('statement_hub_builder'),
            recorder=recorder,
        ),
        'procedure_hub_builder': procedure_hubs.ProcedureHubSynthesizer(
            language_model=llm.module_lm('procedure_hub_builder'),
            recorder=recorder,
        ),
        'procedure_hub_adjudicator': procedure_hubs.ProcedureHubAdjudicator(
            language_model=llm.module_lm('procedure_hub_builder'),
            recorder=recorder,
        ),
        'entity_hub_adjudicator': entity_hubs.EntityHubAdjudicator(
            language_model=llm.module_lm('entity_hub_builder'),
        ),
        'entity_hub_builder': entity_hubs.EntityHubSynthesizer(
            language_model=llm.module_lm('entity_hub_builder'),
        ),
        'predicate_hub_adjudicator': predicate_hubs.PredicateHubAdjudicator(
            language_model=llm.module_lm('predicate_hub_builder'),
        ),
        'predicate_hub_builder': predicate_hubs.PredicateHubSynthesizer(
            language_model=llm.module_lm('predicate_hub_builder'),
        ),
        'triplet_hub_builder': llm.module_lm('triplet_hub_builder'),
    }


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
    seam_module = modules['seam_merger']
    seam_rewriter_module = modules['seam_rewriter']
    splitter_module = modules['splitter']
    instruction_router_module = modules['instruction_router']
    instruction_grower_module = modules['instruction_grower']
    component_finder_module = modules['component_finder']
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
    seam_node = seam_merger.SeamMergerNode(
        module=seam_module, rewriter=seam_rewriter_module
    )
    splitter_node = splitter.SplitterNode(module=splitter_module)
    instruction_finder_node = instruction_finder.InstructionFinderNode(
        router=instruction_router_module,
        grower=instruction_grower_module,
    )
    component_finder_node = (
        pedagogical_component_finder.PedagogicalComponentFinderNode(
            module=component_finder_module
        )
    )
    governance_config = config.get_settings().stages.governance
    governance_walker_node = governance_walker.GovernanceStatementWalkerNode(
        judge=modules['governance_judge'],
        backward_budget=governance_config.backward_context_budget,
        forward_budget=governance_config.forward_context_budget,
        threshold=governance_config.threshold,
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
    entity_hub_node = entity_hubs.EntityHubNode(
        modules['entity_hub_adjudicator'],
        modules['entity_hub_builder'],
    )
    predicate_hub_node = predicate_hubs.PredicateHubNode(
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

    statement_hub_node = statement_hubs.StatementHubNode(
        adjudicator=statement_hub_adjudicator,
        synthesizer=statement_hub_module,
    )
    procedure_hub_node = procedure_hubs.ProcedureHubNode(
        adjudicator=procedure_hub_adjudicator,
        synthesizer=procedure_hub_module,
    )

    graph = StateGraph(state.State)
    graph.add_node('ocr', ocr.OCRNode().run)
    graph.add_node('block_corrector_worker', block_corrector_node.worker)
    graph.add_node('block_corrector_collect', block_corrector_node.collect)
    graph.add_node('formatter_worker', formatter_node.worker)
    graph.add_node('formatter_collect', formatter_node.collect)
    graph.add_node('seam_even_worker', seam_node.even_worker)
    graph.add_node('seam_even_collect', seam_node.even_collect)
    graph.add_node('seam_odd_worker', seam_node.odd_worker)
    graph.add_node('seam_odd_collect', seam_node.odd_collect)
    graph.add_node('splitter', splitter_node.run)
    graph.add_node('instruction_finder', instruction_finder_node.run)
    graph.add_node('governance_walker', governance_walker_node.run)
    graph.add_node('pedagogical_component_finder', component_finder_node.run)
    graph.add_node(
        'statement_procedure_builder', statement_procedure_builder_node.run
    )
    graph.add_node('triplet_extraction', triplet_extractor_node.run)
    graph.add_node('entity_enrichment', entity_enrichment_node.run)
    graph.add_node('predicate_enrichment', predicate_enrichment_node.run)

    graph.add_node('entity_hub_builder', entity_hub_node.run)
    graph.add_node('predicate_hub_builder', predicate_hub_node.run)
    graph.add_node('triplet_hub_builder', triplet_hub_node.run)
    graph.add_node('statement_enrichment', statement_enrichment_node.run)
    graph.add_node('procedure_enrichment', procedure_enrichment_node.run)
    graph.add_node('statement_hub_builder', statement_hub_node.run)
    graph.add_node('procedure_hub_builder', procedure_hub_node.run)
    graph.add_node('final_projector', final_projector_node.run)

    graph.add_edge(START, 'ocr')
    block_corrector_entry = 'ocr'
    formatter_entry = 'block_corrector_collect'
    seam_even_entry = 'formatter_collect'
    seam_odd_entry = 'seam_even_collect'
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
        seam_even_entry,
        seam_node.dispatch_even,
        ['seam_even_worker', 'seam_even_collect'],
    )
    graph.add_edge('seam_even_worker', 'seam_even_collect')
    graph.add_conditional_edges(
        seam_odd_entry,
        seam_node.dispatch_odd,
        ['seam_odd_worker', 'seam_odd_collect'],
    )
    graph.add_edge('seam_odd_worker', 'seam_odd_collect')
    graph.add_edge('seam_odd_collect', splitter_entry)
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
