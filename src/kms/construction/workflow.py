"""Build the document-construction workflow graph."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from kms import config
from kms.construction import (
    block_corrector,
    enrichment,
    enrichment_inputs,
    entity_enrichment,
    entity_hubs,
    formatter,
    hub_inputs,
    instruction_finder,
    ocr,
    pedagogical_component_finder,
    predicate_enrichment,
    predicate_hubs,
    procedure_creator,
    procedure_hubs,
    procedure_inputs,
    seam_merger,
    splitter,
    statement_hubs,
    statement_procedure_builder,
    triplet_extractor,
    triplet_hubs,
)
from kms.core import llm, recording, serve, state
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


class ProcedureCreatorNode:
    """Creates learnable procedures from source-scoped inputs."""

    def __init__(self, language_model) -> None:
        self._language_model = language_model

    async def run(self, current_state: state.State) -> dict:
        """Creates procedures without graph reads or writes."""
        bundle = state.to_construction_bundle(current_state)
        result = await procedure_creator.create_procedures(
            bundle.procedure_materialization_inputs,
            language_model=self._language_model,
        )
        bundle.generated_procedures = result.get('generated_procedures', [])
        bundle.procedure_step_updates = result.get('procedure_step_updates', [])
        bundle.procedure_links = result.get('procedure_links', [])
        bundle.procedures.extend(bundle.generated_procedures)
        return {
            **result,
            'procedures': bundle.procedures,
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
        'statement_enrichment': enrichment.StatementEnricher(
            language_model=llm.module_lm('statement_enrichment'),
            recorder=recorder,
        ),
        'procedure_enrichment': enrichment.ProcedureEnricher(
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

_MANAGED_MODEL_GROUPS = (
    ('seam_merger', 'seam_rewriter'),
    ('instruction_router', 'instruction_grower'),
    ('role_typer', 'statement_partitioner', 'procedure_partitioner'),
    ('atomic_fact_extractor', 'triplet_extractor'),
)


def _validate_managed_model_groups() -> None:
    """Ensure compound nodes use one resident serving model."""
    module_models = config.get_settings().serving.module_models
    for group in _MANAGED_MODEL_GROUPS:
        configured = {module_models.get(name) for name in group}
        if None in configured or len(configured) != 1:
            raise RuntimeError(
                'Managed compound stage requires one serving model: '
                f'{group!r} -> {sorted(configured, key=str)!r}'
            )


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
    if model_manager:
        _validate_managed_model_groups()
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
    statement_enrichment_input_node = (
        enrichment_inputs.StatementEnrichmentInputNode()
    )
    procedure_enrichment_input_node = (
        enrichment_inputs.ProcedureEnrichmentInputNode()
    )
    procedure_creator_node = ProcedureCreatorNode(
        language_model=llm.module_lm('procedure_creator'),
    )
    statement_enrichment_node = enrichment.StatementEnrichmentNode(
        enricher=statement_enrichment_module,
    )
    procedure_enrichment_node = enrichment.ProcedureEnrichmentNode(
        enricher=procedure_enrichment_module,
    )
    procedure_materialization_input_node = (
        procedure_inputs.ProcedureMaterializationInputNode()
    )
    hub_input_node = hub_inputs.HubInputNode()
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
    graph.add_node('pedagogical_component_finder', component_finder_node.run)
    graph.add_node(
        'statement_procedure_builder', statement_procedure_builder_node.run
    )
    graph.add_node('triplet_extraction', triplet_extractor_node.run)
    graph.add_node('entity_enrichment', entity_enrichment_node.run)
    graph.add_node('predicate_enrichment', predicate_enrichment_node.run)
    graph.add_node('hub_input', hub_input_node.run)
    graph.add_node('entity_hub_builder', entity_hub_node.run)
    graph.add_node('predicate_hub_builder', predicate_hub_node.run)
    graph.add_node('triplet_hub_builder', triplet_hub_node.run)
    graph.add_node(
        'statement_enrichment_input', statement_enrichment_input_node.run
    )
    graph.add_node(
        'procedure_enrichment_input', procedure_enrichment_input_node.run
    )
    graph.add_node(
        'procedure_materialization_input',
        procedure_materialization_input_node.run,
    )
    graph.add_node('procedure_creator', procedure_creator_node.run)
    graph.add_node('statement_enrichment', statement_enrichment_node.run)
    graph.add_node('procedure_enrichment', procedure_enrichment_node.run)
    graph.add_node('statement_hub_builder', statement_hub_node.run)
    graph.add_node('procedure_hub_builder', procedure_hub_node.run)
    graph.add_node('final_projector', final_projector_node.run)

    block_corrector_entry = 'ocr'
    formatter_entry = 'block_corrector_collect'
    entity_enrichment_entry = 'entity_enrichment'
    predicate_enrichment_entry = 'predicate_enrichment'
    entity_hub_entry = 'entity_hub_builder'
    predicate_hub_entry = 'predicate_hub_builder'
    triplet_hub_entry = 'triplet_hub_builder'
    statement_enrichment_entry = 'statement_enrichment'
    procedure_creator_entry = 'procedure_creator'
    procedure_enrichment_entry = 'procedure_enrichment'
    statement_hub_entry = 'statement_hub_builder'
    procedure_hub_entry = 'procedure_hub_builder'
    procedure_hub_exit = 'procedure_hub_builder'
    seam_even_entry = 'formatter_collect'
    seam_odd_entry = 'seam_even_collect'
    splitter_entry = 'splitter'
    instruction_entry = 'instruction_finder'
    component_entry = 'pedagogical_component_finder'
    statement_builder_entry = 'statement_procedure_builder'
    triplet_entry = 'triplet_extraction'
    if model_manager:
        graph.add_node(
            'switch_to_corrector',
            serve.SwitchNode(model_manager, 'corrector').run,
        )
        graph.add_node(
            'switch_to_formatter',
            serve.SwitchNode(model_manager, 'formatter').run,
        )
        graph.add_node(
            'switch_to_seam_merger',
            serve.SwitchNode(model_manager, 'seam_merger').run,
        )
        graph.add_node(
            'switch_to_splitter',
            serve.SwitchNode(model_manager, 'splitter').run,
        )
        graph.add_node(
            'switch_to_instruction_finder',
            serve.SwitchNode(model_manager, 'instruction_router').run,
        )
        graph.add_node(
            'switch_to_component_finder',
            serve.SwitchNode(
                model_manager, 'pedagogical_component_finder'
            ).run,
        )
        graph.add_node(
            'switch_to_statement_builder',
            serve.SwitchNode(model_manager, 'role_typer').run,
        )
        graph.add_node(
            'switch_to_triplet_extraction',
            serve.SwitchNode(model_manager, 'atomic_fact_extractor').run,
        )
        graph.add_node(
            'switch_to_entity_enrichment',
            serve.SwitchNode(model_manager, 'entity_enrichment').run,
        )
        graph.add_node(
            'switch_to_predicate_enrichment',
            serve.SwitchNode(model_manager, 'predicate_enrichment').run,
        )
        graph.add_node(
            'switch_to_entity_hub_builder',
            serve.SwitchNode(model_manager, 'entity_hub_builder').run,
        )
        graph.add_edge(
            'switch_to_entity_hub_builder', 'entity_hub_builder'
        )
        graph.add_node(
            'switch_to_predicate_hub_builder',
            serve.SwitchNode(model_manager, 'predicate_hub_builder').run,
        )
        graph.add_node(
            'switch_to_triplet_hub_builder',
            serve.SwitchNode(model_manager, 'triplet_hub_builder').run,
        )
        graph.add_node(
            'switch_to_statement_enrichment',
            serve.SwitchNode(model_manager, 'statement_enrichment').run,
        )
        graph.add_node(
            'switch_to_procedure_enrichment',
            serve.SwitchNode(model_manager, 'procedure_enrichment').run,
        )
        graph.add_node(
            'switch_to_procedure_creator',
            serve.SwitchNode(model_manager, 'procedure_creator').run,
        )
        graph.add_node(
            'switch_to_statement_hub_builder',
            serve.SwitchNode(model_manager, 'statement_hub_builder').run,
        )
        graph.add_node(
            'switch_to_procedure_hub_builder',
            serve.SwitchNode(model_manager, 'procedure_hub_builder').run,
        )
        graph.add_edge(START, 'ocr')
        graph.add_edge('ocr', 'switch_to_corrector')
        graph.add_edge('block_corrector_collect', 'switch_to_formatter')
        graph.add_edge('formatter_collect', 'switch_to_seam_merger')
        graph.add_edge('seam_even_collect', 'switch_to_seam_merger')
        graph.add_edge('seam_odd_collect', 'switch_to_splitter')
        graph.add_edge('splitter', 'switch_to_instruction_finder')
        graph.add_edge(
            'instruction_finder', 'switch_to_component_finder'
        )
        graph.add_edge(
            'pedagogical_component_finder', 'switch_to_statement_builder'
        )
        graph.add_edge(
            'statement_procedure_builder', 'switch_to_triplet_extraction'
        )
        graph.add_edge(
            'switch_to_entity_enrichment', 'entity_enrichment'
        )
        graph.add_edge(
            'switch_to_predicate_enrichment', 'predicate_enrichment'
        )
        graph.add_edge(
            'switch_to_entity_hub_builder', 'entity_hub_builder'
        )
        graph.add_edge(
            'switch_to_predicate_hub_builder', 'predicate_hub_builder'
        )
        graph.add_edge(
            'switch_to_triplet_hub_builder', 'triplet_hub_builder'
        )
        graph.add_edge(
            'switch_to_statement_enrichment', 'statement_enrichment'
        )
        graph.add_edge(
            'switch_to_procedure_creator', 'procedure_creator'
        )
        graph.add_edge(
            'switch_to_procedure_enrichment', 'procedure_enrichment'
        )
        graph.add_edge(
            'switch_to_statement_hub_builder', 'statement_hub_builder'
        )
        graph.add_edge(
            'switch_to_procedure_hub_builder', 'procedure_hub_builder'
        )
        block_corrector_entry = 'switch_to_corrector'
        formatter_entry = 'switch_to_formatter'
        statement_enrichment_entry = 'switch_to_statement_enrichment'
        procedure_creator_entry = 'switch_to_procedure_creator'
        procedure_enrichment_entry = 'switch_to_procedure_enrichment'
        entity_enrichment_entry = 'switch_to_entity_enrichment'
        predicate_enrichment_entry = 'switch_to_predicate_enrichment'
        entity_hub_entry = 'switch_to_entity_hub_builder'
        predicate_hub_entry = 'switch_to_predicate_hub_builder'
        triplet_hub_entry = 'switch_to_triplet_hub_builder'
        statement_hub_entry = 'switch_to_statement_hub_builder'
        procedure_hub_entry = 'switch_to_procedure_hub_builder'
        procedure_hub_exit = 'procedure_hub_builder'
        seam_even_entry = 'switch_to_seam_merger'
        seam_odd_entry = 'switch_to_seam_merger'
        splitter_entry = 'switch_to_splitter'
        instruction_entry = 'switch_to_instruction_finder'
        component_entry = 'switch_to_component_finder'
        statement_builder_entry = 'switch_to_statement_builder'
        triplet_entry = 'switch_to_triplet_extraction'
    if not model_manager:
        graph.add_edge(START, 'ocr')
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
    if model_manager:
        graph.add_edge('switch_to_splitter', 'splitter')
    graph.add_edge('splitter', instruction_entry)
    if model_manager:
        graph.add_edge(
            'switch_to_instruction_finder', 'instruction_finder'
        )
    graph.add_edge('instruction_finder', component_entry)
    if model_manager:
        graph.add_edge(
            'switch_to_component_finder', 'pedagogical_component_finder'
        )
    graph.add_edge('pedagogical_component_finder', statement_builder_entry)
    if model_manager:
        graph.add_edge(
            'switch_to_statement_builder', 'statement_procedure_builder'
        )
    graph.add_edge('statement_procedure_builder', triplet_entry)
    if model_manager:
        graph.add_edge(
            'switch_to_triplet_extraction', 'triplet_extraction'
        )
    graph.add_edge('triplet_extraction', entity_enrichment_entry)
    graph.add_edge('entity_enrichment', predicate_enrichment_entry)
    graph.add_edge('predicate_enrichment', 'hub_input')
    graph.add_edge('hub_input', entity_hub_entry)
    graph.add_edge('entity_hub_builder', predicate_hub_entry)
    graph.add_edge('predicate_hub_builder', triplet_hub_entry)
    graph.add_edge('triplet_hub_builder', 'statement_enrichment_input')
    graph.add_edge('statement_enrichment_input', statement_enrichment_entry)
    graph.add_edge('statement_enrichment', 'procedure_materialization_input')
    graph.add_edge('procedure_materialization_input', procedure_creator_entry)
    graph.add_edge('procedure_creator', 'procedure_enrichment_input')
    graph.add_edge('procedure_enrichment_input', procedure_enrichment_entry)
    graph.add_edge('procedure_enrichment', statement_hub_entry)
    graph.add_edge('statement_hub_builder', procedure_hub_entry)
    graph.add_edge(procedure_hub_exit, 'final_projector')
    graph.add_edge('final_projector', END)
    return graph.compile()
