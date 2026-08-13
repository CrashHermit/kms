import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import dspy
from langgraph.graph import END, START, StateGraph

from kms.core import llm, recording, state
from kms.graph import db, persister
from kms.ingestion import (
    corrector,
    entity_canonicalizer,
    extractor,
    formatter,
    hub_builder,
    instruction_finder,
    ocr,
    pedagogical_component_finder,
    procedure_creator,
    seam_merger,
    splitter,
    triplet_extractor,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class EntityCanonicalizerNode:
    def __init__(
        self,
        session_factory,
        language_model,
    ) -> None:
        self._session_factory = session_factory
        self._language_model = language_model

    async def run(self, state: state.State) -> dict:
        if not self._session_factory:
            return {}
        result = await entity_canonicalizer.rebuild(
            threshold=0.8,
            language_model=self._language_model,
            session_factory=self._session_factory,
        )
        return {'entity_clusters': result['clusters']}


class ProcedureCreatorNode:
    def __init__(
        self,
        session_factory,
        language_model,
        creator_model=None,
    ) -> None:
        self._session_factory = session_factory
        self._language_model = language_model
        self._creator_model = creator_model

    async def run(self, state: state.State) -> dict:
        if not self._session_factory:
            return {}
        created = await procedure_creator.create_procedures(
            self._session_factory,
            language_model=self._creator_model or self._language_model,
        )
        return {'procedures_created': created}


def build_graph(
    text_language_model: dspy.LM,
    corrector_language_model: dspy.LM,
    *,
    recorder: recording.Recorder | None = None,
    neo4j_session_factory: Callable | None = None,
    neo4j_configured: bool = False,
    procedure_creator_model: dspy.LM | None = None,
) -> 'CompiledStateGraph':
    corrector_module = corrector.Corrector(
        language_model=corrector_language_model,
        recorder=recorder,
    )
    formatter_module = formatter.Formatter(
        language_model=text_language_model,
        recorder=recorder,
    )
    extractor_module = extractor.Extractor(
        language_model=text_language_model,
        recorder=recorder,
    )
    seam_module = seam_merger.SeamMerger(
        language_model=text_language_model,
        recorder=recorder,
    )
    seam_rewriter_module = seam_merger.SeamRewriter(
        language_model=text_language_model,
        recorder=recorder,
    )
    splitter_module = splitter.Splitter(
        language_model=text_language_model,
        recorder=recorder,
    )
    instruction_finder_module = instruction_finder.InstructionFinder(
        language_model=text_language_model,
        recorder=recorder,
    )
    component_finder_module = (
        pedagogical_component_finder.PedagogicalComponentFinder(
            language_model=text_language_model,
            recorder=recorder,
        )
    )
    role_typer_module = hub_builder.RoleTyper(
        language_model=text_language_model,
        recorder=recorder,
    )
    statement_partitioner_module = hub_builder.StatementPartitioner(
        language_model=text_language_model,
        recorder=recorder,
    )
    procedure_partitioner_module = hub_builder.ProcedurePartitioner(
        language_model=text_language_model,
        recorder=recorder,
    )
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
    node_persister_node = persister.IngestionPersisterNode(
        session_factory=neo4j_session_factory,
        neo4j_configured=neo4j_configured,
    )
    component_finder_node = (
        pedagogical_component_finder.PedagogicalComponentFinderNode(
            module=component_finder_module
        )
    )
    hub_builder_node = hub_builder.HubBuilderNode(
        role_module=role_typer_module,
        statement_partitioner=statement_partitioner_module,
        procedure_partitioner=procedure_partitioner_module,
    )
    triplet_extractor_node = triplet_extractor.TripletNode(
        language_model=text_language_model,
        recorder=recorder,
    )
    canonicalizer_node = EntityCanonicalizerNode(
        session_factory=neo4j_session_factory,
        language_model=text_language_model,
    )
    procedure_creator_node = ProcedureCreatorNode(
        session_factory=neo4j_session_factory,
        language_model=text_language_model,
        creator_model=procedure_creator_model,
    )

    graph = StateGraph(state.State)
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
    graph.add_node('ingestion_persister', node_persister_node.run)
    graph.add_node('pedagogical_component_finder', component_finder_node.run)
    graph.add_node('hub_builder', hub_builder_node.run)
    graph.add_node('triplet_extraction', triplet_extractor_node.run)
    graph.add_node('entity_canonicalizer', canonicalizer_node.run)
    graph.add_node('procedure_creator', procedure_creator_node.run)
    graph.add_conditional_edges(
        START,
        corrector_node.dispatch,
        ['corrector_worker', 'corrector_collect'],
    )
    graph.add_edge('corrector_worker', 'corrector_collect')
    graph.add_conditional_edges(
        'corrector_collect',
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
    graph.add_edge('pedagogical_component_finder', 'hub_builder')
    graph.add_edge('hub_builder', 'triplet_extraction')
    graph.add_edge('triplet_extraction', 'ingestion_persister')
    graph.add_edge('ingestion_persister', 'entity_canonicalizer')
    graph.add_edge('entity_canonicalizer', 'procedure_creator')
    graph.add_edge('procedure_creator', END)

    return graph.compile()


async def run(
    pdf_path: str | Path,
    output_dir: str | Path = 'output',
    pages: list[int] | None = None,
    source: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> dict:
    output_dir = Path(output_dir)
    source = source or Path(pdf_path).name
    example_recorder = None
    if os.environ.get('KMS_RECORD'):
        example_recorder = recording.Recorder(
            source,
            output_dir=str(output_dir / 'examples'),
            pdf=str(pdf_path),
            pages=pages,
            title=title,
            author=author,
        )

    neo4j_configured = db.is_configured()
    if neo4j_configured:

        def neo4j_session_factory():
            return db.session()
    else:
        neo4j_session_factory = None

    metadata = {'title': title, 'author': author}
    segments = ocr.extract(pdf_path, output_dir=output_dir, pages=pages)
    text_language_model = llm.pipeline_lm()
    corrector_language_model = llm.corrector_lm()
    procedure_creator_language_model = llm.procedure_creator_lm()
    graph = build_graph(
        text_language_model=text_language_model,
        corrector_language_model=corrector_language_model,
        recorder=example_recorder,
        neo4j_session_factory=neo4j_session_factory,
        neo4j_configured=neo4j_configured,
        procedure_creator_model=procedure_creator_language_model,
    )
    try:
        result = await graph.ainvoke(
            {
                'segments': segments,
                'source': source,
                'source_metadata': metadata,
            },
            {'recursion_limit': 1000},
        )
        return result
    finally:
        await db.close_driver()
