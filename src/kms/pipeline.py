"""
LangGraph pipeline that turns a PDF into Neo4j graph nodes via Mistral OCR.

Stage order:
    corrector -> formatter -> extractor -> seam_merger (even, odd) -> splitter
              -> instruction_finder -> instruction_distributor
              -> pedagogical_component_finder -> hub_builder
              -> equation_variable -> atomic_facts -> fact_embedding
              -> ingestion_persister

Two phases. Ingestion is per-page map-reduce: the corrector proofreads each
page's transcription against its image, the formatter standardises markup, the
extractor parses it into structural nodes, and the seam merger heals cross-page
splits then flattens the backbone into the global ``nodes`` stream. The
splitter then normalises packed-exercise nodes, the instruction finder tags
lead-ins, and the distributor prepends directives onto governed exercises.

One semantic chain follows. The pedagogical component finder cuts the stream
into untyped spans; the hub builder classifies each span's roles and partitions
both-blocks in one pass (router + gated partitioners). The equation/variable
node extracts equations and variable bindings per provenance node, feeding
equations as context into the variable extractor. The atomic fact pass then
decomposes the final node stream into atomic facts for the downstream concept
and relation passes, and the fact embedding stage enriches them with vectors
in one batched pass.

The ingestion persister writes everything: a ``:Source`` root, its ``:Node``
provenance chain, ``:Statement`` and ``:Procedure`` hubs hung off member nodes
via ``:MEMBER_OF``, and ``:Equation`` / ``:Variable`` hung off their provenance
``:Node`` via ``:HAS_EQUATION`` / ``:HAS_VARIABLE``. A no-op when Neo4j isn't
configured. After the graph returns, ``run()`` assembles the markdown string
and returns it without writing to disk.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import dspy
from langgraph.graph import END, START, StateGraph

from kms.core import embeddings, llm, state
from kms.core.recorder import Recorder
from kms.graph import db, persister
from kms.ingestion import (
    atomic_fact_extractor,
    corrector,
    extractor,
    fact_embedder,
    formatter,
    hub_builder,
    instruction_distributor,
    instruction_finder,
    pedagogical_component_finder,
    seam_merger,
    splitter,
    variable_extractor,
)
from kms.output import assembler

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_graph(
    text_language_model: dspy.LM,
    corrector_language_model: dspy.LM,
    *,
    recorder: Recorder | None = None,
    neo4j_session_factory: Callable | None = None,
    neo4j_configured: bool = False,
) -> 'CompiledStateGraph':
    """Assemble and compile the LangGraph pipeline over ``state.State``.

    A single straight path: the correction pass proofreads each
    Mistral-transcribed page against its image, the formatter standardises that
    page's markup, the extractor parses the result into structural nodes, the
    seam merger heals page-split
    nodes and flattens to the global stream, and the pedagogical component
    finder then cuts that stream into untyped spans for the hub builder to
    label and partition in one pass — before the equation/variable node pulls
    equations and bindings out and the persister writes every tier.

    Args:
        text_language_model: The language model for all text-reasoning
            stages (extractor, formatter, hub builder, variable
            extractor, etc.).
        corrector_language_model: The vision-capable language model for
            the correction pass.
        recorder: Optional recorder for capturing DSPy training examples.
        neo4j_session_factory: A callable that returns an async context
            manager with a ``run(query, **params)`` method.
        neo4j_configured: Whether a Neo4j target is wired.

    Returns:
        The compiled graph.
    """
    # --- DSPy modules ---
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
    instruction_distributor_module = (
        instruction_distributor.InstructionDistributor(
            language_model=text_language_model,
            recorder=recorder,
        )
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
    router_module = variable_extractor.Router(
        language_model=text_language_model,
        recorder=recorder,
    )
    equation_module = variable_extractor.EquationExtractor(
        language_model=text_language_model,
        recorder=recorder,
    )
    variable_module = variable_extractor.VariableExtractor(
        language_model=text_language_model,
        recorder=recorder,
    )
    atomic_fact_module = atomic_fact_extractor.AtomicFactExtractor(
        language_model=text_language_model,
        recorder=recorder,
    )

    # --- LangGraph nodes ---
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
    equation_variable_node = variable_extractor.EquationAndVariableNode(
        router_module=router_module,
        equation_module=equation_module,
        variable_module=variable_module,
    )
    atomic_fact_node = atomic_fact_extractor.AtomicFactNode(
        module=atomic_fact_module
    )
    fact_embedder_node = fact_embedder.FactEmbedderNode(
        embedder=embeddings.embedder(),
        embedding_configured=embeddings.is_configured(),
    )
    instruction_distributor_node = (
        instruction_distributor.InstructionDistributorNode(
            module=instruction_distributor_module
        )
    )

    graph = StateGraph(state.State)

    # Each stage registers its worker (Send target) and collect (drain) nodes.
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
    graph.add_node(
        'instruction_distributor',
        instruction_distributor_node.run,
    )
    graph.add_node('ingestion_persister', node_persister_node.run)
    graph.add_node('pedagogical_component_finder', component_finder_node.run)
    graph.add_node('hub_builder', hub_builder_node.run)
    graph.add_node('equation_variable', equation_variable_node.run)
    graph.add_node('atomic_facts', atomic_fact_node.run)
    graph.add_node('fact_embedding', fact_embedder_node.run)

    # A stage's dispatch is a conditional edge off the previous collect: it
    # either fans out Sends to the worker or short-circuits straight to its own
    # collect.
    graph.add_conditional_edges(
        START,
        corrector_node.dispatch,
        ['corrector_worker', 'corrector_collect'],
    )
    graph.add_edge('corrector_worker', 'corrector_collect')

    # Formatting runs after correction, never before: anything that
    # deliberately makes the text diverge from the page image must come after
    # the pass whose contract is that the two agree.
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

    # Seam healing: even pass then odd pass, so no two concurrent workers touch
    # the same segment (see seam_merger's parity note).
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

    # The instruction finder tags exercise lead-in nodes with type INSTRUCTION.
    # The instruction distributor prepends each lead-in's directive onto its
    # governed exercises and removes the instruction nodes from the stream.
    graph.add_edge('splitter', 'instruction_finder')
    graph.add_edge('instruction_finder', 'instruction_distributor')

    # The persister runs last, after every stream mutation, so the persisted
    # node ids match the overlay's members and instruction nodes are excluded.
    graph.add_edge('instruction_distributor', 'pedagogical_component_finder')
    graph.add_edge('pedagogical_component_finder', 'hub_builder')
    graph.add_edge('hub_builder', 'equation_variable')
    graph.add_edge('equation_variable', 'atomic_facts')
    graph.add_edge('atomic_facts', 'fact_embedding')
    graph.add_edge('fact_embedding', 'ingestion_persister')
    graph.add_edge('ingestion_persister', END)

    return graph.compile()


async def run(
    pdf_path: str | Path,
    output_dir: str | Path = 'output',
    pages: list[int] | None = None,
    source: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> str:
    """Run the full pipeline on a PDF.

    The Mistral OCR API turns each page into reading-ordered markdown plus
    extracted figures (no GPU, no docling); the graph then corrects, parses,
    heals, builds the statement overlay, extracts equations and variable
    bindings and atomic facts, embeds the facts, and (when Neo4j is
    configured) persists the ``:Node`` provenance
    layer, the ``:Statement`` overlay, and the procedural, equation and
    variable layers on top of it. Graph persistence is skipped entirely
    when Neo4j isn't configured — a DB-less run still returns the assembled
    markdown but persists no nodes or statements.

    Args:
        pdf_path: The source PDF.
        output_dir: Directory the document's assets are written into.
        pages: 0-based pages to limit the OCR request to, or None for all.
        source: The book identity used as the graph's Neo4j key. Defaults to
            the PDF's filename.
        title: Optional book title, stored on the ``:Source`` node.
        author: Optional book author, stored on the ``:Source`` node.

    Returns:
        The assembled markdown document as a string.
    """
    # Deferred so importing the pipeline does not require the OCR extra.
    from kms.ingestion import ocr

    output_dir = Path(output_dir)
    source = source or Path(pdf_path).name

    # -- Wiring: construct every injectable dependency --------------------
    example_recorder = None
    if os.environ.get('KMS_RECORD'):
        example_recorder = Recorder(
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
            return db.driver().session(database=db.database())
    else:
        neo4j_session_factory = None

    metadata = {'title': title, 'author': author}
    segments = ocr.extract(pdf_path, output_dir=output_dir, pages=pages)
    text_language_model = llm.text_lm()
    corrector_language_model = llm.corrector_lm()
    graph = build_graph(
        text_language_model=text_language_model,
        corrector_language_model=corrector_language_model,
        recorder=example_recorder,
        neo4j_session_factory=neo4j_session_factory,
        neo4j_configured=neo4j_configured,
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
        nodes = result['nodes']
        return assembler.assemble(
            nodes, result['segments'], output_dir=output_dir
        )
    finally:
        await db.close_driver()
        await embeddings.embedder().aclose()
