"""
LangGraph wiring for the document-processing pipeline.

Builds the ordered graph that turns a PDF (via the Mistral OCR front-end) into
a finished AST, then assembles it to a single markdown document plus the
statement overlay. The ingestion stages (corrector, extractor, seam merger) are
map-reduce: a conditional edge fans out one Send per unit of work to the
stage's worker, the workers append to a per-stage reducer channel, and the
collect step drains that channel back into the ordered backbone before the next
stage runs. The later stages are plain sequential nodes (a growing look-ahead
cursor cannot be sharded).

Stage order:
    corrector -> formatter -> extractor -> seam_merger (even, odd) -> splitter
              -> instruction_finder -> instruction_distributor
              -> pedagogical_component_finder -> role_typer
              -> statement_extractor -> procedure_extractor
              -> ingestion_persister

Two phases split at the seam merger. Ingestion is per-page: `segments` (already
carrying Mistral's markdown + figures) is the backbone, the corrector proofreads
each page's transcription against its image, and the formatter then standardises
that page's markup — the two are exact inverses, the corrector changing content
but never presentation and the formatter presentation but never content — before
the (purely structural) extractor parses it into nodes. The extractor is also
where page apparatus leaves the document: it identifies running heads, folios
and colophons as `furniture` and discards them inside the stage, so no later
stage — the seam merger first among them — ever sees a node for them. The formatter runs second
because anything that deliberately makes the text diverge from the image has to
come after the pass whose contract is that they agree. The seam merger heals nodes split
across page breaks and then flattens the healed backbone into the global
ordered `nodes` list (stable ids + segment_index). The splitter then normalises
that stream — it rewrites any node that packs several exercises into one node
per exercise (embedded lead-ins broken out onto their own nodes too) — so the
finder sees atomic exercises. The instruction finder then tags every lead-in
node with type INSTRUCTION; the instruction distributor prepends each lead-in's
directive onto the governed exercise nodes and removes the lead-ins from the
stream.

One semantic chain then runs, each stage asking ONE question. The pedagogical
component finder walks `nodes` once and cuts it into UNTYPED spans —
boundaries only, including the cut between a statement and the working that
resolves it, so the old semantic proof/solution boundary call is now a
structural detection. The role typer then labels each span `statement` (a
block) or `procedure` (a derivation), and the statement and procedure
extractors fill in each one's content, which is transcription rather than
judgement. The chain is split this way because fusing these questions made each
one worse — the finder read a missing "Solution." marker as "no derivation".

The ingestion persister is the terminal stage. It runs after every stream
mutation, so the persisted node ids match the overlay's members and instruction
nodes are excluded: it writes the provenance layer (a `:Source` root with its
`:Node` chain), the ``:Statement`` overlay rooted under that ``:Source``, and
the procedural layer (`:Procedure` per derivation, `:Act` per step, threaded
`:FIRST`/`:THEN`). A no-op when Neo4j isn't configured. After the graph
returns, `run()` only assembles the markdown document: assembly walks `nodes`,
consulting `segments` only for picture inventories.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from kms.core import llm, recorder, state
from kms.graph import db, persister
from kms.ingestion import (
    corrector,
    extractor,
    formatter,
    instruction_distributor,
    instruction_finder,
    pedagogical_component_finder,
    procedure_extractor,
    role_typer,
    seam_merger,
    splitter,
    statement_extractor,
)
from kms.output import assembler

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_graph() -> 'CompiledStateGraph':
    """Assemble and compile the LangGraph pipeline over ``state.State``.

    A single straight path: the correction pass proofreads each
    Mistral-transcribed page against its image, the formatter standardises that
    page's markup, the extractor parses the result into structural nodes, the
    seam merger heals page-split
    nodes and flattens to the global stream, and the pedagogical component
    finder then cuts that stream into untyped spans for the role typer and the
    two extractors to classify and fill in.

    Returns:
        The compiled graph.
    """
    # --- DSPy modules ---
    corrector_module = corrector.Corrector(language_model=llm.corrector_lm())
    formatter_module = formatter.Formatter(language_model=llm.text_lm())
    extractor_module = extractor.Extractor(language_model=llm.text_lm())
    seam_module = seam_merger.SeamMerger(language_model=llm.text_lm())
    splitter_module = splitter.Splitter(language_model=llm.text_lm())
    instruction_finder_module = instruction_finder.InstructionFinder(language_model=llm.text_lm())
    instruction_distributor_module = instruction_distributor.InstructionDistributor(
        language_model=llm.text_lm()
    )
    component_finder_module = pedagogical_component_finder.PedagogicalComponentFinder(
        language_model=llm.text_lm()
    )
    role_typer_module = role_typer.RoleTyper(language_model=llm.text_lm())

    # --- LangGraph nodes ---
    corrector_node = corrector.CorrectorNode(module=corrector_module)
    formatter_node = formatter.FormatterNode(module=formatter_module)
    extractor_node = extractor.ExtractorNode(module=extractor_module)
    seam_node = seam_merger.SeamMergerNode(module=seam_module)
    splitter_node = splitter.SplitterNode(module=splitter_module)
    instruction_finder_node = instruction_finder.InstructionFinderNode(
        module=instruction_finder_module
    )
    node_persister_node = persister.IngestionPersisterNode()
    component_finder_node = pedagogical_component_finder.PedagogicalComponentFinderNode(
        module=component_finder_module
    )
    role_typer_node = role_typer.RoleTyperNode(module=role_typer_module)
    statement_extractor_node = statement_extractor.StatementExtractorNode()
    procedure_extractor_node = procedure_extractor.ProcedureExtractorNode()
    instruction_distributor_node = instruction_distributor.InstructionDistributorNode(
        module=instruction_distributor_module
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
    graph.add_node('role_typer', role_typer_node.run)
    graph.add_node('statement_extractor', statement_extractor_node.run)
    graph.add_node('procedure_extractor', procedure_extractor_node.run)

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
    graph.add_edge('pedagogical_component_finder', 'role_typer')
    graph.add_edge('role_typer', 'statement_extractor')
    graph.add_edge('statement_extractor', 'procedure_extractor')
    graph.add_edge('procedure_extractor', 'ingestion_persister')
    graph.add_edge('ingestion_persister', END)

    return graph.compile()


async def run(
    pdf_path: str | Path,
    output_dir: str | Path = 'output',
    filename: str = 'document.md',
    pages: list[int] | None = None,
    source: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> Path:
    """Run the full pipeline on a PDF.

    The Mistral OCR API turns each page into reading-ordered markdown plus
    extracted figures (no GPU, no docling); the graph then corrects, parses,
    heals, builds the statement overlay, and (when Neo4j is configured)
    persists the ``:Node`` provenance layer and the ``:Statement`` overlay plus
    its procedural layer on top of it. Graph persistence is skipped entirely
    when Neo4j isn't configured — a DB-less run still produces ``document.md``
    but persists no nodes or statements.

    Args:
        pdf_path: The source PDF.
        output_dir: Directory the document and its assets are written into.
        filename: Name of the assembled markdown file.
        pages: 0-based pages to limit the OCR request to, or None for all.
        source: The book identity used as the graph's Neo4j key. Defaults to
            the PDF's filename.
        title: Optional book title, stored on the ``:Source`` node.
        author: Optional book author, stored on the ``:Source`` node.

    Returns:
        The path of the assembled document.
    """
    # Deferred so importing the pipeline does not require the OCR extra.
    from kms.ingestion import ocr

    output_dir = Path(output_dir)
    source = source or Path(pdf_path).name
    if os.environ.get('KMS_RECORD'):
        recorder.set_run(
            source,
            output_dir=str(output_dir / 'examples'),
            pdf=str(pdf_path),
            pages=pages,
            title=title,
            author=author,
        )
    metadata = {'title': title, 'author': author}
    segments = ocr.extract(pdf_path, output_dir=output_dir, pages=pages)
    graph = build_graph()
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
        written = assembler.assemble(
            nodes, result['segments'], output_dir=output_dir, filename=filename
        )
        return written
    finally:
        await db.close_driver()
