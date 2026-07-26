"""
LangGraph wiring for the document-processing pipeline.

Builds the ordered graph that turns a PDF (via the Mistral OCR front-end) into a
finished AST, then assembles it to a single markdown document plus the block overlay.
The ingestion stages (corrector, extractor, seam merger) are map-reduce: a conditional
edge fans out one Send per unit of work to the stage's worker, the workers append to a
per-stage reducer channel, and the collect step drains that channel back into the
ordered backbone before the next stage runs. The entity stages are plain sequential
nodes (a growing look-ahead cursor cannot be sharded).

Stage order:
    corrector -> extractor -> seam_merger (even, odd) -> splitter -> instruction_finder
              -> node_persister -> group_finder -> role_typer -> block_typer
              -> statement_extractor -> procedure_extractor -> instruction_distributor
              -> entity_persister

Two phases split at the seam merger. Ingestion is per-page: `segments` (already carrying
Mistral's markdown + figures) is the backbone, and the corrector proofreads each page's
transcription against its image before the (purely structural) extractor parses it into
nodes. The seam merger heals nodes split across page breaks and then flattens the healed
backbone into the global ordered `nodes` list (stable ids + segment_index). The splitter then
normalises that stream — it rewrites any node that packs several exercises into one node per
exercise (embedded lead-ins broken out onto their own nodes too) — so the finder sees atomic
exercises. The instruction finder then tags every lead-in node `role="instruction"` over that
atomic stream, one uniform pass. The node persister then writes the finalized stream to Neo4j as
the graph's provenance layer (a `:Source` root with its `:Node` chain); it runs after the splitter
so the persisted ids match the overlay's `members`, and is a no-op when Neo4j isn't configured.

One entity chain then runs, each stage asking ONE question. The group finder walks `nodes` once and
cuts it into UNTYPED spans — boundaries only, including the cut between a statement and the working
that resolves it, so the old semantic proof/solution boundary call is now a structural detection.
The role typer then labels each span `entity` (a block) or `procedure` (a derivation); the block
typer induces each block's open `type`; and the statement extractor fills the remaining attributes
(label, number, title, contents), which are transcription rather than judgement. The chain is split
this way because fusing these questions made each one worse — the finder read a missing "Solution."
marker as "no derivation", and a shared type/label call typed problem-set items by their subject
matter. The procedure extractor then decomposes every procedure span into verbatim ordered steps and
attaches it to the block it derives. Decomposition is universal: a solution's steps are as real as a
proof's. The
instruction distributor then stamps `instruction` from the lead-in tags (the shared directive of a
grouped-exercise run), which is what makes an atomic exercise mean anything on its own.

The entity persister is the terminal stage: it orders the overlay into one document-ordered,
globally-id'd list and upserts it as the graph's `:Entity` layer (rooted under the `:Source`,
linked back to its member `:Node` chunks), then the procedural layer (`:Procedure` per derivation,
`:Act` per step, threaded `:FIRST`/`:THEN`). A no-op when Neo4j isn't configured. The concept layer
is currently dark. After the graph returns, `run()` only assembles the markdown document: assembly
walks `nodes`, consulting `segments` only for picture inventories.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from kms.core import state, tracing
from kms.entity.block_typer import BlockTyperNode
from kms.entity.group_finder import GroupFinderNode
from kms.entity.instruction_distributor import InstructionDistributorNode
from kms.entity.instruction_finder import InstructionFinderNode
from kms.entity.procedure_extractor import ProcedureExtractorNode
from kms.entity.role_typer import RoleTyperNode
from kms.entity.splitter import SplitterNode
from kms.entity.statement_extractor import StatementExtractorNode
from kms.graph.db import close_driver
from kms.graph.persister import EntityPersisterNode, NodePersisterNode
from kms.ingestion.corrector import CorrectorNode
from kms.ingestion.extractor import ExtractorNode
from kms.ingestion.seam_merger import SeamMergerNode
from kms.output.assembler import assemble

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_graph() -> 'CompiledStateGraph':
    """Assemble and compile the LangGraph pipeline over the shared state.State.

    A single straight path: the correction pass proofreads each Mistral-transcribed
    page against its image, the extractor parses the corrected markdown into structural
    nodes, the seam merger heals page-split nodes and flattens to the global stream, and
    the group finder then cuts that stream into untyped spans for the role typer, the block
    typer and the two extractors to classify and fill in.
    """
    # Hooked here rather than in run(), so EVERY entry point traces — a caller that drives
    # the compiled graph directly (a harness, a notebook) gets capture too. Idempotent, and
    # a no-op unless KMS_TRACE_DIR is set.
    tracing.enable_from_env()

    corrector = CorrectorNode()
    extractor = ExtractorNode()
    seam = SeamMergerNode()
    splitter = SplitterNode()
    instruction_finder = InstructionFinderNode()
    node_persister = NodePersisterNode()
    group_finder = GroupFinderNode()
    role_typer = RoleTyperNode()
    block_typer = BlockTyperNode()
    statement_extractor = StatementExtractorNode()
    procedure_extractor = ProcedureExtractorNode()
    instruction_distributor = InstructionDistributorNode()
    entity_persister = EntityPersisterNode()

    graph = StateGraph(state.State)

    # Each stage registers its worker (Send target) and collect (drain) nodes.
    graph.add_node('corrector_worker', corrector.worker)
    graph.add_node('corrector_collect', corrector.collect)
    graph.add_node('extractor_worker', extractor.worker)
    graph.add_node('extractor_collect', extractor.collect)
    graph.add_node('seam_even_worker', seam.even_worker)
    graph.add_node('seam_even_collect', seam.even_collect)
    graph.add_node('seam_odd_worker', seam.odd_worker)
    graph.add_node('seam_odd_collect', seam.odd_collect)
    graph.add_node('splitter', splitter.run)
    graph.add_node('instruction_finder', instruction_finder.run)
    graph.add_node('node_persister', node_persister.run)
    graph.add_node('group_finder', group_finder.run)
    graph.add_node('role_typer', role_typer.run)
    graph.add_node('block_typer', block_typer.run)
    graph.add_node('statement_extractor', statement_extractor.run)
    graph.add_node('procedure_extractor', procedure_extractor.run)
    graph.add_node('instruction_distributor', instruction_distributor.run)
    graph.add_node('entity_persister', entity_persister.run)

    # A stage's dispatch is a conditional edge off the previous collect: it either fans
    # out Sends to the worker or short-circuits straight to its own collect.
    graph.add_conditional_edges(
        START, corrector.dispatch, ['corrector_worker', 'corrector_collect']
    )
    graph.add_edge('corrector_worker', 'corrector_collect')

    graph.add_conditional_edges(
        'corrector_collect',
        extractor.dispatch,
        ['extractor_worker', 'extractor_collect'],
    )
    graph.add_edge('extractor_worker', 'extractor_collect')

    # Seam healing: even pass then odd pass, so no two concurrent workers touch the
    # same segment (see seam_merger's parity note).
    graph.add_conditional_edges(
        'extractor_collect',
        seam.dispatch_even,
        ['seam_even_worker', 'seam_even_collect'],
    )
    graph.add_edge('seam_even_worker', 'seam_even_collect')
    graph.add_conditional_edges(
        'seam_even_collect',
        seam.dispatch_odd,
        ['seam_odd_worker', 'seam_odd_collect'],
    )
    graph.add_edge('seam_odd_worker', 'seam_odd_collect')

    # The splitter runs once after the seam collect, normalising the node stream so each
    # exercise (and each embedded lead-in) is its own node before the finder walks it.
    graph.add_edge('seam_odd_collect', 'splitter')

    # The instruction finder then tags every lead-in node `role="instruction"` over the
    # now-atomic stream — one uniform pass, standalone and embedded lead-ins alike.
    graph.add_edge('splitter', 'instruction_finder')

    # Persist the finalized node stream as the graph's provenance layer BEFORE the finder runs.
    # It sits after the splitter (which re-ids the stream) and the instruction finder so the
    # persisted node ids match the overlay's members. A no-op when Neo4j isn't configured.
    graph.add_edge('instruction_finder', 'node_persister')

    # One entity chain, sequential. The finder's cursor-walk is not shardable, and the two
    # extractors both write the `entities` channel, so sequencing avoids a reducer clash for no
    # meaningful latency cost. The instruction distributor runs last because it reads each block's
    # contents/number (which the statement extractor fills) to judge governance.
    graph.add_edge('node_persister', 'group_finder')
    graph.add_edge('group_finder', 'role_typer')
    graph.add_edge('role_typer', 'block_typer')
    graph.add_edge('block_typer', 'statement_extractor')
    graph.add_edge('statement_extractor', 'procedure_extractor')
    graph.add_edge('procedure_extractor', 'instruction_distributor')
    graph.add_edge('instruction_distributor', 'entity_persister')

    # The entity persister orders the overlay into one document-ordered list and upserts it as the
    # graph's `:Entity` layer plus the procedural layer on top (a no-op when Neo4j isn't
    # configured). It is the pipeline's terminal stage.
    graph.add_edge('entity_persister', END)

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
    """Run the full pipeline on a PDF: assemble the markdown document and persist the graph.

    The Mistral OCR API turns each page into reading-ordered markdown plus extracted
    figures (no GPU, no docling); the graph then corrects, parses, heals, builds the typed
    block overlay, and (when Neo4j is configured) persists the ``:Node`` provenance layer and
    the ``:Entity`` overlay plus its procedural layer on top of it. ``pages`` (0-based) optionally limits which pages are
    sent. ``source`` is the book identity used as the graph's Neo4j key (defaults to the PDF's
    filename); ``title``/``author`` are optional book attributes stored on the ``:Source`` node.
    Graph persistence is skipped entirely when Neo4j isn't configured — a DB-less run still
    produces ``document.md`` but persists no nodes or entities. Returns the path of the assembled
    document. Setting ``KMS_TRACE_DIR`` additionally captures every DSPy call's inputs and
    outputs as JSONL (see ``core.tracing``).
    """
    output_dir = Path(output_dir)
    from kms.ingestion import ocr

    source = source or Path(pdf_path).name
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
        written = assemble(
            nodes, result['segments'], output_dir=output_dir, filename=filename
        )
        return written
    finally:
        await (
            close_driver()
        )  # release the Neo4j connection pool (a no-op if never opened)
