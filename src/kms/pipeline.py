"""
LangGraph wiring for the document-processing pipeline.

Builds the ordered graph that turns a PDF (via the Mistral OCR front-end) into a
finished AST, then assembles it to a single markdown document plus the entity overlay.
The ingestion stages (corrector, extractor, seam merger) are map-reduce: a conditional
edge fans out one Send per unit of work to the stage's worker, the workers append to a
per-stage reducer channel, and the collect step drains that channel back into the
ordered backbone before the next stage runs. The three per-type finders are plain
sequential nodes (their cursor-walk cannot be sharded) that fan out in parallel off the
seam-merge collect.

Stage order:
    corrector -> extractor -> seam_merger (even, odd) -> splitter -> instruction_finder
              -> node_persister -> {problem, definition, theorem}_finder -> {…}_attributor
              -> {…}_referencer -> (problem chain only) instruction_distributor -> entity_persister

Two phases split at the seam merger. Ingestion is per-page: `segments` (already carrying
Mistral's markdown + figures) is the backbone, and the corrector proofreads each page's
transcription against its image before the (purely structural) extractor parses it into
nodes. The seam merger heals nodes split across page breaks and then flattens the healed
backbone into the global ordered `nodes` list (stable ids + seg_index). The splitter then
normalises that stream — it rewrites any node that packs several exercises into one node per
exercise (embedded lead-ins broken out onto their own nodes too) — so the finders see atomic
exercises and no longer collapse them into duplicate-membered entities. The instruction finder
then tags every lead-in node `role="instruction"` over that atomic stream, one uniform pass.
The node persister then writes the finalized stream to Neo4j as the graph's provenance layer (a
`:Source` root with its `:Node` chain); it runs after the splitter so the persisted ids match the
entity `members`, and is a no-op when Neo4j isn't configured. Three per-type chains then run
in parallel: each finder walks `nodes` to build its entity overlay, its attributor enriches those
entities with the self-contained AutoMathKG attributes (label, number, title, field, contents,
bodylist; plus proofs for theorems and solutions for problems), and its referencer then extracts the
one cross-entity attribute — `refs`, the definitions/theorems the entity cites, each with a tactic
label. On the problem chain one further stage, the instruction distributor, then stamps
`Problem.instruction` from the instruction finder's tagged lead-in nodes (the shared directive of a
grouped-exercise run). The three chains fan into the entity persister, the terminal stage: it flattens
the overlays into one document-ordered, globally-id'd list and upserts them as the graph's `:Entity`
layer (rooted under the `:Source`, linked back to their member `:Node` chunks), then the procedural
layer (`:Procedure` / `:Event` reified from proofs and solutions), the concept layer (`:Concept` +
`:INSTANCE_OF`, from each entity's field), the reference layer — `:REFERENCES` edges onto
`:Entity:Canonical` targets, so citations from any entity converge on one target — and the step-level
`:USES` edges (a proof step to the canonical it names). A no-op when Neo4j isn't configured.
After the graph returns, `run()` only assembles the markdown document: assembly walks `nodes`,
consulting `segments` only for picture inventories.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from kms.core.state import State
from kms.entity.attributors.definition import DefinitionAttributorNode
from kms.entity.attributors.problem import ProblemAttributorNode
from kms.entity.attributors.theorem import TheoremAttributorNode
from kms.entity.finders.definition import DefinitionFinderNode
from kms.entity.finders.problem import ProblemFinderNode
from kms.entity.finders.theorem import TheoremFinderNode
from kms.entity.instruction_distributor import InstructionDistributorNode
from kms.entity.instruction_finder import InstructionFinderNode
from kms.entity.referencers.definition import DefinitionReferencerNode
from kms.entity.referencers.problem import ProblemReferencerNode
from kms.entity.referencers.theorem import TheoremReferencerNode
from kms.entity.splitter import SplitterNode
from kms.graph.db import close_driver
from kms.graph.persister import EntityPersisterNode, NodePersisterNode
from kms.ingestion.corrector import CorrectorNode
from kms.ingestion.extractor import ExtractorNode
from kms.ingestion.seam_merger import SeamMergerNode
from kms.output.assembler import assemble

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_graph() -> "CompiledStateGraph":
    """Assemble and compile the LangGraph pipeline over the shared State.

    A single straight path: the correction pass proofreads each Mistral-transcribed
    page against its image, the extractor parses the corrected markdown into structural
    nodes, the seam merger heals page-split nodes and flattens to the global stream, and
    the three per-type finders (problem / definition / theorem) each build their overlay.
    """
    corrector = CorrectorNode()
    extractor = ExtractorNode()
    seam = SeamMergerNode()
    problem_finder = ProblemFinderNode()
    definition_finder = DefinitionFinderNode()
    theorem_finder = TheoremFinderNode()
    problem_attributor = ProblemAttributorNode()
    definition_attributor = DefinitionAttributorNode()
    theorem_attributor = TheoremAttributorNode()
    problem_referencer = ProblemReferencerNode()
    definition_referencer = DefinitionReferencerNode()
    theorem_referencer = TheoremReferencerNode()
    splitter = SplitterNode()
    instruction_finder = InstructionFinderNode()
    node_persister = NodePersisterNode()
    entity_persister = EntityPersisterNode()
    instruction_distributor = InstructionDistributorNode()

    g = StateGraph(State)

    # Each stage registers its worker (Send target) and collect (drain) nodes.
    g.add_node("corrector_worker", corrector.worker)
    g.add_node("corrector_collect", corrector.collect)
    g.add_node("extractor_worker", extractor.worker)
    g.add_node("extractor_collect", extractor.collect)
    g.add_node("seam_even_worker", seam.even_worker)
    g.add_node("seam_even_collect", seam.even_collect)
    g.add_node("seam_odd_worker", seam.odd_worker)
    g.add_node("seam_odd_collect", seam.odd_collect)
    g.add_node("problem_finder", problem_finder.run)
    g.add_node("definition_finder", definition_finder.run)
    g.add_node("theorem_finder", theorem_finder.run)
    g.add_node("problem_attributor", problem_attributor.run)
    g.add_node("definition_attributor", definition_attributor.run)
    g.add_node("theorem_attributor", theorem_attributor.run)
    g.add_node("problem_referencer", problem_referencer.run)
    g.add_node("definition_referencer", definition_referencer.run)
    g.add_node("theorem_referencer", theorem_referencer.run)
    g.add_node("splitter", splitter.run)
    g.add_node("instruction_finder", instruction_finder.run)
    g.add_node("node_persister", node_persister.run)
    g.add_node("entity_persister", entity_persister.run)
    g.add_node("instruction_distributor", instruction_distributor.run)

    # A stage's dispatch is a conditional edge off the previous collect: it either fans
    # out Sends to the worker or short-circuits straight to its own collect.
    g.add_conditional_edges(START, corrector.dispatch, ["corrector_worker", "corrector_collect"])
    g.add_edge("corrector_worker", "corrector_collect")

    g.add_conditional_edges(
        "corrector_collect", extractor.dispatch, ["extractor_worker", "extractor_collect"]
    )
    g.add_edge("extractor_worker", "extractor_collect")

    # Seam healing: even pass then odd pass, so no two concurrent workers touch the
    # same segment (see seam_merger's parity note).
    g.add_conditional_edges(
        "extractor_collect", seam.dispatch_even, ["seam_even_worker", "seam_even_collect"]
    )
    g.add_edge("seam_even_worker", "seam_even_collect")
    g.add_conditional_edges(
        "seam_even_collect", seam.dispatch_odd, ["seam_odd_worker", "seam_odd_collect"]
    )
    g.add_edge("seam_odd_worker", "seam_odd_collect")

    # The splitter runs once after the seam collect, normalising the node stream so each
    # exercise (and each embedded lead-in) is its own node before any finder walks it.
    g.add_edge("seam_odd_collect", "splitter")

    # The instruction finder then tags every lead-in node `role="instruction"` over the
    # now-atomic stream — one uniform pass, standalone and embedded lead-ins alike.
    g.add_edge("splitter", "instruction_finder")

    # Persist the finalized node stream as the graph's provenance layer BEFORE any finder runs.
    # It sits after the splitter (which re-ids the stream) and the instruction finder so the
    # persisted node ids and role tags match the entity overlay. A no-op when Neo4j isn't
    # configured.
    g.add_edge("instruction_finder", "node_persister")

    # Three per-type chains run in parallel off the persister: each finder does a sequential
    # cursor-walk (not shardable) to build its overlay, then its attributor enriches those
    # entities with the self-contained AutoMathKG attributes. Each chain writes only its own
    # entity channel; overlap between overlays is fine (members are node-id pointers).
    # `run()` concatenates the three channels after the graph returns.
    chains = [
        ("problem_finder", "problem_attributor"),
        ("definition_finder", "definition_attributor"),
        ("theorem_finder", "theorem_attributor"),
    ]
    for finder, attributor in chains:
        g.add_edge("node_persister", finder)
        g.add_edge(finder, attributor)

    # Each attributor is followed by its per-type referencer, which extracts the entity's
    # cross-entity `refs` (needs `contents`, so it runs after the attributor). The definition and
    # theorem chains then fan into the entity persister; the problem chain has one more step first,
    # the instruction distributor, which stamps `Problem.instruction` from the instruction finder's
    # tagged lead-in nodes and must run after the attributor because it matches on each problem's
    # `number` (which the attributor fills).
    g.add_edge("definition_attributor", "definition_referencer")
    g.add_edge("definition_referencer", "entity_persister")
    g.add_edge("theorem_attributor", "theorem_referencer")
    g.add_edge("theorem_referencer", "entity_persister")
    g.add_edge("problem_attributor", "problem_referencer")
    g.add_edge("problem_referencer", "instruction_distributor")
    g.add_edge("instruction_distributor", "entity_persister")

    # The entity persister is the fan-in: it runs once all three chains complete, flattens the
    # overlays into one document-ordered list, and upserts them as the graph's `:Entity` layer
    # (a no-op when Neo4j isn't configured). It is the pipeline's terminal stage.
    g.add_edge("entity_persister", END)

    return g.compile()


async def run(
    pdf_path: str | Path,
    output_dir: str | Path = "output",
    filename: str = "document.md",
    pages: list[int] | None = None,
    source: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> Path:
    """Run the full pipeline on a PDF: assemble the markdown document and persist the graph.

    The Mistral OCR API turns each page into reading-ordered markdown plus extracted
    figures (no GPU, no docling); the graph then corrects, parses, heals, builds the typed
    entity overlay, and (when Neo4j is configured) persists the ``:Node`` provenance layer and
    the ``:Entity`` overlay on top of it. ``pages`` (0-based) optionally limits which pages are
    sent. ``source`` is the book identity used as the graph's Neo4j key (defaults to the PDF's
    filename); ``title``/``author`` are optional book attributes stored on the ``:Source`` node.
    Graph persistence is skipped entirely when Neo4j isn't configured — a DB-less run still
    produces ``document.md`` but persists no nodes or entities. Returns the path of the assembled
    document.
    """
    output_dir = Path(output_dir)
    from kms.ingestion import ocr

    source = source or Path(pdf_path).name
    metadata = {"title": title, "author": author}
    segments = ocr.extract(pdf_path, output_dir=output_dir, pages=pages)
    graph = build_graph()
    try:
        result = await graph.ainvoke(
            {"segments": segments, "source": source, "source_metadata": metadata},
            {"recursion_limit": 1000},
        )
        nodes = result["nodes"]
        written = assemble(nodes, result["segments"], output_dir=output_dir, filename=filename)
        return written
    finally:
        await close_driver()  # release the Neo4j connection pool (a no-op if never opened)
