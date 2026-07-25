"""
LangGraph wiring for the document-processing pipeline.

Builds the ordered graph that turns a PDF (via the Mistral OCR front-end) into a
finished AST, then assembles it to a single markdown document plus the entity overlay.
The ingestion stages (corrector, extractor, seam merger) are map-reduce: a conditional
edge fans out one Send per unit of work to the stage's worker, the workers append to a
per-stage reducer channel, and the collect step drains that channel back into the
ordered backbone before the next stage runs. The entity finders are plain sequential
nodes (their cursor-walk cannot be sharded).

Stage order:
    corrector -> extractor -> seam_merger (even, odd) -> splitter -> instruction_finder
              -> node_persister -> <entity layer> -> collector -> conceptualizer
              -> dependency_finder -> entity_persister

Two phases split at the seam merger. Ingestion is per-page: `segments` (already carrying
Mistral's markdown + figures) is the backbone, and the corrector proofreads each page's
transcription against its image before the (purely structural) extractor parses it into
nodes. The seam merger heals nodes split across page breaks and then flattens the healed
backbone into the global ordered `nodes` list (stable ids + segment_index). The splitter then
normalises that stream — it rewrites any node that packs several exercises into one node per
exercise (embedded lead-ins broken out onto their own nodes too) — so the finders see atomic
exercises and no longer collapse them into duplicate-membered entities. The instruction finder
then tags every lead-in node `role="instruction"` over that atomic stream, one uniform pass.
The node persister then writes the finalized stream to Neo4j as the graph's provenance layer (a
`:Source` root with its `:Node` chain); it runs after the splitter so the persisted ids match the
entity `members`, and is a no-op when Neo4j isn't configured.

**Two entity layers, one selected per run** (see `build_graph`), because the generalization is
mid-strangler-fig: the validated math path is three per-type chains, and the general path is one
block-finder chain, and the second does not replace the first until it is measured at parity on real
math books (`docs/GENERALIZATION.md`, build sequence steps 4-5). Whichever runs, everything
downstream is identical — that is the point of the collector seam.

Downstream of the entity layer the stages are shared. The collector flattens whichever overlays ran
into one document-ordered, globally-id'd `entities` list. The conceptualizer tags every entity and
every procedure step with induced concepts (which replaced AutoMathKG's fixed `field`). The
dependency finder rolls those concepts up, along the reference graph, into concept-level
`:DEPENDS_ON` prerequisites. The entity persister, the terminal stage, upserts the overlay as the
graph's `:Entity` layer (rooted under the `:Source`, linked back to their member `:Node` chunks) and
then every layer above it: procedural (`:Procedure` / `:Event`), concept (`:Concept` +
`:INSTANCE_OF`), prerequisite (`:DEPENDS_ON`), reference (`:REFERENCES` edges onto `:Entity:Canonical`
targets, so citations from any entity converge on one target), step-level `:USES`, and `:REALIZES`.
It is a no-op when Neo4j isn't configured. After the graph returns, `run()` only assembles the
markdown document: assembly walks `nodes`, consulting `segments` only for picture inventories.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from kms.core import state
from kms.entity.attributors.definition import DefinitionAttributorNode
from kms.entity.attributors.problem import ProblemAttributorNode
from kms.entity.attributors.theorem import TheoremAttributorNode
from kms.entity.attributors.universal import UniversalAttributorNode
from kms.entity.collector import CollectorNode
from kms.entity.conceptualizer import ConceptualizerNode
from kms.entity.dependency_finder import DependencyFinderNode
from kms.entity.finders.block import BlockFinderNode
from kms.entity.finders.definition import DefinitionFinderNode
from kms.entity.finders.problem import ProblemFinderNode
from kms.entity.finders.theorem import TheoremFinderNode
from kms.entity.instruction_distributor import InstructionDistributorNode
from kms.entity.instruction_finder import InstructionFinderNode
from kms.entity.procedure_finder import ProcedureFinderNode
from kms.entity.referencers.open import ReferencerNode
from kms.entity.splitter import SplitterNode
from kms.graph.db import close_driver
from kms.graph.persister import EntityPersisterNode, NodePersisterNode
from kms.ingestion.corrector import CorrectorNode
from kms.ingestion.extractor import ExtractorNode
from kms.ingestion.seam_merger import SeamMergerNode
from kms.output.assembler import assemble

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

# Which entity layer to run. `per-type` is the validated math path (three parallel chains, one per
# AutoMathKG type); `block` is the general path (one type-agnostic block finder, an induced open
# type, and a procedure finder). The default is the validated one and stays that way until the
# general path is measured at parity — deleting validated extraction before its replacement is
# proven is the one risk greenfield does not remove (`docs/GENERALIZATION.md`, step 5).
ENTITY_LAYERS = ('per-type', 'block')
ENTITY_LAYER_ENV = 'KMS_ENTITY_LAYER'


def entity_layer() -> str:
    """The entity layer this run uses: ``KMS_ENTITY_LAYER`` when it names a known one, else the
    validated ``per-type`` default. An unknown value falls back rather than raising — a typo in an
    env var should not decide that a book gets no entities."""
    requested = os.environ.get(ENTITY_LAYER_ENV, '').strip().lower()
    return requested if requested in ENTITY_LAYERS else ENTITY_LAYERS[0]


def _wire_per_type_layer(graph: StateGraph) -> list[str]:
    """Wire the three per-type chains between the node persister and the collector.

    Each chain is finder -> attributor -> referencer, running in parallel with the others: the finder
    does a sequential cursor-walk (not shardable) to build its overlay, the attributor enriches those
    entities with the self-contained AutoMathKG attributes (label, number, title, contents, bodylist;
    plus a proof procedure for theorems and a solution procedure for problems), and the referencer
    extracts the cross-entity `refs`. Each chain writes only its own entity channel; overlap between
    overlays is fine (members are node-id pointers). The problem chain has one further stage, the
    instruction distributor, which stamps `Problem.instruction` from the instruction finder's tagged
    lead-in nodes and must run after the attributor because it matches on each problem's `number`.
    Returns the terminal node names to fan into the collector."""
    graph.add_node('problem_finder', ProblemFinderNode().run)
    graph.add_node('definition_finder', DefinitionFinderNode().run)
    graph.add_node('theorem_finder', TheoremFinderNode().run)
    graph.add_node('problem_attributor', ProblemAttributorNode().run)
    graph.add_node('definition_attributor', DefinitionAttributorNode().run)
    graph.add_node('theorem_attributor', TheoremAttributorNode().run)
    graph.add_node('problem_referencer', ReferencerNode('problem_entities').run)
    graph.add_node(
        'definition_referencer', ReferencerNode('definition_entities').run
    )
    graph.add_node('theorem_referencer', ReferencerNode('theorem_entities').run)
    graph.add_node(
        'instruction_distributor',
        InstructionDistributorNode('problem_entities').run,
    )

    for kind in ('problem', 'definition', 'theorem'):
        graph.add_edge('node_persister', f'{kind}_finder')
        graph.add_edge(f'{kind}_finder', f'{kind}_attributor')
        graph.add_edge(f'{kind}_attributor', f'{kind}_referencer')
    graph.add_edge('problem_referencer', 'instruction_distributor')
    return [
        'instruction_distributor',
        'definition_referencer',
        'theorem_referencer',
    ]


def _wire_block_layer(graph: StateGraph) -> list[str]:
    """Wire the single block-finder chain between the node persister and the collector.

    One sequential chain: the block finder walks the stream and emits untyped spans, the universal
    attributor induces each block's open `type` alongside its label/number/title/contents, the
    instruction distributor stamps the shared directive of a grouped exercise run, the procedure
    finder extracts (or creates) the block's worked derivation — it runs after the distributor
    because a created solution needs the instruction to know what is being asked — and the referencer
    extracts the cross-entity `refs` from statement and derivation alike. Returns the terminal node
    name to fan into the collector."""
    graph.add_node('block_finder', BlockFinderNode().run)
    graph.add_node('universal_attributor', UniversalAttributorNode().run)
    graph.add_node(
        'instruction_distributor',
        InstructionDistributorNode('block_entities').run,
    )
    graph.add_node('procedure_finder', ProcedureFinderNode().run)
    graph.add_node('block_referencer', ReferencerNode('block_entities').run)

    graph.add_edge('node_persister', 'block_finder')
    graph.add_edge('block_finder', 'universal_attributor')
    graph.add_edge('universal_attributor', 'instruction_distributor')
    graph.add_edge('instruction_distributor', 'procedure_finder')
    graph.add_edge('procedure_finder', 'block_referencer')
    return ['block_referencer']


def build_graph(layer: str | None = None) -> 'CompiledStateGraph':
    """Assemble and compile the LangGraph pipeline over the shared state.State.

    A single straight path: the correction pass proofreads each Mistral-transcribed page against its
    image, the extractor parses the corrected markdown into structural nodes, the seam merger heals
    page-split nodes and flattens to the global stream, the selected entity layer builds the overlay,
    and the semantic stages (conceptualizer, dependency finder) enrich it before it is persisted.

    ``layer`` selects the entity layer (``per-type`` or ``block``); it defaults to
    ``entity_layer()``, which reads ``KMS_ENTITY_LAYER`` and falls back to the validated per-type
    path. Everything downstream of the collector is identical either way, which is what makes the two
    comparable on the same book.
    """
    corrector = CorrectorNode()
    extractor = ExtractorNode()
    seam = SeamMergerNode()

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
    graph.add_node('splitter', SplitterNode().run)
    graph.add_node('instruction_finder', InstructionFinderNode().run)
    graph.add_node('node_persister', NodePersisterNode().run)
    graph.add_node('collector', CollectorNode().run)
    graph.add_node('conceptualizer', ConceptualizerNode().run)
    graph.add_node('dependency_finder', DependencyFinderNode().run)
    graph.add_node('entity_persister', EntityPersisterNode().run)

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
    # exercise (and each embedded lead-in) is its own node before any finder walks it.
    graph.add_edge('seam_odd_collect', 'splitter')

    # The instruction finder then tags every lead-in node `role="instruction"` over the
    # now-atomic stream — one uniform pass, standalone and embedded lead-ins alike.
    graph.add_edge('splitter', 'instruction_finder')

    # Persist the finalized node stream as the graph's provenance layer BEFORE any finder runs.
    # It sits after the splitter (which re-ids the stream) and the instruction finder so the
    # persisted node ids and role tags match the entity overlay. A no-op when Neo4j isn't
    # configured.
    graph.add_edge('instruction_finder', 'node_persister')

    # The entity layer: either the three per-type chains or the one block chain, both bounded by
    # the node persister upstream and the collector downstream.
    selected = layer if layer in ENTITY_LAYERS else entity_layer()
    terminals = (
        _wire_block_layer(graph)
        if selected == 'block'
        else _wire_per_type_layer(graph)
    )

    # The collector is the fan-in: it runs once every chain completes and flattens their overlays
    # into the one document-ordered, globally-id'd `entities` list the semantic stages work over.
    for terminal in terminals:
        graph.add_edge(terminal, 'collector')
    graph.add_edge('collector', 'conceptualizer')
    graph.add_edge('conceptualizer', 'dependency_finder')
    graph.add_edge('dependency_finder', 'entity_persister')
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
    layer: str | None = None,
) -> Path:
    """Run the full pipeline on a PDF: assemble the markdown document and persist the graph.

    The Mistral OCR API turns each page into reading-ordered markdown plus extracted
    figures (no GPU, no docling); the graph then corrects, parses, heals, builds the
    entity overlay, conceptualizes it, and (when Neo4j is configured) persists the ``:Node``
    provenance layer and every semantic layer above it. ``pages`` (0-based) optionally limits which
    pages are sent. ``source`` is the book identity used as the graph's Neo4j key (defaults to the
    PDF's filename); ``title``/``author`` are optional book attributes stored on the ``:Source``
    node; ``layer`` selects the entity layer (see ``build_graph``). Graph persistence is skipped
    entirely when Neo4j isn't configured — a DB-less run still produces ``document.md`` but persists
    no nodes or entities. Returns the path of the assembled document.
    """
    output_dir = Path(output_dir)
    from kms.ingestion import ocr

    source = source or Path(pdf_path).name
    metadata = {'title': title, 'author': author}
    segments = ocr.extract(pdf_path, output_dir=output_dir, pages=pages)
    graph = build_graph(layer)
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
