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
    corrector -> extractor -> seam_merger (even, odd) -> splitter
              -> {problem, definition, theorem}_finder -> {…}_attributor

Two phases split at the seam merger. Ingestion is per-page: `segments` (already carrying
Mistral's markdown + figures) is the backbone, and the corrector proofreads each page's
transcription against its image before the (purely structural) extractor parses it into
nodes. The seam merger heals nodes split across page breaks and then flattens the healed
backbone into the global ordered `nodes` list (stable ids + seg_index). The splitter then
normalises that stream — it rewrites any node that packs several exercises into one node per
exercise (and tags lead-in nodes `role="instruction"`) — so the finders see atomic exercises
and no longer collapse them into duplicate-membered entities. Three per-type chains then run
in parallel: each finder walks `nodes` to build its entity overlay, and its attributor
enriches those entities with the self-contained AutoMathKG attributes (label, number, title,
field, contents, bodylist; plus proofs for theorems and solutions for problems). After the
graph returns, `run()` concatenates the three overlays into one flat, document-ordered entity
list (global ids) and writes both the entities and the node stream itself — the nodes are
persisted for provenance so an entity's `members` resolve to real source chunks. Cross-entity
attributes (refs/references_tactics) and the instruction distributor (which stamps
`Problem.instruction` from the tagged lead-ins) are later work. Assembly walks `nodes` after
the graph returns, consulting `segments` only
for picture inventories.
"""

from pathlib import Path

from langgraph.graph import StateGraph, START, END

from .assembler import assemble
from .state import State
from .corrector import CorrectorNode
from .extractor import ExtractorNode
from .seam_merger import SeamMergerNode
from .problem_finder import ProblemFinderNode
from .definition_finder import DefinitionFinderNode
from .theorem_finder import TheoremFinderNode
from .problem_attributor import ProblemAttributorNode
from .definition_attributor import DefinitionAttributorNode
from .theorem_attributor import TheoremAttributorNode
from .exercise_splitter import SplitterNode


def build_graph():
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
    splitter = SplitterNode()

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
    g.add_node("splitter", splitter.run)

    # A stage's dispatch is a conditional edge off the previous collect: it either fans
    # out Sends to the worker or short-circuits straight to its own collect.
    g.add_conditional_edges(START, corrector.dispatch, ["corrector_worker", "corrector_collect"])
    g.add_edge("corrector_worker", "corrector_collect")

    g.add_conditional_edges("corrector_collect", extractor.dispatch, ["extractor_worker", "extractor_collect"])
    g.add_edge("extractor_worker", "extractor_collect")

    # Seam healing: even pass then odd pass, so no two concurrent workers touch the
    # same segment (see seam_merger's parity note).
    g.add_conditional_edges("extractor_collect", seam.dispatch_even, ["seam_even_worker", "seam_even_collect"])
    g.add_edge("seam_even_worker", "seam_even_collect")
    g.add_conditional_edges("seam_even_collect", seam.dispatch_odd, ["seam_odd_worker", "seam_odd_collect"])
    g.add_edge("seam_odd_worker", "seam_odd_collect")

    # The splitter runs once after the seam collect, normalising the node stream so each
    # exercise is its own node (and lead-ins are tagged) before any finder walks it.
    g.add_edge("seam_odd_collect", "splitter")

    # Three per-type chains run in parallel off the splitter: each finder does a sequential
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
        g.add_edge("splitter", finder)
        g.add_edge(finder, attributor)
        g.add_edge(attributor, END)

    return g.compile()


async def run(
    pdf_path: str | Path,
    output_dir: str | Path = "output",
    filename: str = "document.md",
    pages: list[int] | None = None,
) -> Path:
    """Run the full pipeline on a PDF and write the assembled markdown + entities.

    The Mistral OCR API turns each page into reading-ordered markdown plus extracted
    figures (no GPU, no docling); the graph then corrects, parses, heals, and builds the
    typed entity overlay. ``pages`` (0-based) optionally limits which pages are sent.
    Returns the path of the assembled document.
    """
    output_dir = Path(output_dir)
    from . import mistral_ocr

    segments = mistral_ocr.extract(pdf_path, output_dir=output_dir, pages=pages)
    graph = build_graph()
    result = await graph.ainvoke({"segments": segments}, {"recursion_limit": 1000})
    nodes = result["nodes"]
    written = assemble(nodes, result["segments"], output_dir=output_dir, filename=filename)
    _write_nodes(nodes, output_dir)
    _write_entities(_flatten_entities(result, nodes), output_dir)
    return written


def _flatten_entities(result: dict, nodes: list) -> list:
    """Concatenate the three per-type finder overlays into one flat, document-ordered
    entity list and assign each a global id. The overlays are independent and may
    reference the same node more than once (members are node-id pointers) — they are
    concatenated, not merged. Because the splitter made exercise nodes atomic upstream,
    the problem finder already emits one entity per exercise with distinct members, so no
    coarse-vs-fine reconciliation is needed here."""
    entities = (
        result.get("problem_entities", [])
        + result.get("definition_entities", [])
        + result.get("theorem_entities", [])
    )
    order = {node.id: i for i, node in enumerate(nodes)}
    big = len(order)
    entities.sort(key=lambda e: order.get(e.members[0], big) if e.members else big)
    for i, entity in enumerate(entities):
        entity.id = i
    return entities


def _write_nodes(nodes: list, output_dir: Path) -> Path:
    """Persist the flat node stream as JSON for provenance — an entity's `members` are node
    ids into this file, so the later graph phase can link an entity to its source chunks."""
    import json

    payload = [
        {
            "id": node.id,
            "type": node.type.value if node.type else None,
            "content": node.content,
            "seg_index": node.seg_index,
            **({"role": node.role} if node.role else {}),
        }
        for node in nodes
    ]
    path = Path(output_dir) / "nodes.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_entities(entities: list, output_dir: Path) -> Path:
    """Persist the flat entity overlay as JSON beside the assembled document — the artifact
    the later graph phase (edges, fusion, completion) consumes. Each entity carries `{id,
    type, members}` (members are node ids into `nodes.json`) plus whatever self-contained
    attributes its attributor filled in; unset attributes are omitted, so a bare (un-attributed)
    entity serializes to just `{id, type, members}`. This is a debug/inspection dump of what
    the entity holds, not a designed schema — the graph tier will own persistence."""
    import json

    payload = [_entity_payload(e) for e in entities]
    path = Path(output_dir) / "entities.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _entity_payload(e) -> dict:
    """One entity as a JSON-ready dict: id/type/members always, then any attribute that is
    set. Structured attributes (bodylist/proofs/solutions) are unpacked by hand rather than
    via pydantic `.model_dump()` so this stays importable under the test stubs."""
    d = {"id": e.id, "type": e.type.value, "members": e.members}
    for key in ("label", "number", "title", "field", "instruction"):
        value = getattr(e, key)
        if value is not None:
            d[key] = value
    if e.contents:
        d["contents"] = e.contents
    if e.bodylist:
        d["bodylist"] = [_seg(s) for s in e.bodylist]
    if e.proofs:
        d["proofs"] = [{"contents": p.contents, "bodylist": [_seg(s) for s in p.bodylist]} for p in e.proofs]
    if e.solutions:
        d["solutions"] = [{"contents": s.contents} for s in e.solutions]
    return d


def _seg(segment) -> dict:
    """A bodylist segment as a plain dict (no pydantic dependency)."""
    return {"description": segment.description, "action": segment.action}


if __name__ == "__main__":
    import asyncio
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    written = asyncio.run(run(pdf, output_dir=out_dir))
    print(f"Wrote assembled document to: {written}")
