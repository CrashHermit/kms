"""
LangGraph wiring for the document-processing pipeline.

Builds the ordered map-reduce graph that turns a PDF (via the Mistral OCR front-end)
into a finished AST, then assembles it to a single markdown document plus the entity
overlay. Every stage follows the same dispatch/worker/collect shape: a conditional edge
fans out one Send per unit of work to the stage's worker, the workers append to a
per-stage reducer channel, and the collect step drains that channel back into the
ordered backbone before the next stage's dispatch runs.

Stage order:
    corrector -> extractor -> seam_merger (even, odd) -> problem_finder

Two phases split at the seam merger. Ingestion is per-page: `segments` (already carrying
Mistral's markdown + figures) is the backbone, and the corrector proofreads each page's
transcription against its image before the (purely structural) extractor parses it into
nodes. The seam merger heals nodes split across page breaks and then flattens the healed
backbone into the global ordered `nodes` list (stable ids + seg_index); the problem
finder then walks `nodes` to build the sparse `{id, type, members}` overlay. Sibling
Definition and Theorem finders, and the per-attribute passes (member roles, number,
instruction, …), come later. Assembly walks `nodes` after the graph returns, consulting
`segments` only for picture inventories.
"""

from pathlib import Path

from langgraph.graph import StateGraph, START, END

from .assembler import assemble
from .state import State
from .corrector import CorrectorNode
from .extractor import ExtractorNode
from .seam_merger import SeamMergerNode
from .problem_finder import ProblemFinderNode


def build_graph():
    """Assemble and compile the LangGraph pipeline over the shared State.

    A single straight path: the correction pass proofreads each Mistral-transcribed
    page against its image, the extractor parses the corrected markdown into structural
    nodes, the seam merger heals page-split nodes and flattens to the global stream, and
    the problem finder builds the Problem overlay.
    """
    corrector = CorrectorNode()
    extractor = ExtractorNode()
    seam = SeamMergerNode()
    finder = ProblemFinderNode()

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
    g.add_node("problem_finder", finder.run)

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

    # Problem finder: a single sequential cursor-walk (not shardable), so a plain node —
    # it builds the sparse `entities` overlay with the Problems it finds (worked examples
    # AND exercises). The Definition and Theorem finders are added alongside it later;
    # per-attribute passes (roles, number, instruction) come after that.
    g.add_edge("seam_odd_collect", "problem_finder")
    g.add_edge("problem_finder", END)

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
    written = assemble(result["nodes"], result["segments"], output_dir=output_dir, filename=filename)
    _write_entities(result.get("entities", []), output_dir)
    return written


def _write_entities(entities, output_dir: Path) -> Path:
    """Persist the sparse entity overlay as JSON beside the assembled document — the
    artifact the later graph phase (edges, fusion, completion) consumes."""
    import json

    path = Path(output_dir) / "entities.json"
    payload = [
        {"id": e.id, "type": e.type.value, "members": e.members}
        for e in entities
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    import asyncio
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    written = asyncio.run(run(pdf, output_dir=out_dir))
    print(f"Wrote assembled document to: {written}")
