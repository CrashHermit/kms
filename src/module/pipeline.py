"""
LangGraph wiring for the document-processing pipeline.

Builds the ordered map-reduce graph that turns a picture_extractor disk tree into
a finished AST, then assembles it to a single markdown document. Every stage
follows the same dispatch/worker/collect shape defined on the node classes: a
conditional edge fans out one Send per unit of work to the stage's worker, the
workers append to a per-stage reducer channel, and the collect step drains that
channel back into the ordered segment backbone before the next stage's dispatch
runs.

Stage order:
    image_filter -> ocr -> extractor -> seam_merger (even, odd)
    -> problem_refiner -> instruction_governor

Two phases split at the seam merger. Ingestion (image_filter -> ocr -> extractor)
is per-page: `segments` is the backbone. image_filter runs before ocr so the
transcriber only emits placeholders for pictures that survived filtering. The seam
merger heals nodes split across page breaks and then flattens the healed backbone
into the global ordered `nodes` list (stable ids + seg_index); every refinement
stage after it (problem_refiner, instruction_governor, and the future entity
grouping) works on `nodes`, not on the per-segment nesting. Assembly walks `nodes`
after the graph returns, consulting `segments` only for picture inventories.
"""

from pathlib import Path

from langgraph.graph import StateGraph, START, END

from .assembler import assemble
from .state import State, load_segments
from .image_filter import ImageFilterNode
from .ocr import OCRNode
from .extractor import ExtractorNode
from .seam_merger import SeamMergerNode
from .problem_refiner import ProblemRefinerNode
from .instruction_governor import InstructionGovernorNode
from .entity_grouper import EntityGrouperNode
from .entity_attributor import EntityAttributorNode


def build_graph(vision_frontend: bool = True):
    """Assemble and compile the LangGraph pipeline over the shared State.

    ``vision_frontend`` controls how ``segments`` get their markdown ``content``:

    - ``True`` (default): the docling front-end. The graph starts with the ingestion
      vision stages (image_filter → ocr), which transcribe each page render to markdown
      before the extractor runs.
    - ``False``: the Mistral OCR front-end. ``segments`` already carry ``content`` +
      ``pictures`` (see ``mistral_ocr``), so those two vision stages are omitted and the
      graph starts at the extractor. Reading order, duplication avoidance, and figure
      placement were handled server-side, so image_filter's noise-picture pruning is
      not applied on this path.
    """
    image_filter = ImageFilterNode()
    ocr = OCRNode()
    extractor = ExtractorNode()
    seam = SeamMergerNode()
    problem = ProblemRefinerNode()
    governor = InstructionGovernorNode()
    entity = EntityGrouperNode()
    attributor = EntityAttributorNode()

    g = StateGraph(State)

    # Each stage registers its worker (Send target) and collect (drain) nodes. The
    # ingestion vision stages are only added when the docling front-end is in use.
    if vision_frontend:
        g.add_node("image_filter_worker", image_filter.worker)
        g.add_node("image_filter_collect", image_filter.collect)
        g.add_node("ocr_worker", ocr.worker)
        g.add_node("ocr_collect", ocr.collect)
    g.add_node("extractor_worker", extractor.worker)
    g.add_node("extractor_collect", extractor.collect)
    g.add_node("seam_even_worker", seam.even_worker)
    g.add_node("seam_even_collect", seam.even_collect)
    g.add_node("seam_odd_worker", seam.odd_worker)
    g.add_node("seam_odd_collect", seam.odd_collect)
    g.add_node("problem_refiner_worker", problem.worker)
    g.add_node("problem_refiner_collect", problem.collect)
    g.add_node("governor_worker", governor.worker)
    g.add_node("governor_collect", governor.collect)
    g.add_node("entity_grouper_worker", entity.worker)
    g.add_node("entity_grouper_collect", entity.collect)
    g.add_node("entity_attributor_worker", attributor.worker)
    g.add_node("entity_attributor_collect", attributor.collect)

    # A stage's dispatch is a conditional edge off the previous collect: it either
    # fans out Sends to the worker or short-circuits straight to its own collect.
    # The docling front-end runs image_filter → ocr before the extractor; the Mistral
    # front-end feeds segments that already have content, so the graph starts at the
    # extractor.
    if vision_frontend:
        g.add_conditional_edges(START, image_filter.dispatch, ["image_filter_worker", "image_filter_collect"])
        g.add_edge("image_filter_worker", "image_filter_collect")

        g.add_conditional_edges("image_filter_collect", ocr.dispatch, ["ocr_worker", "ocr_collect"])
        g.add_edge("ocr_worker", "ocr_collect")

        g.add_conditional_edges("ocr_collect", extractor.dispatch, ["extractor_worker", "extractor_collect"])
    else:
        g.add_conditional_edges(START, extractor.dispatch, ["extractor_worker", "extractor_collect"])
    g.add_edge("extractor_worker", "extractor_collect")

    # Seam healing: even pass then odd pass, so no two concurrent workers touch the
    # same segment (see seam_merger's parity note).
    g.add_conditional_edges("extractor_collect", seam.dispatch_even, ["seam_even_worker", "seam_even_collect"])
    g.add_edge("seam_even_worker", "seam_even_collect")
    g.add_conditional_edges("seam_even_collect", seam.dispatch_odd, ["seam_odd_worker", "seam_odd_collect"])
    g.add_edge("seam_odd_worker", "seam_odd_collect")

    g.add_conditional_edges("seam_odd_collect", problem.dispatch, ["problem_refiner_worker", "problem_refiner_collect"])
    g.add_edge("problem_refiner_worker", "problem_refiner_collect")

    # Instruction governance: annotate each governed problem with its lead (no rewrite).
    g.add_conditional_edges("problem_refiner_collect", governor.dispatch, ["governor_worker", "governor_collect"])
    g.add_edge("governor_worker", "governor_collect")

    # Entity grouping: gather def/thm/example spans across dumb-greedy windows, wrap
    # atomic problems 1:1, reconcile cross-window spans into the sparse `entities` overlay.
    g.add_conditional_edges("governor_collect", entity.dispatch, ["entity_grouper_worker", "entity_grouper_collect"])
    g.add_edge("entity_grouper_worker", "entity_grouper_collect")

    # Entity attribution (Stage 2): label member roles (statement/proof/solution) and
    # lift number/instruction onto the entities.
    g.add_conditional_edges("entity_grouper_collect", attributor.dispatch, ["entity_attributor_worker", "entity_attributor_collect"])
    g.add_edge("entity_attributor_worker", "entity_attributor_collect")

    g.add_edge("entity_attributor_collect", END)

    return g.compile()


async def run(
    pdf_path: str | Path,
    output_dir: str | Path = "output",
    filename: str = "document.md",
    extract_pictures: bool = True,
    frontend: str = "docling",
    pages: list[int] | None = None,
) -> Path:
    """Run the full pipeline on a PDF and write the assembled markdown.

    ``frontend`` selects how pages become markdown-bearing segments:

    - ``"docling"`` (default): render + crop locally, then transcribe each page with
      the vision OCR stage. When ``extract_pictures`` is True the docling
      picture_extractor runs first to build the ``<output_dir>/Segments`` tree; pass
      False to reuse an existing tree (e.g. to re-run just the LLM stages, or to feed
      a tree produced by ``scripts/pdf_to_segments.py`` on a GPU-less box).
    - ``"mistral"``: call the Mistral OCR API, which returns reading-ordered markdown
      plus extracted figures per page — no GPU, no docling. ``extract_pictures`` is
      ignored; ``pages`` (0-based) optionally limits which pages are sent.

    Returns the written path.
    """
    output_dir = Path(output_dir)
    if frontend == "mistral":
        # Lazy import: keeps the module out of the docling path's import graph.
        from . import mistral_ocr

        segments = mistral_ocr.extract(pdf_path, output_dir=output_dir, pages=pages)
        graph = build_graph(vision_frontend=False)
    elif frontend == "docling":
        if extract_pictures:
            # Lazy import: pulls in docling/torch, which the LLM-only path does not need.
            from . import picture_extractor

            picture_extractor.extract(pdf_path, output_dir=output_dir)
        segments = load_segments(output_dir)
        graph = build_graph(vision_frontend=True)
    else:
        raise ValueError(f"unknown frontend {frontend!r} (expected 'docling' or 'mistral')")

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
        {
            "id": e.id,
            "type": e.type.value,
            "number": e.number,
            "instruction": e.instruction,
            "members": [
                {"node_id": m.node_id, "role": m.role.value if m.role else None}
                for m in e.members
            ],
        }
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
