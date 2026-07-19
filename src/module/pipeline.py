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
    -> exercise_refiner -> instruction_refiner -> instruction_distributor

image_filter runs before ocr so the transcriber only emits placeholders for
pictures that survived filtering. Seam healing runs right after extraction (per
the extractor's contract) so the exercise/instruction stages see whole nodes.
Assembly runs after the graph returns.
"""

from pathlib import Path

from langgraph.graph import StateGraph, START, END

from .assembler import assemble
from .state import State, load_segments
from .image_filter import ImageFilterNode
from .ocr import OCRNode
from .extractor import ExtractorNode
from .seam_merger import SeamMergerNode
from .exercise_refiner import ExerciseRefinerNode
from .instruction_refiner import InstructionRefinerNode
from .instruction_distributor import InstructionDistributorNode


def build_graph():
    """Assemble and compile the LangGraph pipeline over the shared State."""
    image_filter = ImageFilterNode()
    ocr = OCRNode()
    extractor = ExtractorNode()
    seam = SeamMergerNode()
    exercise = ExerciseRefinerNode()
    instruction = InstructionRefinerNode()
    distributor = InstructionDistributorNode()

    g = StateGraph(State)

    # Each stage registers its worker (Send target) and collect (drain) nodes.
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
    g.add_node("exercise_refiner_worker", exercise.worker)
    g.add_node("exercise_refiner_collect", exercise.collect)
    g.add_node("instruction_refiner_worker", instruction.worker)
    g.add_node("instruction_refiner_collect", instruction.collect)
    g.add_node("instruction_distributor_worker", distributor.worker)
    g.add_node("instruction_distributor_collect", distributor.collect)

    # A stage's dispatch is a conditional edge off the previous collect: it either
    # fans out Sends to the worker or short-circuits straight to its own collect.
    g.add_conditional_edges(START, image_filter.dispatch, ["image_filter_worker", "image_filter_collect"])
    g.add_edge("image_filter_worker", "image_filter_collect")

    g.add_conditional_edges("image_filter_collect", ocr.dispatch, ["ocr_worker", "ocr_collect"])
    g.add_edge("ocr_worker", "ocr_collect")

    g.add_conditional_edges("ocr_collect", extractor.dispatch, ["extractor_worker", "extractor_collect"])
    g.add_edge("extractor_worker", "extractor_collect")

    # Seam healing: even pass then odd pass, so no two concurrent workers touch the
    # same segment (see seam_merger's parity note).
    g.add_conditional_edges("extractor_collect", seam.dispatch_even, ["seam_even_worker", "seam_even_collect"])
    g.add_edge("seam_even_worker", "seam_even_collect")
    g.add_conditional_edges("seam_even_collect", seam.dispatch_odd, ["seam_odd_worker", "seam_odd_collect"])
    g.add_edge("seam_odd_worker", "seam_odd_collect")

    g.add_conditional_edges("seam_odd_collect", exercise.dispatch, ["exercise_refiner_worker", "exercise_refiner_collect"])
    g.add_edge("exercise_refiner_worker", "exercise_refiner_collect")

    g.add_conditional_edges("exercise_refiner_collect", instruction.dispatch, ["instruction_refiner_worker", "instruction_refiner_collect"])
    g.add_edge("instruction_refiner_worker", "instruction_refiner_collect")

    g.add_conditional_edges("instruction_refiner_collect", distributor.dispatch, ["instruction_distributor_worker", "instruction_distributor_collect"])
    g.add_edge("instruction_distributor_worker", "instruction_distributor_collect")

    g.add_edge("instruction_distributor_collect", END)

    return g.compile()


async def run(
    pdf_path: str | Path,
    output_dir: str | Path = "output",
    filename: str = "document.md",
    extract_pictures: bool = True,
) -> Path:
    """Run the full pipeline on a PDF and write the assembled markdown.

    When ``extract_pictures`` is True (default) the docling picture_extractor runs
    first to build the ``<output_dir>/Segments`` tree. Pass False to reuse an
    existing tree (e.g. to re-run just the LLM stages). Returns the written path.
    """
    output_dir = Path(output_dir)
    if extract_pictures:
        # Lazy import: pulls in docling/torch, which the LLM-only path does not need.
        from . import picture_extractor

        picture_extractor.extract(pdf_path, output_dir=output_dir)

    segments = load_segments(output_dir)
    graph = build_graph()
    result = await graph.ainvoke({"segments": segments}, {"recursion_limit": 1000})
    return assemble(result["segments"], output_dir=output_dir, filename=filename)


if __name__ == "__main__":
    import asyncio
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    written = asyncio.run(run(pdf, output_dir=out_dir))
    print(f"Wrote assembled document to: {written}")
