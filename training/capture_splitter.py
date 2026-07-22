"""
Capture splitter input/output traces from fixture PDFs, cheaply.

Runs a TRUNCATED pipeline — corrector -> extractor -> seam merger -> splitter, then
stops — so we pay only the front-end (which is cache-warm) plus the splitter itself,
not the finders/attributors/distributor. With ``KMS_TRACE_DIR`` set, the splitter
appends every window's (inputs, outputs) to ``<KMS_TRACE_DIR>/splitter.jsonl``; that
file is the raw material the compile step curates into a trainset.

Run:
  KMS_TRACE_DIR=out/traces PYTHONPATH=src uv run --extra mistral \
      python -m training.capture_splitter tests/fixtures/books/ea2e_ch1_review.pdf ...
(no PDF args => every committed fixture under tests/fixtures/books/)
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from module import mistral_ocr
from module.state import State
from module.corrector import CorrectorNode
from module.extractor import ExtractorNode
from module.seam_merger import SeamMergerNode
from module.exercise_splitter import SplitterNode

FIXTURES = Path("tests/fixtures/books")


def _truncated_graph():
    """corrector -> extractor -> seam(even, odd) -> splitter -> END."""
    corrector, extractor, seam, splitter = CorrectorNode(), ExtractorNode(), SeamMergerNode(), SplitterNode()
    g = StateGraph(State)
    g.add_node("corrector_worker", corrector.worker)
    g.add_node("corrector_collect", corrector.collect)
    g.add_node("extractor_worker", extractor.worker)
    g.add_node("extractor_collect", extractor.collect)
    g.add_node("seam_even_worker", seam.even_worker)
    g.add_node("seam_even_collect", seam.even_collect)
    g.add_node("seam_odd_worker", seam.odd_worker)
    g.add_node("seam_odd_collect", seam.odd_collect)
    g.add_node("splitter", splitter.run)
    g.add_conditional_edges(START, corrector.dispatch, ["corrector_worker", "corrector_collect"])
    g.add_edge("corrector_worker", "corrector_collect")
    g.add_conditional_edges("corrector_collect", extractor.dispatch, ["extractor_worker", "extractor_collect"])
    g.add_edge("extractor_worker", "extractor_collect")
    g.add_conditional_edges("extractor_collect", seam.dispatch_even, ["seam_even_worker", "seam_even_collect"])
    g.add_edge("seam_even_worker", "seam_even_collect")
    g.add_conditional_edges("seam_even_collect", seam.dispatch_odd, ["seam_odd_worker", "seam_odd_collect"])
    g.add_edge("seam_odd_worker", "seam_odd_collect")
    g.add_edge("seam_odd_collect", "splitter")
    g.add_edge("splitter", END)
    return g.compile()


async def _capture_one(pdf: Path, graph, workdir: Path) -> None:
    segments = mistral_ocr.extract(str(pdf), output_dir=str(workdir / pdf.stem))
    result = await graph.ainvoke({"segments": segments}, {"recursion_limit": 1000})
    print(f"  {pdf.name}: {len(result.get('nodes', []))} nodes after splitter")


def main(pdfs: list[str]) -> None:
    targets = [Path(p) for p in pdfs] if pdfs else sorted(FIXTURES.glob("*.pdf"))
    if not targets:
        raise SystemExit("no fixture PDFs found")
    graph = _truncated_graph()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for pdf in targets:
            print(f"capturing {pdf} ...")
            asyncio.run(_capture_one(pdf, graph, work))


if __name__ == "__main__":
    main(sys.argv[1:])
