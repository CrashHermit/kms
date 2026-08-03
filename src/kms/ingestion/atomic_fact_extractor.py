r"""
Atomic fact extraction — one LangGraph node, one DSPy module.

The first SEMANTIC pass over the provenance stream: it reads the final node
stream and decomposes it into atomic facts — short, self-contained snippets,
each conveying exactly one piece of information — for the downstream concept
and relation passes to consume.

Design commitments:

* FIXED CONTEXT WINDOW. The stream is cut into adjacent windows of whole
  nodes up to a fixed token budget. This is deliberately NOT the PCF
  grow-and-bank look-ahead: PCF must keep boundary spans whole across window
  cuts, but atomic fact decomposition is per-window output (ATOM-style fixed
  chunking), so no growing or rewinding is needed. A fact whose content
  straddles a cut is a known limitation of fixed windows — the tradeoff for
  not paying the grow/rewind cost.

* DOMAIN-AGNOSTIC, NO FACT TAXONOMY. The pass targets facts generally: any
  document, any subject. The prompt does not enumerate fact kinds
  (definition / theorem / claim / …) and carries no genre vocabulary. The
  only criteria are atomicity (one piece of information per fact) and
  durability (something worth knowing, not navigational or transitional
  prose).

* MINIMAL OUTPUT. ``models.AtomicFact`` carries only ``text`` + ``node_ids``
  — no kind, no source. Classification is a downstream pass's job;
  provenance is recoverable by resolving the node ids into the stream.

* STRUCTURAL FILTERING ONLY. ``image`` nodes (placeholder references, no
  text) are dropped from the window. ``header``, ``bibliographic``, and
  ``caption`` nodes ride along as context so the model can place the facts,
  but the prompt instructs the model not to extract facts from them.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, models, state, walker
from kms.core.recorder import Recorder

logger = logging.getLogger(__name__)

# Fixed context window over whole nodes (~4 chars/token). A single node
# larger than the budget still forms a window of its own.
WINDOW_BUDGET = 2000


class DSPyAtomicFact(BaseModel):
    """One atomic fact emitted by the extractor."""

    text: str = Field(
        description=(
            'The fact as a short, self-contained snippet conveying exactly '
            'one piece of information.'
        )
    )
    node_ids: list[int] = Field(
        description=(
            'The ids of every node in the window the fact is drawn from.'
        )
    )


class Signature(dspy.Signature):
    r"""
    You are given a run of nodes from a document, in document order. Each
    node carries its id, its structural type, and its content. Decompose the
    window into ATOMIC FACTS.

    AN ATOMIC FACT is a short, self-contained snippet that conveys exactly
    one piece of information: a claim, a property, a relationship, an event,
    a definition, a result, an instruction to act. It is something worth
    knowing, stated so it reads standalone — the reader should not need the
    surrounding text to understand it.

    This is domain-neutral: the document may be about anything. Do not assume
    a subject or a genre, and do not classify facts into kinds. Just find the
    facts.

    RULES:
    - ONE PIECE OF INFORMATION PER FACT. If a sentence makes two independent
      claims, emit two facts. If a passage asserts several things, emit one
      fact per assertion.
    - SELF-CONTAINED. Keep the wording close to the source, but include
      whatever the fact needs to stand alone (names, qualifiers, conditions).
      A fact drawn from several nodes lists ALL their ids.
    - LATEX FORMAT. Everything that can be in LaTeX format is written in
      LaTeX WITH its delimiters, exactly as in the source: inline math in
      `$...$`, display math in `$$...$$`. This covers mathematical notation,
      chemical formulas, units, and any other technical notation. When a
      fact mentions an equation, a symbol, or any such content, keep it in
      that delimited LaTeX form inside the fact text — never plain text,
      never Unicode (no `x⁴`, `≤`, `α`, bare `H₂O`) when a LaTeX spelling
      exists.
    - DURABLE, NOT TRANSITIONAL. Emit facts — things worth knowing — not
      navigation ("in this section", "as we will see"), not rhetorical
      framing, not formatting.
    - CONTEXT-ONLY NODES. header (a title), bibliographic (a reference
      entry), and caption nodes are context to help you place the facts —
      do NOT extract facts from them.
    - FIND EVERYTHING. A missed fact is a lost fact. When unsure whether
      something is a fact, include it.
    - Return an empty list if the window contains no facts.
    """

    current_nodes: list[walker.WindowNode] = dspy.InputField(
        description=(
            "The window's nodes, in document order, each with its id, type, "
            'and content.'
        )
    )
    facts: list[DSPyAtomicFact] = dspy.OutputField(
        description='Every atomic fact found in the window; empty if none.'
    )


class AtomicFactExtractor(dspy.Module):
    """Extracts atomic facts from one fixed window of nodes.

    Args:
        language_model: The LM to run on.
    """

    def __init__(
        self, language_model: dspy.LM, recorder: Recorder | None = None
    ) -> None:
        super().__init__()
        self.extractor = dspy.ChainOfThought(Signature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self,
        current_nodes: list[walker.WindowNode],
    ) -> list[models.AtomicFact]:
        """Extract the atomic facts from one window.

        Args:
            current_nodes: The window's nodes, in document order.

        Returns:
            The atomic facts found, or an empty list.
        """
        result = await self.extractor.acall(current_nodes=current_nodes)
        if self._recorder:
            self._recorder.record(
                'atomic_fact_extractor',
                {
                    'current_nodes': [
                        node.model_dump() for node in current_nodes
                    ],
                },
                result,
            )
        facts = [
            models.AtomicFact(
                text=fact.text,
                node_ids=list(fact.node_ids or []),
            )
            for fact in (result.facts or [])
        ]
        logger.debug(
            'atomic fact extractor: %d node(s) -> %d fact(s)',
            len(current_nodes),
            len(facts),
        )
        return facts

    def forward(
        self,
        current_nodes: list[walker.WindowNode],
    ) -> list[models.AtomicFact]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(current_nodes=current_nodes))


# ============================================================================
# Entry point
# ============================================================================


async def extract_atomic_facts(
    nodes: list[models.ASTNode],
    module: AtomicFactExtractor,
    max_concurrency: int | None = None,
) -> list[models.AtomicFact]:
    """Extract atomic facts from the whole node stream.

    The stream is cut into adjacent fixed windows of whole nodes; every
    window is decomposed concurrently, and the facts are collected in
    document order. The pass is a pure reader of ``nodes`` — nothing writes
    back to the stream.

    Args:
        nodes: The flat node stream.
        module: The atomic fact extractor.
        max_concurrency: Windows in flight at once. None uses
            ``llm.MAX_CONCURRENT_CALLS``.

    Returns:
        The atomic facts, in document order.
    """
    windows = walker.fixed_windows(nodes, WINDOW_BUDGET)
    if not windows:
        logger.info('atomic fact extractor: no windows')
        return []

    gate = llm.gate(max_concurrency)

    async def _extract_one(
        window: list[models.ASTNode],
    ) -> list[models.AtomicFact]:
        async with gate:
            return await module.aforward(
                [
                    walker.WindowNode(
                        node_id=node.id,
                        type=node.type,
                        content=node.content,
                    )
                    for node in window
                ]
            )

    per_window = await asyncio.gather(
        *(_extract_one(window) for window in windows)
    )
    facts = [fact for window_facts in per_window for fact in window_facts]

    logger.info(
        'atomic fact extractor: %d node(s) in %d window(s) -> %d fact(s)',
        len(nodes),
        len(windows),
        len(facts),
    )
    return facts


# ============================================================================
# LangGraph node
# ============================================================================


class AtomicFactNode:
    """Extracts atomic facts from the node stream.

    Runs after the equation/variable node and before the ingestion
    persister. Reads only the final ``nodes`` stream; writes the
    ``atomic_facts`` channel.

    Args:
        module: The atomic fact extractor.
    """

    def __init__(self, module: AtomicFactExtractor) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Extract atomic facts from the final node stream.

        Args:
            state: The pipeline state, holding the node stream.

        Returns:
            The ``atomic_facts`` channel.
        """
        nodes = state.get('nodes', [])
        facts = await extract_atomic_facts(nodes, module=self.module)
        return {'atomic_facts': facts}
