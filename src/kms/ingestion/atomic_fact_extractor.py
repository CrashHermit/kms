r"""
Atomic fact extraction — one LangGraph node, one DSPy module.

The first SEMANTIC pass over the provenance stream: it reads the final node
stream and decomposes it into atomic facts — short, self-contained snippets,
each conveying exactly one piece of information — for the downstream concept
and relation passes to consume.

Design commitments:

* THREE-PART WINDOW. Each window has three independently-tunable parts:
  backward context (BACKWARD_CONTEXT_BUDGET), the central extraction
  window (WINDOW_BUDGET, ~400 tokens — ATOM's empirically optimal chunk
  for decomposition), and forward context (FORWARD_CONTEXT_BUDGET). The
  central window is what facts are extracted from and attributed to; the
  context around it exists so the model can place the window and resolve
  referents — it is placement-only and never extracted from. This is
  deliberately NOT the PCF grow-and-bank look-ahead: the budgets are
  fixed, so no growing or rewinding is needed. A fact whose content
  straddles a cut is a known limitation of fixed windows — the tradeoff
  for not paying the grow/rewind cost.

* DOMAIN-AGNOSTIC, NO FACT TAXONOMY. The pass targets facts generally: any
  document, any subject. The prompt does not enumerate fact kinds
  (definition / theorem / claim / …) and carries no genre vocabulary. The
  only criteria are atomicity and durability.

* OPERATIONAL ATOMICITY. "One piece of information" is defined by an
  apply-able test, not left to taste: a fact conveys exactly one unit — one
  assertion, one instruction, or one question — stated as a complete
  standalone sentence. The prompt's SPLIT TEST — can the fact be broken at a
  conjunction or comma into two pieces still true of (or still posed by)
  the source? — is the judgment call, and compound-to-split examples
  demonstrate atomicity rather than assert it. Referent-less fragments
  ("since $a \neq 0$") are rejected: a fact must read standalone.

* NO CONTENT SPECIAL CASES. Nothing in the stream is a special case: no
  exercise/lead-in/statement/procedure taxonomy, no genre vocabulary. The
  same atomicity and durability criteria apply to every node; whether a
  passage turns out to be practice material or not is a decision for later
  passes, not for this one.

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

# Central extraction window over whole nodes, in TOKENS. ATOM's
# empirical sweet spot is <400 tokens per chunk: smaller chunks hold
# exhaustivity and stability as the context grows. A single node larger
# than the budget still forms a window of its own.
WINDOW_BUDGET = 400

# Surrounding context budgets (tokens), tunable independently of the
# central window and of each other. The context is placement-only: the
# model sees it to place facts and resolve referents, but must not
# extract facts from it.
BACKWARD_CONTEXT_BUDGET = 200
FORWARD_CONTEXT_BUDGET = 200


class DSPyAtomicFact(BaseModel):
    """One atomic fact emitted by the extractor."""

    text: str = Field(
        description=(
            'The fact as a short, self-contained standalone sentence '
            'conveying exactly one unit of information: one assertion, one '
            'instruction, or one question.'
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

    AN ATOMIC FACT is the smallest piece of the text that is still worth
    knowing on its own: a claim, a property, a relationship, an event, a
    definition, a result, an instruction to act, or a question posed for
    the reader. It conveys exactly ONE such unit, stated as a complete
    standalone sentence. This is domain-neutral: the document may be about
    anything. Do not assume a subject or a genre, and do not classify
    facts into kinds. Just find the facts.

    THE ATOMICITY TEST. A fact is atomic when it conveys exactly one unit
    of information — one assertion, one instruction, or one question —
    with no second independent unit joined on. Apply the SPLIT TEST before
    emitting: if you can break the fact at a conjunction or a comma into
    two pieces that would each still be true of — or each still be posed
    by — the source, it is not atomic — split it into those pieces. When
    in doubt, split.

    EXAMPLES.

    - "The discriminant of $ax^2 + bx + c = 0$ is $b^2 - 4ac$, and its
      roots are given by the quadratic formula" is TWO facts:
        1. "The discriminant of $ax^2 + bx + c = 0$ is $b^2 - 4ac$."
        2. "The roots of $ax^2 + bx + c = 0$ are given by the quadratic
           formula."
    - "Since $a \neq 0$, the equation $ax^2 + bx + c = 0$ is quadratic" is
      ONE fact with its condition carried inside: "When $a \neq 0$, the
      equation $ax^2 + bx + c = 0$ is quadratic." Never emit the bare
      fragment "Since $a \neq 0$".
    - Successive lines of a worked manipulation — "$8a - 3a > 5a + 18$",
      then "$5a > 5a + 18$" — are scratch work, not facts. The durable
      content is the conclusion: "From $8a - 3a > 5a + 18$ it follows that
      $0 > 18$, a contradiction."

    RULES:
    - ONE UNIT PER FACT. One assertion, one instruction, or one question
      per fact. A sentence that makes two independent claims yields two
      facts; a passage that asserts several things yields one fact per
      assertion.
    - STANDALONE, NOT FRAGMENTED. State every fact as a complete sentence
      that names its own subject and carries its own conditions and
      qualifiers — whether it asserts, instructs, or asks. Resolve every
      "it", "this", "the former" into its referent. Never emit a fragment
      ("since $a \neq 0$", "which is continuous", "as above").
    - SELF-CONTAINED IS NOT COPYING. Include what the fact needs to stand
      alone (names, conditions, values) — but a multi-claim source sentence
      yields several SHORTER facts, never one copied sentence.
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
      framing, not formatting, not the scratch lines of a worked
      manipulation.
    - NO DUPLICATES. State each distinct claim once per window. If the
      window restates the same claim — rephrased, repeated, re-derived —
      emit it once.
    - CONTEXT-ONLY NODES. header (a title), bibliographic (a reference
      entry), and caption nodes are context to help you place the facts —
      do NOT extract facts from them.
    - CONTEXT-ONLY SURROUNDING TEXT. context_before and context_after are
      the text immediately around the window, included so you can place
      the facts and resolve referents. They are context only — never
      extract facts from them, and never attribute a fact to them.
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
    context_before: str | None = dspy.InputField(
        default=None,
        description=(
            'Optional text immediately before the window, in document '
            'order. CONTEXT ONLY — use it to place the facts; never '
            'extract facts from it.'
        ),
    )
    context_after: str | None = dspy.InputField(
        default=None,
        description=(
            'Optional text immediately after the window, in document '
            'order. CONTEXT ONLY — use it to place the facts; never '
            'extract facts from it.'
        ),
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
        context_before: str | None = None,
        context_after: str | None = None,
    ) -> list[models.AtomicFact]:
        """Extract the atomic facts from one window.

        Args:
            current_nodes: The window's nodes, in document order.
            context_before: Optional text immediately before the window,
                placement-only, never extracted from.
            context_after: Optional text immediately after the window,
                placement-only, never extracted from.

        Returns:
            The atomic facts found, or an empty list.
        """
        result = await self.extractor.acall(
            current_nodes=current_nodes,
            context_before=context_before or '',
            context_after=context_after or '',
        )
        if self._recorder:
            self._recorder.record(
                'atomic_fact_extractor',
                {
                    'current_nodes': [
                        node.model_dump() for node in current_nodes
                    ],
                    'context_before': context_before,
                    'context_after': context_after,
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
        context_before: str | None = None,
        context_after: str | None = None,
    ) -> list[models.AtomicFact]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(
            self.aforward(
                current_nodes=current_nodes,
                context_before=context_before,
                context_after=context_after,
            )
        )


# ============================================================================
# Entry point
# ============================================================================


async def extract_atomic_facts(
    nodes: list[models.ASTNode],
    module: AtomicFactExtractor,
    max_concurrency: int | None = None,
) -> list[models.AtomicFact]:
    """Extract atomic facts from the whole node stream.

    The stream is cut into adjacent three-part windows — a central
    extraction window of whole nodes plus placement-only backward/forward
    context; every window is decomposed concurrently, and the facts are
    collected in document order. The pass is a pure reader of ``nodes`` —
    nothing writes back to the stream.

    Args:
        nodes: The flat node stream.
        module: The atomic fact extractor.
        max_concurrency: Windows in flight at once. None uses
            ``llm.MAX_CONCURRENT_CALLS``.

    Returns:
        The atomic facts, in document order.
    """
    windows = walker.fixed_windows_with_context(
        nodes,
        WINDOW_BUDGET,
        BACKWARD_CONTEXT_BUDGET,
        FORWARD_CONTEXT_BUDGET,
    )
    if not windows:
        logger.info('atomic fact extractor: no windows')
        return []

    gate = llm.gate(max_concurrency)

    async def _extract_one(
        window: tuple[list[models.ASTNode], str | None, str | None],
    ) -> list[models.AtomicFact]:
        window_nodes, before, after = window
        async with gate:
            return await module.aforward(
                [
                    walker.WindowNode(
                        node_id=node.id,
                        type=node.type,
                        content=node.content,
                    )
                    for node in window_nodes
                ],
                context_before=before,
                context_after=after,
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

    Runs after the hub builder and before the ingestion
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
