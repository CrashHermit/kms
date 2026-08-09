import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, models, recording, state, walker

logger = logging.getLogger(__name__)
WINDOW_BUDGET = 600
BACKWARD_CONTEXT_BUDGET = 400
FORWARD_CONTEXT_BUDGET = 400


class DSPyAtomicFact(BaseModel):
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
      emit it once. When a sentence asserts X and then gives a reason
      ("X because Y"), emit X as one fact — do NOT emit a second
      near-identical fact that restates the whole sentence including the
      reason clause.
    - META-TEXT IS NOT A FACT. Do not emit facts about the document itself:
      "The text states that …", "The author writes …", "This passage
      says …", "The book now turns to …". These are about the writing, not
      the subject. Extract the subject-matter claim they describe, or
      nothing if there is none.
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
    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
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
        return asyncio.run(
            self.aforward(
                current_nodes=current_nodes,
                context_before=context_before,
                context_after=context_after,
            )
        )


async def extract_atomic_facts(
    nodes: list[models.ASTNode],
    module: AtomicFactExtractor,
    max_concurrency: int | None = None,
) -> list[models.AtomicFact]:
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


class AtomicFactNode:
    def __init__(self, module: AtomicFactExtractor) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        nodes = state.get('nodes', [])
        facts = await extract_atomic_facts(nodes, module=self.module)
        return {'atomic_facts': facts}

