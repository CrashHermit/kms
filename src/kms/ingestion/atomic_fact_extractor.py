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

* EXTRACTED-ARTIFACT CONTEXT. The equations and variable bindings already
  extracted from the window's nodes ride along as read-only context. The
  model may use the artifact names (canonical equation names, variable
  meanings) when writing facts. Restatement is harmless — the concept
  pass deduplicates later — but using the structured names produces
  richer fact text.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, models, state
from kms.core.recorder import Recorder

logger = logging.getLogger(__name__)

# Fixed context window over whole nodes (~4 chars/token). A single node
# larger than the budget still forms a window of its own.
WINDOW_BUDGET = 2000


class WindowNode(BaseModel):
    """One node of the fixed window as the extractor sees it."""

    node_id: int
    type: str
    content: str | None = None


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


class ContextEquation(BaseModel):
    """An equation already extracted from the window, read-only context."""

    latex: str
    name: str | None = None


class ContextVariable(BaseModel):
    """A variable binding already extracted from the window, read-only
    context."""

    symbol: str
    meaning: str
    kind: str


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

    EXTRACTED ARTIFACTS. An optional EQUATIONS list and an optional
    VARIABLES list give equations and variable bindings already extracted
    from these nodes. They are names the fact pass may use when writing
    facts — e.g. referencing an equation by its canonical name rather
    than restating it — but they supplement the node content, never
    replace it.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description=(
            "The window's nodes, in document order, each with its id, type, "
            'and content.'
        )
    )
    equations: list[ContextEquation] | None = dspy.InputField(
        default=None,
        description=(
            'Equations already extracted from these nodes. Read-only context '
            '— do not restate them as facts.'
        ),
    )
    variables: list[ContextVariable] | None = dspy.InputField(
        default=None,
        description=(
            'Variable bindings already extracted from these nodes. Read-only '
            'context — do not restate them as facts.'
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
        current_nodes: list[WindowNode],
        equations: list[models.Equation] | None = None,
        variables: list[models.Variable] | None = None,
    ) -> list[models.AtomicFact]:
        """Extract the atomic facts from one window.

        Args:
            current_nodes: The window's nodes, in document order.
            equations: Equations already extracted from these nodes,
                read-only context.
            variables: Variable bindings already extracted from these nodes,
                read-only context.

        Returns:
            The atomic facts found, or an empty list.
        """
        dspy_equations = (
            [ContextEquation(latex=eq.latex, name=eq.name) for eq in equations]
            if equations
            else None
        )
        dspy_variables = (
            [
                ContextVariable(
                    symbol=variable.symbol,
                    meaning=variable.meaning,
                    kind=variable.kind,
                )
                for variable in variables
            ]
            if variables
            else None
        )
        result = await self.extractor.acall(
            current_nodes=current_nodes,
            equations=dspy_equations,
            variables=dspy_variables,
        )
        if self._recorder:
            self._recorder.record(
                'atomic_fact_extractor',
                {
                    'current_nodes': [
                        node.model_dump() for node in current_nodes
                    ],
                    'equations': [eq.model_dump() for eq in dspy_equations]
                    if dspy_equations
                    else None,
                    'variables': [
                        variable.model_dump() for variable in dspy_variables
                    ]
                    if dspy_variables
                    else None,
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
        current_nodes: list[WindowNode],
        equations: list[models.Equation] | None = None,
        variables: list[models.Variable] | None = None,
    ) -> list[models.AtomicFact]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(
            self.aforward(
                current_nodes=current_nodes,
                equations=equations,
                variables=variables,
            )
        )


# ============================================================================
# Window cutting
# ============================================================================


def _window_nodes(
    nodes: list[models.ASTNode], budget: int = WINDOW_BUDGET
) -> list[list[models.ASTNode]]:
    """Cut the node stream into adjacent fixed windows of whole nodes.

    ``image`` nodes (placeholder references, no text) are dropped; every
    other node with content is eligible. Windows are adjacent and
    non-overlapping, in document order. A window always contains at least one
    node — a single node larger than the budget forms a window of its own.

    Args:
        nodes: The flat node stream.
        budget: The fixed content budget in characters.

    Returns:
        The windows, each a list of nodes in document order.
    """
    eligible = [
        node
        for node in nodes
        if node.id is not None
        and node.content
        and node.content.strip()
        and node.type != 'image'
    ]

    windows: list[list[models.ASTNode]] = []
    current: list[models.ASTNode] = []
    current_size = 0
    for node in eligible:
        size = len(node.content or '')
        if current and current_size + size > budget:
            windows.append(current)
            current = []
            current_size = 0
        current.append(node)
        current_size += size
    if current:
        windows.append(current)
    return windows


# ============================================================================
# Entry point
# ============================================================================


async def extract_atomic_facts(
    nodes: list[models.ASTNode],
    module: AtomicFactExtractor,
    equations: list[tuple[int, list[models.Equation]]] | None = None,
    variables: list[tuple[int, list[models.Variable]]] | None = None,
    max_concurrency: int | None = None,
) -> list[models.AtomicFact]:
    """Extract atomic facts from the whole node stream.

    The stream is cut into adjacent fixed windows of whole nodes; every
    window is decomposed concurrently, and the facts are collected in
    document order. The pass is a pure reader of ``nodes`` — nothing writes
    back to the stream.

    The equations and variables already extracted from each window's nodes
    ride along as read-only context (the model may reference them but must
    not restate them as facts).

    Args:
        nodes: The flat node stream.
        module: The atomic fact extractor.
        equations: The ``(node_id, equations)`` channel from the
            equation/variable node, read-only context.
        variables: The ``(node_id, variables)`` channel from the
            equation/variable node, read-only context.
        max_concurrency: Windows in flight at once. None uses
            ``llm.MAX_CONCURRENT_CALLS``.

    Returns:
        The atomic facts, in document order.
    """
    windows = _window_nodes(nodes)
    if not windows:
        logger.info('atomic fact extractor: no windows')
        return []

    equations_by_node = dict(equations or [])
    variables_by_node = dict(variables or [])
    gate = llm.gate(max_concurrency)

    async def _extract_one(
        window: list[models.ASTNode],
    ) -> list[models.AtomicFact]:
        node_ids = [node.id for node in window if node.id is not None]
        window_equations = [
            eq
            for node_id in node_ids
            for eq in equations_by_node.get(node_id, [])
        ]
        window_variables = [
            variable
            for node_id in node_ids
            for variable in variables_by_node.get(node_id, [])
        ]
        async with gate:
            return await module.aforward(
                [
                    WindowNode(
                        node_id=node.id,
                        type=node.type,
                        content=node.content,
                    )
                    for node in window
                ],
                equations=window_equations or None,
                variables=window_variables or None,
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

    Runs after the equation/variable node — whose equations and variable
    bindings ride along as read-only context the model may reference by
    name — and before the ingestion persister. Reads only the final
    ``nodes`` stream and the artifact channels; writes the ``atomic_facts``
    channel.

    Args:
        module: The atomic fact extractor.
    """

    def __init__(self, module: AtomicFactExtractor) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Extract atomic facts from the final node stream.

        Args:
            state: The pipeline state, holding the node stream and the
                equation/variable artifact channels.

        Returns:
            The ``atomic_facts`` channel.
        """
        nodes = state.get('nodes', [])
        facts = await extract_atomic_facts(
            nodes,
            module=self.module,
            equations=state.get('equations'),
            variables=state.get('variables'),
        )
        return {'atomic_facts': facts}
