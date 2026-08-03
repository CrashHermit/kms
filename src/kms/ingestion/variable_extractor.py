r"""
Variable binding extraction — one LangGraph node, one DSPy module.

The node cuts the node stream into fixed adjacent windows of whole nodes
(the same cutter the atomic fact pass uses) and runs one windowed pass over
each: the variable extractor finds stand-in bindings — compact symbols,
expressions, or labels that stand for something fuller — and gives each
one's symbol, meaning, and kind.

Windowed exactly as the atomic fact and entity passes are: one DSPy call
per fixed window, no per-node round-trip. Node attribution travels as a
``node_id`` on every binding — exactly as ``fact_index`` does in the entity
pass — so the channel keeps its ``(node_id, [result])`` shape and the
persister needs no changes.

The equation extractor is gone (ADR 0001, step 4): equations are folded
into facts. The fact pass already states them (its text carries the
delimited LaTeX), so equation identity now belongs to the concept pass —
clustering facts about the same equation across books — instead of a
parallel ``(source, node_id, index)`` vertex space that could never merge.
The ``:Equation`` tier, its writer, and the variable→equation attachment
are deleted with it; every binding hangs off its ``:Node``.

Artifacts hang directly off the ``:Node`` they were extracted from via
``:HAS_VARIABLE``. Statement and procedure hubs inherit them through
``:MEMBER_OF`` — every variable is reachable from every hub that covers its
provenance node.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, logs, models, state, walker
from kms.core.recorder import Recorder

logger = logging.getLogger(__name__)

# Fixed context window over whole nodes (~4 chars/token), the same budget
# as the atomic fact pass. A single node larger than the budget still forms
# a window of its own.
WINDOW_BUDGET = 2000


# ============================================================================
# DSPy Pydantic models
# ============================================================================


class DSPyVariable(BaseModel):
    """A single stand-in binding emitted by the variable extractor."""

    node_id: int = Field(
        description=(
            'The id of the node this binding was extracted from — one of '
            'the given nodes.'
        )
    )
    symbol: str = Field(
        description=(
            r'The compact notation — a single symbol, an expression, or a '
            r'short label. Mathematical notation KEEPS its LaTeX '
            r'delimiters exactly as in the source: "$X$" not "X", '
            r'"$d|_{Y \times Y}$" not "d|_{Y \times Y}". A label that is '
            r'not mathematical notation (an abbreviation, a code '
            r'identifier, a table heading) stays as written, undelimited.'
        )
    )
    meaning: str = Field(
        description='What the symbol stands for, in plain language'
    )
    kind: str = Field(
        description='What sort of thing: variable, constant, parameter, element, unit, abbreviation, function, operator, etc.'
    )


# ============================================================================
# Variable extractor
# ============================================================================


class VariableSignature(dspy.Signature):
    r"""
    You are given a run of nodes from a document, in document order. Each
    node carries its id, its structural type, and its content. Find every
    stand-in notation — compact symbols, expressions, or labels that stand
    for something fuller — and give each one's node_id, symbol, meaning,
    and kind.

    WHAT COUNTS AS A STAND-IN

    A stand-in is any notation where a compact form represents a fuller meaning
    within a bounded region. This includes:

    - Mathematical variables and constants: "Let $\alpha$ be the learning
      rate" — $\alpha$ stands for the learning rate.
    - Chemical symbols and formulas: "2H₂ + O₂ → 2H₂O" — H stands for
      hydrogen, O for oxygen.
    - Circuit and schematic labels: "$R_1 = 10k\Omega$" — $R_1$ stands for
      resistor 1.
    - Code identifiers with semantic meaning: "def train(model, lr=0.01)" —
      ``lr`` stands for learning rate.
    - Defined abbreviations and aliases: "the Company (hereinafter 'Acme')" —
      ``Acme`` stands for Acme Corporation.
    - Table headings and row labels that name a quantity.
    - Greek letters, single letters, multi-character symbols used as
      placeholders in technical writing.
    - Units and dimensions: "the distance $d$ (in metres)" — $d$ stands for
      distance.

    Every stand-in must carry:
    - node_id: which of the given nodes it came from.
    - meaning: What the symbol stands for, in plain language.
    - kind: What sort of thing — variable, constant, parameter, element, unit,
      abbreviation, function, operator, etc.

    SYMBOL FORMAT. Mathematical notation keeps its LaTeX delimiters, exactly
    as in the source: emit "$X$", never bare "X"; "$d'$", never "d'". The
    symbol string is part of this binding's stored identity, so an
    undelimited symbol and a delimited one become two different vertices for
    the same thing. Labels that are not mathematical notation — an
    abbreviation like "Acme", a code identifier like ``lr`` — stay
    undelimited.

    WHAT DOES NOT COUNT
    - A symbol merely used, not defined or bound ("$f(x) = x^2$" with no
      prior binding of $f$ or $x$).
    - An entity name. "Albert Einstein" is the thing, not a stand-in.
    - A heading or section number.
    - A footnote superscript or citation bracket.

    CONTEXT-ONLY NODES. header (a title), bibliographic (a reference
    entry), and caption nodes are context to help you place the bindings
    — do NOT extract bindings from them.

    WHEN IN DOUBT, INCLUDE IT. A downstream stage can filter.
    """

    current_nodes: list[walker.WindowNode] = dspy.InputField(
        description=(
            "The window's nodes, in document order, each with its id, type, "
            'and content.'
        )
    )
    variables: list[DSPyVariable] = dspy.OutputField(
        description='Every stand-in notation found, with its node_id, symbol, meaning, and kind. Empty if none.'
    )


class VariableExtractor(dspy.Module):
    """Extracts stand-in bindings from one fixed window of nodes.

    Args:
        language_model: The LM to run on.
    """

    def __init__(
        self, language_model: dspy.LM, recorder: Recorder | None = None
    ) -> None:
        super().__init__()
        self.extractor = dspy.ChainOfThought(VariableSignature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self, current_nodes: list[walker.WindowNode]
    ) -> list[tuple[int, models.Variable]]:
        """Extract every stand-in binding from one window.

        Args:
            current_nodes: The window's nodes, in document order.

        Returns:
            ``(node_id, variable)`` pairs, in the order the model emitted
            them.
        """
        result = await self.extractor.acall(current_nodes=current_nodes)
        if self._recorder:
            self._recorder.record(
                'variable_extractor',
                {
                    'current_nodes': [
                        node.model_dump() for node in current_nodes
                    ],
                },
                result,
            )
        variables = [
            (
                variable.node_id,
                models.Variable(
                    symbol=variable.symbol,
                    meaning=variable.meaning,
                    kind=variable.kind,
                ),
            )
            for variable in (result.variables or [])
        ]
        logger.debug(
            'variable extractor: %d node(s) -> %d variable(s) | %s',
            len(current_nodes),
            len(variables),
            logs.counts([variable.symbol for _, variable in variables]),
        )
        return variables

    def forward(
        self, current_nodes: list[walker.WindowNode]
    ) -> list[tuple[int, models.Variable]]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(current_nodes=current_nodes))


# ============================================================================
# Entry point — orchestrates the module over fixed windows
# ============================================================================


def _group_by_node[T](
    pairs: list[tuple[int, T]], node_order: list[int]
) -> list[tuple[int, list[T]]]:
    """Group ``(node_id, item)`` pairs back into the channel shape.

    Channel entries follow the window's node order; items within a node keep
    the model's emission order. ``node_id`` values that name no node in the
    window are dropped — the model must not mint vertices for nodes it was
    not shown.

    Args:
        pairs: The ``(node_id, item)`` pairs from one window.
        node_order: The window's node ids, in document order.

    Returns:
        The channel entries, ``(node_id, [item])``, in *node_order*.
    """
    by_node: dict[int, list[T]] = {}
    for node_id, item in pairs:
        by_node.setdefault(node_id, []).append(item)
    return [
        (node_id, by_node[node_id])
        for node_id in node_order
        if node_id in by_node
    ]


async def _extract_window(
    window: list[models.ASTNode],
    variable_module: VariableExtractor,
    gate: asyncio.Semaphore,
) -> list[tuple[int, models.Variable]]:
    """Run the extractor over one fixed window.

    The call sits inside the semaphore, which therefore bounds windows in
    flight rather than raw requests.

    Args:
        window: The window's nodes, in document order.
        variable_module: The variable extractor.
        gate: The stage's concurrency limiter.

    Returns:
        The ``(node_id, variable)`` pairs for the window, in the model's
        emission order.
    """
    window_nodes = [
        walker.WindowNode(node_id=node.id, type=node.type, content=node.content)
        for node in window
    ]
    async with gate:
        return await variable_module.aforward(window_nodes)


async def extract_variables(
    nodes: list[models.ASTNode],
    variable_module: VariableExtractor,
    max_concurrency: int | None = None,
) -> list[tuple[int, list[models.Variable]]]:
    """Extract variable bindings from the whole node stream.

    The stream is cut into adjacent fixed windows of whole nodes
    (``walker.fixed_windows``, the same cutter the atomic fact pass uses);
    every window is processed concurrently, and the channel is collected in
    document order. Each binding carries the ``node_id`` of the node it was
    extracted from, so the channel keeps its ``(node_id, [result])`` shape.

    The pass is a pure reader of ``nodes`` — nothing writes back to the
    stream, so every window sees the same stream it would have seen running
    serially.

    Args:
        nodes: The flat node stream.
        variable_module: The variable extractor.
        max_concurrency: Windows in flight at once. None uses
            ``llm.MAX_CONCURRENT_CALLS``.

    Returns:
        The ``variables`` channel — ``(node_id, [Variable])`` entries, in
        document order.
    """
    windows = walker.fixed_windows(nodes, WINDOW_BUDGET)
    if not windows:
        logger.info('variable extractor: no windows')
        return []

    gate = llm.gate(max_concurrency)
    per_window = await asyncio.gather(
        *(_extract_window(window, variable_module, gate) for window in windows)
    )

    variables_result: list[tuple[int, list[models.Variable]]] = []
    for window, window_variables in zip(windows, per_window, strict=True):
        node_order = [node.id for node in window if node.id is not None]
        variables_result.extend(_group_by_node(window_variables, node_order))

    logger.info(
        'variable extractor: %d node(s) in %d window(s) -> '
        '%d variable binding(s)',
        len(nodes),
        len(windows),
        sum(len(bindings) for _, bindings in variables_result),
    )
    return variables_result


# ============================================================================
# LangGraph node
# ============================================================================


class VariableNode:
    """Extracts variable bindings from the node stream.

    Runs after the partitioners (which narrow hub memberships) and before
    the atomic fact pass. Reads only the final ``nodes`` stream; writes the
    ``variables`` channel.

    Args:
        module: The variable extractor.
    """

    def __init__(self, module: VariableExtractor) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Extract variable bindings from every eligible node.

        Args:
            state: The pipeline state, holding the node stream.

        Returns:
            The ``variables`` channel.
        """
        nodes = state.get('nodes', [])
        variables = await extract_variables(nodes, module=self.module)
        return {'variables': variables}
