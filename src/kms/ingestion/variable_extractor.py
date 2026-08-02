r"""
Equation and variable extraction — one LangGraph node, three DSPy modules.

The node runs over every content node in the stream. For each:

1. **Router** — classifies the node: does it contain equations, variable
   bindings, both, or neither? Two independent booleans, one cheap call.

2. **Equation extractor** — runs when ``has_equation`` is true. Extracts
   equations from the node's content, giving each a LaTeX form, an optional
   identity (resolved against the existing graph), and an optional domain.

3. **Variable extractor** — runs when ``has_variable`` is true. Extracts
   stand-in bindings from the node's content, with the equation list as
   additional context when available — the same symbol resolution task, just
   richer input.

All three modules share the same context window (content + before + after).
The equation extractor feeds its output into the variable extractor so a
variable inside ``$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$``
inherits the equation's identity as evidence for its meaning.

Artifacts hang directly off the ``:Node`` they were extracted from via
``:HAS_EQUATION`` / ``:HAS_VARIABLE``. Statement and procedure hubs inherit
them through ``:MEMBER_OF`` — every equation and variable is reachable from
every hub that covers its provenance node.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import logs, models, state, walker
from kms.core.recorder import Recorder

logger = logging.getLogger(__name__)

# Token budgets for the context window on either side of the focus node.
DEFAULT_BACKWARD_BUDGET = 500
DEFAULT_FORWARD_BUDGET = 200


# ============================================================================
# DSPy Pydantic models
# ============================================================================


class DSPyEquation(BaseModel):
    """One equation emitted by the equation extractor."""

    latex: str = Field(
        description=(
            r'The equation in LaTeX notation, WITH its surrounding '
            r'delimiters: \$\$...\$\$ or \$...\$ or \(...\) or '
            r'\[...\] — exactly as it appears in the source'
        )
    )
    name: str | None = Field(
        default=None,
        description='The canonical name of this equation if known (e.g. "heat equation", "Schrödinger equation", "Ohm\'s law"). Null if unknown.',
    )
    domain: str | None = Field(
        default=None,
        description='The domain of the equation (e.g. "physics", "chemistry", "circuit analysis"). Null if uncertain.',
    )


class DSPyVariable(BaseModel):
    """A single stand-in binding emitted by the variable extractor."""

    symbol: str = Field(
        description='The compact notation — a single symbol, an expression, or a short label'
    )
    meaning: str = Field(
        description='What the symbol stands for, in plain language'
    )
    kind: str = Field(
        description='What sort of thing: variable, constant, parameter, element, unit, abbreviation, function, operator, etc.'
    )


# ============================================================================
# 1. Router
# ============================================================================


class RouterSignature(dspy.Signature):
    r"""
    You are given the text of one block from a document. Two optional context
    sections surround it. Answer two yes/no questions about this block.

    1. Does this block contain an EQUATION — a relationship between two
       or more quantities expressed in notation? This is always a
       COMPLETE statement with a relational operator (=, ≤, →, ⇌, etc.)
       connecting two sides:
         * display math: $$y = mx + b$$, $$\nabla \cdot \mathbf{E} = \rho/\varepsilon_0$$
         * inline math that states a relationship: $E = mc^2$
         * chemical reactions: 2H₂ + O₂ → 2H₂O
         * schematic formulas: $V = IR$

       NOT an equation — do NOT flag these:
         * a decorated variable: $L$-smooth, $k$-means, $\alpha$-mixing
         * a single expression: $x^2 + 3x$, $\frac{1}{k}$
         * a bound or rate used attributively: $O(1/k)$, $\|x\|_2$
         * a bare inequality that qualifies rather than states: $\epsilon > 0$

    2. Does this block contain a VARIABLE BINDING — a place where a compact
       notation is explicitly bound to a meaning? This includes "Let $\alpha$
       be the learning rate", "the thermal diffusivity $\alpha$", "H₂
       (hydrogen gas)", "def train(model, lr=0.01)", "hereinafter 'Acme'".

    Answer with two booleans. Answer honestly — if there is nothing to
    extract, say so.
    """

    content: str = dspy.InputField(
        description='The text of one document block.'
    )
    content_before: str | None = dspy.InputField(
        description='Preceding context, read-only.'
    )
    content_after: str | None = dspy.InputField(
        description='Following context, read-only.'
    )
    has_equation: bool = dspy.OutputField(
        description='True if this block contains an equation worth extracting.'
    )
    has_variable: bool = dspy.OutputField(
        description='True if this block contains variable bindings worth extracting.'
    )


class Router(dspy.Module):
    """Classifies a content node for equation / variable presence.

    Args:
        language_model: The LM to run on.
    """

    def __init__(
        self, language_model: dspy.LM, recorder: Recorder | None = None
    ) -> None:
        super().__init__()
        self.router = dspy.ChainOfThought(RouterSignature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self,
        content: str,
        content_before: str | None = None,
        content_after: str | None = None,
    ) -> tuple[bool, bool]:
        result = await self.router.acall(
            content=content,
            content_before=content_before,
            content_after=content_after,
        )
        if self._recorder:
            self._recorder.record(
                'router',
                {
                    'content': content,
                    'content_before': content_before,
                    'content_after': content_after,
                },
                result,
            )
        return bool(result.has_equation), bool(result.has_variable)

    def forward(
        self,
        content: str,
        content_before: str | None = None,
        content_after: str | None = None,
    ) -> tuple[bool, bool]:
        return asyncio.run(
            self.aforward(
                content=content,
                content_before=content_before,
                content_after=content_after,
            )
        )


# ============================================================================
# 2. Equation extractor
# ============================================================================


class EquationExtractorSignature(dspy.Signature):
    r"""
    You are given the text of one block from a document. Extract every
    equation — a standalone relationship expressed in notation — and give
    each one's LaTeX form, canonical name (if known), and domain.

    An equation is a COMPLETE RELATIONSHIP: two expressions connected by
    equality, inequality, reaction, or another relational operator. This
    includes:

    - Display math ($$...$$) and inline math ($...$) that expresses a
      complete relationship.
    - Chemical equations: "2H₂ + O₂ → 2H₂O".
    - Circuit laws: "$V = IR$", "$R_{in} = 10k\Omega$".
    - A code fragment that is a named formula: "f(x) = x² + 3x".
    - A table row that pairs a variable with its definition.

    What is NOT an equation:
    - A bare variable reference: "the value $x$", "the constant $c$".
    - A notation fragment: "$x \in \mathbb{R}$".
    - An expression without a relation: "$x^2 + 3x$".

    For each equation, give:
    - latex: The equation, always using \$\$...\$\$ for display math
      and \$...\$ for inline math. If the source uses \\(, \\),
      \\[, or \\], convert them to the dollar convention. Keep the
      LaTeX content between the delimiters exactly as written.
    - name: The canonical name if you recognise it. Null if unknown.
    - domain: The subject area. Null if uncertain.

    Return an empty list if there are no equations.
    """

    content: str = dspy.InputField(
        description='The text of one document block.'
    )
    content_before: str | None = dspy.InputField(
        description='Preceding context, read-only.'
    )
    content_after: str | None = dspy.InputField(
        description='Following context, read-only.'
    )
    equations: list[DSPyEquation] = dspy.OutputField(
        description='Every equation found, with its latex, name, and domain. Empty if none.'
    )


class EquationExtractor(dspy.Module):
    """Extracts equations from a content node.

    Args:
        language_model: The LM to run on.
    """

    def __init__(
        self, language_model: dspy.LM, recorder: Recorder | None = None
    ) -> None:
        super().__init__()
        self.extractor = dspy.ChainOfThought(EquationExtractorSignature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self,
        content: str,
        content_before: str | None = None,
        content_after: str | None = None,
    ) -> list[models.Equation]:
        result = await self.extractor.acall(
            content=content,
            content_before=content_before,
            content_after=content_after,
        )
        if self._recorder:
            self._recorder.record(
                'equation_extractor',
                {
                    'content': content,
                    'content_before': content_before,
                    'content_after': content_after,
                },
                result,
            )
        equations = [
            models.Equation(
                latex=eq.latex,
                name=eq.name,
                domain=eq.domain,
            )
            for eq in (result.equations or [])
        ]
        logger.debug(
            'equation extractor: %d char(s) -> %d equation(s)',
            len(content or ''),
            len(equations),
        )
        return equations

    def forward(
        self,
        content: str,
        content_before: str | None = None,
        content_after: str | None = None,
    ) -> list[models.Equation]:
        return asyncio.run(
            self.aforward(
                content=content,
                content_before=content_before,
                content_after=content_after,
            )
        )


# ============================================================================
# 3. Variable extractor (accepts equations as context)
# ============================================================================


class VariableSignature(dspy.Signature):
    r"""
    You are given the text of one block from a document. Find every stand-in
    notation — compact symbols, expressions, or labels that stand for
    something fuller — and give each one's symbol, meaning, and kind.

    Two optional context sections surround the block: PRECEDING CONTEXT and
    FOLLOWING CONTEXT. Read these to understand what the symbols mean — they
    often define or use the symbols found in the main content — but extract
    bindings ONLY from the main content, never from the context alone.

    An optional EQUATIONS list gives equations already extracted from this
    block, with their LaTeX form and canonical name (if known). When an
    equation is present, its identity is strong evidence for what its symbols
    mean. Use it.

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
    - meaning: What the symbol stands for, in plain language.
    - kind: What sort of thing — variable, constant, parameter, element, unit,
      abbreviation, function, operator, etc.

    WHAT DOES NOT COUNT
    - A symbol merely used, not defined or bound ("$f(x) = x^2$" with no
      prior binding of $f$ or $x$).
    - An entity name. "Albert Einstein" is the thing, not a stand-in.
    - A heading or section number.
    - A footnote superscript or citation bracket.

    WHEN IN DOUBT, INCLUDE IT. A downstream stage can filter.
    """

    content: str = dspy.InputField(
        description='The text of one document block.'
    )
    content_before: str | None = dspy.InputField(
        description='Preceding context, read-only.'
    )
    content_after: str | None = dspy.InputField(
        description='Following context, read-only.'
    )
    equations: list[DSPyEquation] | None = dspy.InputField(
        default=None,
        description='Equations already extracted from this block, with their LaTeX and name. Read-only context for resolving symbol meanings.',
    )
    variables: list[DSPyVariable] = dspy.OutputField(
        description='Every stand-in notation found, with its symbol, meaning, and kind. Empty if none.'
    )


class VariableExtractor(dspy.Module):
    """Extracts stand-in bindings from one content node.

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
        self,
        content: str,
        content_before: str | None = None,
        content_after: str | None = None,
        equations: list[models.Equation] | None = None,
    ) -> list[models.Variable]:
        """Extract every stand-in binding from one block of text.

        Args:
            content: The block's text.
            content_before: Optional preceding context, read-only.
            content_after: Optional following context, read-only.
            equations: Optional equations already extracted from this block.

        Returns:
            The bound variables found, or an empty list.
        """
        dspy_eqs = (
            [
                DSPyEquation(latex=eq.latex, name=eq.name, domain=eq.domain)
                for eq in equations
            ]
            if equations
            else None
        )
        result = await self.extractor.acall(
            content=content,
            content_before=content_before,
            content_after=content_after,
            equations=dspy_eqs,
        )
        if self._recorder:
            self._recorder.record(
                'variable_extractor',
                {
                    'content': content,
                    'content_before': content_before,
                    'content_after': content_after,
                    'equations': [eq.model_dump() for eq in dspy_eqs]
                    if dspy_eqs
                    else None,
                },
                result,
            )
        variables = [
            models.Variable(
                symbol=variable.symbol,
                meaning=variable.meaning,
                kind=variable.kind,
            )
            for variable in (result.variables or [])
        ]
        logger.debug(
            'variable extractor: %d char(s) -> %d variable(s) | %s',
            len(content or ''),
            len(variables),
            logs.counts([variable.symbol for variable in variables]),
        )
        return variables

    def forward(
        self,
        content: str,
        content_before: str | None = None,
        content_after: str | None = None,
        equations: list[models.Equation] | None = None,
    ) -> list[models.Variable]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(
            self.aforward(
                content=content,
                content_before=content_before,
                content_after=content_after,
                equations=equations,
            )
        )


# ============================================================================
# Entry point — orchestrates the three modules over every provenance node
# ============================================================================


async def extract_equations_and_variables(
    nodes: list[models.ASTNode],
    router_module: Router,
    equation_module: EquationExtractor,
    variable_module: VariableExtractor,
) -> tuple[
    list[tuple[int, list[models.Equation]]],
    list[tuple[int, list[models.Variable]]],
]:
    """Extract equations and variable bindings from every provenance node.

    Walks the raw node stream in document order. For each node with content:
    the router decides what to extract, then the equation and/or variable
    modules run as needed. Context windows are built from the surrounding
    nodes via ``walker.content_before`` / ``walker.content_after``. The
    equation output feeds into the variable extractor as additional context.

    Args:
        nodes: The flat node stream.
        router_module: The router.
        equation_module: The equation extractor.
        variable_module: The variable extractor.

    Returns:
        The ``(equations, variables)`` pair — each a list of
        ``(node_id, [result])`` entries.
    """
    router_module = router_module
    equation_module = equation_module
    variable_module = variable_module

    equations_result: list[tuple[int, list[models.Equation]]] = []
    variables_result: list[tuple[int, list[models.Variable]]] = []

    for position, node in enumerate(nodes):
        node_id = node.id
        content = node.content
        if node_id is None or not content or not content.strip():
            continue

        content_before = walker.content_before(
            nodes, position, budget=DEFAULT_BACKWARD_BUDGET
        )
        content_after = walker.content_after(
            nodes, position, budget=DEFAULT_FORWARD_BUDGET
        )

        has_equation, has_variable = await router_module.aforward(
            content=content,
            content_before=content_before,
            content_after=content_after,
        )

        node_equations: list[models.Equation] = []
        if has_equation:
            node_equations = await equation_module.aforward(
                content=content,
                content_before=content_before,
                content_after=content_after,
            )
            if node_equations:
                equations_result.append((node_id, node_equations))

        if has_variable:
            node_variables = await variable_module.aforward(
                content=content,
                content_before=content_before,
                content_after=content_after,
                equations=node_equations or None,
            )
            if node_variables:
                variables_result.append((node_id, node_variables))

    logger.info(
        'equations & variables: %d node(s) -> %d equation(s), %d variable binding(s) total',
        len(nodes),
        sum(len(eqs) for _, eqs in equations_result),
        sum(len(bindings) for _, bindings in variables_result),
    )
    return equations_result, variables_result


# ============================================================================
# LangGraph node
# ============================================================================


class EquationAndVariableNode:
    """Extracts equations and variable bindings from every provenance node.

    Runs after the partitioners (which narrow hub memberships) and before
    any fact extraction pass (which consumes both as context).

    Args:
        router_module: The router.
        equation_module: The equation extractor.
        variable_module: The variable extractor.
    """

    def __init__(
        self,
        router_module: Router,
        equation_module: EquationExtractor,
        variable_module: VariableExtractor,
    ) -> None:
        self.router_module = router_module
        self.equation_module = equation_module
        self.variable_module = variable_module

    async def run(self, state: state.State) -> dict:
        """Extract equations and variable bindings from every eligible node.

        Args:
            state: The pipeline state, holding the node stream.

        Returns:
            The ``equations`` and ``variables`` channels.
        """
        nodes = state.get('nodes', [])
        equations, variables = await extract_equations_and_variables(
            nodes,
            router_module=self.router_module,
            equation_module=self.equation_module,
            variable_module=self.variable_module,
        )
        return {'equations': equations, 'variables': variables}
