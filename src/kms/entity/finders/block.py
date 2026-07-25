r"""
Block finder — a cursor-walk over the flat structural node stream that lifts out every labeled
pedagogical block, whatever a textbook calls it.

This is the generalization of the three per-type finders (problem / definition / theorem) into one
(``docs/GENERALIZATION.md``, "Entity layer"). Two things make that collapse safe, and they are worth
stating because they are what the design turns on:

* **The walk was never type-specific.** The three finders were copies of one cursor-walk differing
  only in the "what am I looking for" clause of their Signature; the entity's type came from *which*
  finder ran (``Entity(type=PROBLEM)``), never from a classification. So widening that one clause and
  running the walk once is the same machinery doing the same job — not a new, weaker method.
* **Detecting a labeled block is genre, not domain.** Every textbook — math, physics, biology — sets
  its claims and its tasks off from the prose with a label ("Theorem 3.2", "Example 6.7", "Law of
  Conservation of Energy"). That structure is pedagogical, so one prompt finds it across domains
  while a *typed* prompt would need a new vocabulary per subject.

The finder emits **spans only, never a type**: it says where a block starts and stops. What kind of
block it is is induced downstream by the universal attributor, which reads the content anyway
(``entity/attributors/universal.py``), and whether it has something to work out is decided by the
procedure finder. Keeping detection and typing apart is what lets the type vocabulary stay open.

The walk itself is unchanged from ``finders/problem.py``, verbatim in its structure:

  * A cursor moves along the node stream. From the cursor it takes a *look-ahead window* of whole
    nodes up to a soft token budget, and the LLM returns the blocks inside it, each as an inclusive
    [start, end] span of local positions.
  * How far the cursor advances is decided *structurally* — no self-report from the LLM. A block is
    only "banked" once a node is seen to follow it, so it can never be split by a window cut:
      - bank every block whose end is BEFORE the window's edge (a node follows it → bounded) and
        advance the cursor to just after the last such block;
      - if the ONLY block reaches the window's edge, it may continue past the cut — so instead of
        banking it, GROW the window (double the budget) and re-read from the same cursor, repeating
        until a node follows it (bounded) or the document ends.
    Growing, not rewinding, means "a block bigger than the window" stops being a special case: the
    window just expands until the block is whole, so nothing is ever truncated and no size guard is
    needed. Termination is automatic — growth strictly increases and eventually reaches the document
    end, which banks the final block outright. A ``MAX_LOOKAHEAD_BUDGET`` cap bounds a pathologically
    long block to the model's context (banked as-is at the cap); that is a resource limit at the edge
    of the system, not part of the core rule.

Entities are a sparse OVERLAY: nodes keep their stable ids; a block just records the node ids that
are its members. Nothing about the node list is mutated or renumbered, so the forward walk emits
blocks already in document order.

The finder is wired into the pipeline by ``BlockFinderNode`` (bottom of this file), which runs the
walk over the flat node stream and writes its entities to the ``block_entities`` channel.
"""

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, models, state

# Soft look-ahead budget (~4 chars/token). A single node larger than the budget still forms a window
# (at least one node). When the only block in a window reaches its edge, the window grows (doubling)
# until it is bounded or the document ends — capped so a pathological block can't grow past the
# model's context (banked as-is there).
LOOKAHEAD_BUDGET = 2000
MAX_LOOKAHEAD_BUDGET = 8000


def _estimate_tokens(node: models.ASTNode) -> int:
    return len(node.content or '') // 4 + 1


class WindowNode(BaseModel):
    """One look-ahead node as the LLM sees it: a local position, its content, and its role
    annotation (the splitter marks an exercise lead-in with role "instruction")."""

    position: int
    type: str
    content: str | None = None
    role: str = ''


class BlockSpan(BaseModel):
    """A block the LLM found, as an inclusive span of local positions in the window."""

    start: int = Field(
        description='First local position of the block (inclusive).'
    )
    end: int = Field(
        description='Last local position of the block (inclusive).'
    )


class Signature(dspy.Signature):
    r"""
    Find the labeled BLOCKS in a run of textbook nodes and return each as a span of node positions.
    Anchor on the node that opens a block, gather the run of nodes that belongs to it, stop at its
    boundary. This is domain-neutral — it applies to ANY textbook (mathematics, physics, biology,
    computer science, …).

    WHAT IS A BLOCK:
    A block is a self-contained unit the book sets off from its running prose — the things a reader
    would look up, quote, or work through. This covers BOTH:
    - a STATEMENT the book asserts: a definition, a theorem, proposition, lemma or corollary, and
      their non-mathematical equivalents — a law, a principle, a rule, a model, a named mechanism —
      TOGETHER WITH its proof or derivation if one is shown;
    - a TASK posed for the reader: a worked EXAMPLE from the exposition (usually with a shown
      solution) and an EXERCISE from a problem set (usually with none — the reader is meant to solve
      it), together with its solution if one is shown.
    Treat them all the same: they are blocks. Do NOT require a solution or proof to be present, and
    do NOT decide what kind of block it is — only where it starts and ends.

    NOT BLOCKS: ordinary narrative prose, section headers, figures and their captions, tables of
    contents, running remarks and asides that carry no label. Ignore them.

    EXERCISE LEAD-INS ARE BOUNDARIES, NEVER MEMBERS. A grouped-exercise LEAD-IN — a directive that
    introduces a run of exercises and states a shared instruction ("For the following exercises, find
    the domain and range.", "In Exercises 3-8, graph the given relation.") — is NOT a block and is
    NEVER part of any block's span. Such a node has NO number of its own and is marked with role
    "instruction"; treat it exactly like a section header — a boundary. The exercises it governs are
    SEPARATE blocks that FOLLOW it: begin the first one at the first exercise node AFTER the lead-in,
    never at the lead-in itself, and never extend a preceding block forward to absorb it. (A later
    pass attaches the lead-in's shared instruction to those blocks; the finder's only job here is to
    not swallow the lead-in.)

    EXTENT (what nodes a block's span includes):
    - START at the block's OWN label/heading. A block usually opens with a short label that is a
      SEPARATE node from its body — e.g. a node that is just "Example 6.7", "Theorem 3.2",
      "Definition 1.4", "6.3 Check Your Understanding", or "Newton's Second Law". That label node is
      the FIRST node of the block: ALWAYS include it and begin the span there, not at the body node
      after it. (A block's own label is NOT the same as a section heading like "Matrix Operations",
      which names a section and is a boundary — never part of a span. When a heading names a specific
      block, it belongs to that block; when it names a section, it does not.)
    - Its body, with subparts kept together: a stem with parts (a)(b)(c) or (i)(ii)(iii) is ONE
      block; a repeated base number with letter suffixes (12a, 12b, 12c) is ONE block. Do NOT split
      subparts into separate blocks.
    - Its worked part IF shown — a proof, a derivation, a solution, an answer (prose, display math,
      steps): include it in the SAME span as the statement or task it belongs to. A proof is not a
      block of its own; it is part of the theorem it proves. Do not include it if none is shown.
    - Stop at the boundary: the next block's label, a section header, an exercise lead-in (role
      "instruction"), or a clear return to ordinary narrative.

    SEPARATE BLOCKS: distinct labels are distinct blocks (exercise 12 and exercise 13 are two spans,
    never merged; a definition followed by a theorem is two spans). A worked example and a following
    exercise are two blocks.

    POSITIONS:
    - Emit spans over the given nodes ONLY, using their `position` values; a span is the inclusive
      [start, end] range it occupies.
    - Return the blocks in document order.
    - Include a block even if it is unfinished at the last given node — still emit it, spanning it
      out to that last node.
    - If there are no blocks in the window, return an empty list.
    """

    current_nodes: list[WindowNode] = dspy.InputField(
        description="The look-ahead window's nodes, in order, each with a local position and a role "
        '(role "instruction" marks an exercise lead-in — a boundary, never part of a span). '
        'Emit spans over these only.'
    )
    blocks: list[BlockSpan] = dspy.OutputField(
        description='The blocks found in current_nodes, as position spans, in document order. Empty list if none.'
    )


class Module(dspy.Module):
    """Finds labeled block spans in the node stream."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.finder = dspy.ChainOfThought(Signature)
        self.set_lm(language_model or llm.text_lm())

    async def aforward(
        self, current_nodes: list[WindowNode]
    ) -> list[BlockSpan]:
        """Returns the block spans found in the given window of nodes."""
        result = await self.finder.acall(current_nodes=current_nodes)
        return list(result.blocks or [])


def _window_from(nodes: list[models.ASTNode], cursor: int, budget: int) -> int:
    """Return the exclusive end index of a look-ahead window starting at `cursor`:
    whole nodes up to the soft token budget, always at least one node."""
    i, accumulated = cursor, 0
    node_count = len(nodes)
    while i < node_count:
        token_count = _estimate_tokens(nodes[i])
        if i > cursor and accumulated + token_count > budget:
            break
        accumulated += token_count
        i += 1
    return i


async def find_blocks(
    nodes: list[models.ASTNode],
    module: Module | None = None,
    budget: int = LOOKAHEAD_BUDGET,
    max_budget: int = MAX_LOOKAHEAD_BUDGET,
) -> list[models.Entity]:
    """Cursor-walk the node stream and return untyped block entities (sparse overlay).

    From the cursor, read a look-ahead window and ask the LLM for the blocks in it. Bank every block
    a node is seen to follow (bounded) and advance past them; if the only block reaches the window's
    edge it may continue, so grow the window and re-read from the same cursor until a node follows it
    (bounded) or the document ends. Growing — never rewinding — captures a block larger than the
    window whole rather than truncating it, and needs no size guard: growth terminates at the document
    end (or the ``max_budget`` context cap, the one place a rare truncation can remain).

    The returned entities carry members and nothing else: their ``type`` is left None for the
    universal attributor to induce.
    """
    module = module or Module()
    blocks: list[models.Entity] = []
    cursor, node_count = 0, len(nodes)

    while cursor < node_count:
        size = budget
        while True:
            end = _window_from(nodes, cursor, size)
            window = nodes[cursor:end]
            last_local = len(window) - 1
            reached_doc_end = end == node_count

            spans = await module.aforward(
                [
                    WindowNode(
                        position=k,
                        type=(node.type.value if node.type else ''),
                        content=node.content,
                        role=(node.role or ''),
                    )
                    for k, node in enumerate(window)
                ]
            )
            # Clamp to range, drop empties, keep document order.
            clean: list[BlockSpan] = []
            for span in spans:
                start = min(max(span.start, 0), last_local)
                stop = min(max(span.end, start), last_local)
                clean.append(BlockSpan(start=start, end=stop))
            clean.sort(key=lambda span: span.start)

            if not clean:
                cursor = end  # only prose in this window — skip it
                break

            # A block is bounded when a node is seen to follow it inside the window.
            bounded = [span for span in clean if span.end < last_local]

            if reached_doc_end or size >= max_budget:
                # Nothing left to gather (document end), or the window hit the context cap: bank
                # every block as-is and advance past the window.
                to_bank, advance = clean, end
            elif bounded:
                # Commit the bounded blocks; the cursor lands just after the last one (any trailing
                # prose / an unbanked edge block is re-read next).
                to_bank, advance = bounded, cursor + bounded[-1].end + 1
            else:
                # The sole block reaches the edge and may continue — grow and re-read.
                size *= 2
                continue

            for span in to_bank:
                ids = [
                    window[k].id
                    for k in range(span.start, span.end + 1)
                    if window[k].id is not None
                ]
                if ids:
                    blocks.append(models.Entity(members=ids))
            cursor = advance
            break

    return blocks


# --- LangGraph node: emit the found blocks onto their channel ---


class BlockFinderNode:
    """Walks the flat node stream and writes its block entities to the ``block_entities`` channel.

    The walk is one sequential unit (a growing look-ahead cursor cannot be sharded), so this is a
    plain graph node rather than the map-reduce dispatch/worker/collect shape the parallel stages
    use."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: state.State) -> dict:
        """Walks the node stream and writes block entities to the channel."""
        blocks = await find_blocks(state.get('nodes', []), module=self.module)
        return {'block_entities': blocks}
