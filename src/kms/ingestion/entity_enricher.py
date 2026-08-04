r"""
Entity description enrichment — one LangGraph node, one DSPy module.

Walks the provenance node stream node-by-node. For each node that has
local triplets (via fact → node_ids), extracts the distinct subject and
object strings from those triplets, builds a token-bounded context window,
and asks the LLM to write a description for each.

Entities are PURELY LOCAL — no global list, no cross-node tracking, no
canonicalization. The same surface form appearing at two different nodes
produces two independent descriptions from two different contexts. That
is correct: the entity is whatever the text at that point says it is.

Design commitments:

* NODE-BOUND, NO GLOBAL STATE. The walk is over nodes in document order.
  Each node produces its own entities from its own triplets, described
  with its own context window. Nothing persists between nodes.

* TRIPLETS AS EVIDENCE. Each entity is presented with the raw triplets
  where it appears as subject or object, formatted as
  ``"subject | predicate | object"``. The LLM reads these to ground the
  description.

* ONE CALL PER NODE. All entities found at a node are described in one
  LLM call. Entities introduced together benefit from shared context.

* MINIMAL OUTPUT. The pass returns a ``node_entity_descriptions``
  channel: a ``dict[int, list[dict]]`` mapping node id to its entity
  descriptions. No ``Entity`` model — just ``{name, description}`` dicts.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, models, state, walker
from kms.core.recorder import Recorder

logger = logging.getLogger(__name__)

BACKWARD_CONTEXT_BUDGET = 200
FORWARD_CONTEXT_BUDGET = 200


class DSPyEntityDesc(BaseModel):
    """One entity description written by the enricher."""

    name: str = Field(
        description='The entity name exactly as given in the input.'
    )
    description: str = Field(
        description=(
            'A concise 1-2 sentence description of what this entity is. '
            'Grounded in the node content and its triplets — not invented. '
            'If the entity is a property or value (not a thing with '
            'identity), the description may simply restate that.'
        )
    )


class Signature(dspy.Signature):
    r"""
    You are given one node from a document — a block of text — plus the
    text immediately before and after it for context. You are also given a
    list of ENTITIES that appear in this region. Each entity has a NAME
    (the exact subject or object string from a local triplet) and a list
    of TRIPLETS where it appears.

    Your job: for each entity, write a concise 1-2 sentence description of
    what the entity IS. Ground the description in what the node and its
    context say about the entity. The triplets tell you what relations the
    entity participates in (e.g. "X is bounded", "Y has limit 0") — use
    them to inform the description, but the node text is the primary
    source.

    RULES:
    - GROUNDED, NOT INVENTED. Every description must be supported by the
      node text or its triplets. Do not guess.
    - CONCISE. One or two sentences. A description identifies, it does
      not summarize exhaustively.
    - STANDALONE. The description should be readable without the node
      context — someone reading it later should know what the entity is.
    - LATEX FORMAT. Everything that can be in LaTeX format is written in
      LaTeX WITH its delimiters: inline math in `$...$`, display math in
      `$$...$$`. This covers mathematical notation, formulas, variable
      names, sequence expressions, chemical formulas, units, and any
      other technical notation. Never write plain text or Unicode (no
      `x⁴`, `≤`, `α`, bare `H₂O`) when a LaTeX spelling exists. The
      entity name and description must both follow this — if the name
      contains LaTeX, keep its delimiters; if the description mentions
      math, write it in delimited LaTeX.
    - COVER EVERY ENTITY. Return one description per entity in the input.
    - CONTEXT IS PLACEMENT-ONLY. context_before and context_after frame
      where the node sits in the document — section, surrounding topic.
      They are not the primary source; the node is.
    """

    node_content: str = dspy.InputField(
        description='The central node — one block of the document.'
    )
    context_before: str | None = dspy.InputField(
        default=None,
        description='Text immediately before the node. CONTEXT ONLY.',
    )
    context_after: str | None = dspy.InputField(
        default=None,
        description='Text immediately after the node. CONTEXT ONLY.',
    )
    entities: list[dict] = dspy.InputField(
        description=(
            'Entities found in this region. Each entry has: name (the '
            'exact subject/object string), triplets (a list of '
            '"subject | predicate | object" strings where this name '
            'appears as subject or object).'
        )
    )
    descriptions: list[DSPyEntityDesc] = dspy.OutputField(
        description='One description per entity in the input list.'
    )


class EntityEnricher(dspy.Module):
    """Writes descriptions for entities found at a node.

    Args:
        language_model: The LM to run on.
    """

    def __init__(
        self, language_model: dspy.LM, recorder: Recorder | None = None
    ) -> None:
        super().__init__()
        self.enricher = dspy.ChainOfThought(Signature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(
        self,
        node_content: str,
        context_before: str | None,
        context_after: str | None,
        entities: list[dict],
    ) -> list[DSPyEntityDesc]:
        """Write descriptions for the given entities.

        Args:
            node_content: The anchor node's text.
            context_before: Text before the node, or None.
            context_after: Text after the node, or None.
            entities: The entities with their names and local triplets.

        Returns:
            One description per entity.
        """
        result = await self.enricher.acall(
            node_content=node_content,
            context_before=context_before or '',
            context_after=context_after or '',
            entities=entities,
        )
        if self._recorder:
            self._recorder.record(
                'entity_enricher',
                {
                    'node_content': node_content,
                    'context_before': context_before,
                    'context_after': context_after,
                    'entities': entities,
                },
                result,
            )
        descriptions = list(result.descriptions or [])
        logger.debug(
            'entity enricher: %d entity descriptions',
            len(descriptions),
        )
        return descriptions

    def forward(
        self,
        node_content: str,
        context_before: str | None,
        context_after: str | None,
        entities: list[dict],
    ) -> list[DSPyEntityDesc]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(
            self.aforward(
                node_content=node_content,
                context_before=context_before,
                context_after=context_after,
                entities=entities,
            )
        )


# ============================================================================
# Entry point
# ============================================================================


def _triplet_str(triplet: models.Triplet) -> str:
    """Render a triplet as a readable one-line string."""
    return f'{triplet.subject} | {triplet.predicate} | {triplet.object}'


async def enrich_entities(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    nodes: list[models.ASTNode],
    module: EntityEnricher,
    max_concurrency: int | None = None,
) -> dict[int, list[dict]]:
    """Walk the node stream and produce per-node entity descriptions.

    For each node that anchors at least one fact (and therefore at least
    one triplet), extracts the distinct subject/object strings from those
    local triplets, builds a token-bounded context window, and asks the
    LLM to describe every entity found at that node.

    Args:
        triplets: The triplets, in document order.
        facts: The atomic facts, in document order.
        nodes: The provenance node stream, in document order.
        module: The entity enricher.
        max_concurrency: Nodes in flight at once. None uses
            ``llm.MAX_CONCURRENT_CALLS``.

    Returns:
        A ``dict`` mapping ``node_id -> [{name, description}, ...]``.
        Nodes with no triplets or no entities are absent from the dict.
    """
    # Build node_id → set of fact_indices
    node_fact_indices: dict[int, set[int]] = {}
    for i, fact in enumerate(facts):
        for nid in fact.node_ids:
            node_fact_indices.setdefault(nid, set()).add(i)

    if not node_fact_indices:
        logger.info('entity enricher: no node→fact mappings')
        return {}

    result: dict[int, list[dict]] = {}

    for node in nodes:
        nid = node.id
        if nid is None or nid not in node_fact_indices:
            continue

        fact_indices = node_fact_indices[nid]
        node_triplets = [
            t for t in triplets if t.fact_index in fact_indices
        ]
        if not node_triplets:
            continue

        # Distinct subject/object strings from local triplets
        entity_names: set[str] = set()
        for t in node_triplets:
            entity_names.add(t.subject)
            entity_names.add(t.object)

        if not entity_names:
            continue

        # Build entity list for the LLM
        entity_list: list[dict] = []
        for name in entity_names:
            ent_triplets = [
                _triplet_str(t)
                for t in node_triplets
                if t.subject == name or t.object == name
            ]
            entity_list.append({
                'name': name,
                'triplets': ent_triplets,
            })

        # Context window
        before = walker.content_before(
            nodes, nid, BACKWARD_CONTEXT_BUDGET
        )
        after = walker.content_after(
            nodes, nid, FORWARD_CONTEXT_BUDGET
        )

        if not node.content or not node.content.strip():
            continue

        descriptions = await module.aforward(
            node_content=node.content,
            context_before=before,
            context_after=after,
            entities=entity_list,
        )

        result[nid] = [
            {'name': desc.name, 'description': desc.description}
            for desc in descriptions
        ]

    total = sum(len(v) for v in result.values())
    logger.info(
        'entity enricher: %d descriptions across %d nodes',
        total,
        len(result),
    )
    return result


# ============================================================================
# LangGraph node
# ============================================================================


class EntityEnricherNode:
    """Enriches entities with descriptions from the provenance stream.

    Runs after triplet extraction, before the ingestion persister. Reads
    the ``nodes``, ``atomic_facts``, and ``triplets`` channels; writes the
    ``node_entity_descriptions`` channel — a per-node mapping of entity
    names to descriptions.

    Args:
        module: The entity enricher.
    """

    def __init__(self, module: EntityEnricher) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Enrich entity descriptions.

        Args:
            state: The pipeline state.

        Returns:
            The ``node_entity_descriptions`` channel.
        """
        triplets = state.get('triplets', [])
        facts = state.get('atomic_facts', [])
        nodes = state.get('nodes', [])
        descriptions = await enrich_entities(
            triplets=triplets,
            facts=facts,
            nodes=nodes,
            module=self.module,
        )
        return {'node_entity_descriptions': descriptions}
