"""Extract source-level facts and relational evidence for later abstraction.

The fact and triplet passes preserve what a document explicitly says. They do
not write canonical definitions or invent general knowledge. Canonical hubs
later abstract repeated source-level evidence into reusable concepts, relations,
and facts.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import (
    context_window,
    identity,
    llm,
    logs,
    models,
    module,
    state,
)

logger = logging.getLogger(__name__)
class FactNodeInput(BaseModel):
    """Text-only local input for fact extraction."""

    local_index: int = Field(
        description='Zero-based position in the supplied node list.'
    )
    node_type: str = Field(
        description='Canonical node type for the projected source node.'
    )
    node_text: str = Field(
        description=(
            'Canonical node text; numbers here are content, not positions.'
        )
    )




class _FactSignature(dspy.Signature):
    r"""
    Extract explicit, durable facts from the node marked ``<anchor>``.
    Nearby nodes are context only: use them to resolve references, but never
    extract facts from them or assign them provenance.

    A fact is one independent claim or relation. Split combined claims. Each
    result must be a short, complete, standalone sentence with its subject,
    conditions, qualifiers, and resolved pronouns included. Preserve explicit
    premises, including premises inside exercises, but omit task instructions,
    inferred answers, and calculations.

    Do not invent, generalize, duplicate, or solve anything. Skip headers,
    captions, bibliography entries, navigation, document meta-text, rhetorical
    framing, and scratch work. Preserve mathematical and technical notation
    with the source's LaTeX delimiters (``$...$`` or ``$$...$$``).

    Return [] when the anchor contains no explicit subject-matter fact.
    """

    context_before: list[FactNodeInput] = dspy.InputField(
        description='Ordered preceding context records; reference only.'
    )
    target_node: FactNodeInput = dspy.InputField(
        description=(
            'The only evidence record. Extract facts only from target_node; '
            'image descriptions appear as node_text.'
        )
    )
    context_after: list[FactNodeInput] = dspy.InputField(
        description='Ordered following context records; reference only.'
    )
    facts: list[str] = dspy.OutputField(
        description=(
            'Every atomic fact found in the anchor; each must be a short, '
            'self-contained rendering of one explicit source-level claim or '
            'relation. Preserve the fact; do not infer an answer, canonical '
            'definition, or generalization. Empty if none.'
        )
    )


def _fact_input(
    node: context_window.ContextNode, local_index: int | None = None
) -> FactNodeInput:
    """Projects one context node without exposing assets or source identity."""
    return FactNodeInput(
        local_index=node.position if local_index is None else local_index,
        node_type=node.type or '',
        node_text=node.content or '',
    )


class _FactExtractor(module.Module):
    """Decomposes a node window into standalone atomic facts."""

    signature = _FactSignature
    record_name = 'atomic_fact_extractor'

    def encode(
        self,
        context_before: list[context_window.ContextNode],
        target_node: context_window.ContextNode,
        context_after: list[context_window.ContextNode],
    ) -> dict[str, object]:
        """Projects independent document-order context fields."""
        return {
            'context_before': [
                _fact_input(node, index)
                for index, node in enumerate(context_before)
            ],
            'target_node': _fact_input(target_node, 0),
            'context_after': [
                _fact_input(node, index)
                for index, node in enumerate(context_after)
            ],
        }

    def decode(self, prediction, **inputs) -> list[dict]:
        """Returns validated fact text; provenance is assigned by caller."""
        facts = module.as_list(prediction.facts)
        for index, fact in enumerate(facts):
            if not isinstance(fact, str):
                raise TypeError('facts must contain string values')
            if not fact.strip():
                raise ValueError(f'facts[{index}] must be non-empty')
        return [{'text': fact} for fact in facts]


class _TripletInput(BaseModel):
    """One source-level subject/predicate/object decomposition of a fact."""

    subject: str = Field(
        description=(
            'The subject of the relation — an exact verbatim substring '
            'of the fact text. When the subject is a local variable '
            '(a single letter or symbol with no inherent meaning outside '
            'the fact), append a brief parenthetical role: '
            '"$f$ (a function)", "$G$ (a graph)", "$c$ (a point)". '
            r'For named entities with inherent meaning '
            r'("$\mathbb{R}$", "the derivative of $\sin x$"), '
            'no annotation is needed.'
        )
    )
    predicate: str = Field(
        description=(
            'The relation connecting subject to object — a short verb '
            'phrase, typically a verb or verb+preposition (e.g. "is", '
            '"has", "equals", "implies", "is defined as", "is a property '
            'of"). Not a full sentence.'
        )
    )
    object: str = Field(
        description=(
            'The object of the relation — an exact verbatim substring '
            'of the fact text. Same annotation rule as subject: append '
            'a brief parenthetical role for local variables '
            '"$c$ (a point)", "$G_1$ (a graph)", "$E_1$ (an edge set)", '
            '"$[0,1]$ (an interval)"), '
            'omit for named entities. The annotation must be '
            'CONSISTENT with how the same variable is annotated '
            'when it appears as subject elsewhere in the fact.'
        )
    )


class _TripletSignature(dspy.Signature):
    r"""
    Given one atomic source fact, extract every explicit, independent
    (subject, predicate, object) relation it expresses.

    Use source evidence only. Do not infer, solve, generalize, merge
    mentions, or invent entities. A fact may yield several triplets, one per
    independent relation. If no complete relation is decomposable, return [].

    SUBJECT AND OBJECT:
    - Select source substrings; do not rephrase or change their meaning.
    - Lowercase text outside `$...$` math delimiters. Preserve math content
      and all LaTeX delimiters exactly.
    - For a bare local variable, append one short role annotation such as
      `$f$ (a function)` or `$G$ (a graph)`. Use the same annotation for that
      variable throughout the fact.
    - Use complete noun phrases, not dangling fragments, pronouns, generic
      placeholders, or invented referents.

    PREDICATE:
    - Use a short verb or verb phrase, not a clause or sentence.
    - Keep negation in the predicate: "X is NOT a subgraph of Y" has
      predicate "is NOT a subgraph of".
    - Split independent relations and emit each distinct relation once.

    SPEECH ACTS AND FILTERING:
    - Extract relations explicitly asserted, asked about, or framed by
      "prove that"; do not infer the requested answer.
    - Do not emit a value request ("what is ...?", "find ...", "compute ...")
      as a relation. If its only relation is in an if/assuming/given premise,
      omit that premise relation.
    - Skip relations found only in reason clauses, notation conventions, or
      document meta-text.

    OUTPUT CONTRACT:
    - Return a finite list and close it after every relation in the fact has
      been represented.
    - Emit only complete triplets with non-empty subject, predicate, and
      object. Never emit a partial triplet, empty field, or placeholder.
    - Return [] when no complete relation remains.
    """

    fact_text: str = dspy.InputField(
        description=(
            'One source-level atomic fact — a single self-contained '
            'sentence. Preserve its explicit relation; do not generalize it '
            'into a canonical hub fact.'
        )
    )
    triplets: list[_TripletInput] = dspy.OutputField(
        description='Every (subject, predicate, object) triplet found in '
        'the fact; empty if none.'
    )


class _TripletDecomposer(module.Module):
    """Decomposes one source-level fact into relational evidence triplets."""

    signature = _TripletSignature
    record_name = 'triplet_extractor'

    def encode(self, fact_text: str) -> dict:
        """Builds the triplet-signature kwargs for one fact."""
        return {'fact_text': fact_text}

    def decode(self, prediction, **inputs) -> list[models.Triplet]:
        """Returns validated triplets extracted from one fact."""
        triplets = module.as_list(prediction.triplets)
        for index, triplet in enumerate(triplets):
            if not isinstance(triplet, _TripletInput):
                raise TypeError('triplets must contain _TripletInput values')
            if not all(
                value.strip()
                for value in (
                    triplet.subject,
                    triplet.predicate,
                    triplet.object,
                )
            ):
                raise ValueError(
                    f'triplets[{index}] must have non-empty fields'
                )
        return [
            models.Triplet(
                subject=triplet.subject,
                predicate=triplet.predicate,
                object=triplet.object,
            )
            for triplet in triplets
        ]


async def _extract_triplets(
    nodes: list[models.SourceNode],
    fact_module: _FactExtractor,
    triplet_module: _TripletDecomposer,
    max_concurrency: int | None = None,
    source: str | None = None,
) -> list[models.Triplet]:
    """Extracts triplets from deterministic source-node anchors.

    Each eligible source node is sent as the sole fact-extraction anchor.
    Neighboring text is context only. Facts receive their anchor's stable id
    in code before triplets are decomposed.

    Args:
        nodes: The node stream.
        fact_module: The atomic-fact extractor.
        triplet_module: The triplet decomposer.
        max_concurrency: Cap on concurrent LLM calls.

    Returns:
        The extracted triplets.
    """
    if not nodes:
        logger.info('triplet extraction: no nodes')
        return []
    source = source or 'unscoped'
    triplet = config.get_settings().stages.triplet
    eligible_indices = [
        index
        for index, node in enumerate(nodes)
        if node.type == 'image'
        or (node.content is not None and node.content.strip())
    ]
    if not eligible_indices:
        logger.info('triplet extraction: no eligible anchors')
        return []

    gate = llm.gate(max_concurrency)

    async def _extract_one_anchor(node_index: int) -> list[dict]:
        """Extracts facts from one node and assigns deterministic provenance."""
        target_node = context_window.project_nodes([nodes[node_index]])[0]
        context_before = context_window.project_nodes(
            context_window.nodes_before(
                nodes, node_index, triplet.backward_context_budget
            )
        )
        context_after = context_window.project_nodes(
            context_window.nodes_after(
                nodes, node_index, triplet.forward_context_budget
            )
        )
        async with gate:
            facts = await fact_module.aforward(
                context_before=context_before,
                target_node=target_node,
                context_after=context_after,
            )
        return [
            {'text': fact['text'], 'node_positions': [node_index]}
            for fact in facts
        ]

    per_anchor = await asyncio.gather(
        *(_extract_one_anchor(index) for index in eligible_indices)
    )
    facts = [fact for anchor_facts in per_anchor for fact in anchor_facts]

    logger.info(
        'triplet extraction: %d node(s) -> %d fact(s)',
        len(nodes),
        len(facts),
    )

    if not facts:
        return []

    async def _decompose_one(fact: dict) -> list[models.Triplet]:
        """Decomposes one fact and attaches its evidence node positions."""
        logger.debug(
            'triplet extraction: decomposing fact at node position(s) %s: %s',
            fact['node_positions'],
            logs.elide(fact['text']),
        )
        try:
            async with gate:
                triplets = await triplet_module.aforward(fact_text=fact['text'])
        except Exception:
            logger.exception(
                'triplet extraction: triplet decode failed at node position(s) '
                '%s: %s',
                fact['node_positions'],
                logs.elide(fact['text']),
            )
            raise
        for triplet in triplets:
            triplet.evidence_positions = list(fact['node_positions'])
        return triplets

    per_fact = await asyncio.gather(*(_decompose_one(fact) for fact in facts))
    triplets = [
        triplet for fact_triplets in per_fact for triplet in fact_triplets
    ]

    identity.assign_triplet_ids(triplets, source)
    logger.info(
        'triplet extraction: %d fact(s) -> %d triplet(s)',
        len(facts),
        len(triplets),
    )
    return triplets


class TripletNode:
    """Extracts knowledge triplets from document nodes.

    Internally this runs two LLM passes:
    1. Atomic fact extraction — nodes → standalone sentences
    2. Triplet decomposition — each fact → (subject, predicate, object)
    """

    def __init__(
        self,
        fact_module: _FactExtractor,
        triplet_module: _TripletDecomposer,
    ) -> None:
        self._fact_module = fact_module
        self._triplet_module = triplet_module

    async def run(self, state: state.State) -> dict:
        """Extracts triplets from the state's nodes.

        Args:
            state: Pipeline state with nodes.

        Returns:
            A ``triplets`` update for the state.
        """
        nodes = state.get('nodes', [])
        source = state.get('source_key', '').strip()
        if not source:
            source = models.source_key(state.get('source')) or ''
        triplets = await _extract_triplets(
            nodes,
            fact_module=self._fact_module,
            triplet_module=self._triplet_module,
            source=source,
        )
        return {'triplets': triplets}
