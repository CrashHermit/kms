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


class _FactSignature(dspy.Signature):
    r"""
    Extract explicit, durable subject-matter facts from the supplied request.
    The target node contains the authoritative source text. Use neighboring
    nodes only to resolve an otherwise explicit reference; never copy a claim
    that appears only in neighboring context.

    A durable fact is an assertion about the subject matter, such as a
    definition, theorem, stated property, explicit relationship, event,
    state transition, or mathematical condition. Preserve the source's
    notation, participants, temporal expressions, causal wording, and
    qualifiers. Return the smallest faithful claim that can stand alone.
    Resolve a pronoun only when its referent is explicit in the target or
    unambiguous from the supplied context; do not add an explanation.

    Preserve explicit actions, occurrences, transitions, participants, time,
    sequence, duration, negation, and causal wording. Keep an event separate
    from its result or consequence when both are stated. Do not infer
    causation. Split independent assertions, but do not split an event from
    its participants or temporal qualifiers.

    Return no fact for headings, captions, references, metadata, navigation,
    layout text, questions, answer requests, exercise numbers, worked-example
    labels, problem specifications, or imperatives and task directions such
    as ``find``, ``compute``, ``determine``, ``show``, ``prove``, ``sketch``,
    ``draw``, ``use technology``, ``evaluate``, or ``let``.

    Do not return inferred answers, calculations, explanations, or claims
    copied only from context. A formula is a fact only when the target
    explicitly asserts it as subject matter, rather than presenting it as
    part of a task. Preserve formulas and mathematical relationships when
    they are explicitly asserted.

    Output one short, natural-language fact for each independent assertion.
    Each fact must express one subject–predicate–object relation. Return an
    empty list when the target contains no qualifying assertion.
    """

    request: models.FactExtractionInput = dspy.InputField(
        description=(
            'The extraction request. Extract only from target_node.text. '
            'context_before and context_after provide reference context only.'
        )
    )
    facts: list[models.AtomicFact] = dspy.OutputField(
        description=(
            'Structured source-faithful natural-language atomic facts. '
            'Return an empty list when the target contains no qualifying '
            'assertion.'
        )
    )


def _fact_input(
    node: context_window.ContextNode, local_index: int | None = None
) -> models.NodeInput:
    """Projects one context node into a one-based structured record."""
    return models.NodeInput(
        index=(local_index + 1)
        if local_index is not None
        else node.position + 1,
        node_type=node.type or '',
        text=node.content or '',
    )


class _FactExtractor(module.Module):
    """Decomposes a node window into standalone atomic facts."""

    signature = _FactSignature
    record_name = 'atomic_fact_extractor'

    def encode(
        self, request: models.FactExtractionInput
    ) -> dict[str, object]:
        """Passes the validated extraction request to the signature."""
        return {'request': request}

    def decode(self, prediction, **inputs) -> list[dict]:
        """Returns validated fact text; provenance is assigned by caller."""
        facts = models.FactExtractionOutput(
            facts=module.as_list(prediction.facts)
        ).facts
        return [{'text': fact.text} for fact in facts]


class _TripletInput(BaseModel):
    """One source-level subject/predicate/object decomposition of a fact."""

    subject: str = Field(
        description=(
            'The subject of the relation — an exact verbatim substring '
            'of the fact text.'
        )
    )
    predicate: str = Field(
        description='A concise source-grounded relation phrase of 1–5 words.'
    )
    object: str = Field(
        description=(
            'The object of the relation — an exact verbatim substring '
            'of the fact text.'
        )
    )
    subject_kind: models.NodeKind = Field(
        description='Whether the subject is an entity or event.'
    )
    object_kind: models.NodeKind = Field(
        description='Whether the object is an entity or event.'
    )


class _TripletSignature(dspy.Signature):
    r"""
    Decompose one atomic source fact into one or more explicit
    subject–predicate–object relations. The fact has already passed a
    source-fact filter. Reject only directives, questions, answer requests,
    unsupported implications, or text that contains no defensible relation.

    ACCEPT THESE AS RELATIONS:
    - definitions: "A group is a set with an operation" →
      group | is defined as | a set with an operation
    - measurements: "Velocity measures how fast an object is moving" →
      velocity | measures | how fast an object is moving
    - mathematical objects and quantities: preserve symbols, formulas,
      intervals, coordinates, and units as written when they are part of an
      explicit assertion;
    - rules and operations stated by the source: "Dividing miles traveled by
      time elapsed gives velocity" → miles traveled divided by time elapsed |
      gives | velocity;
    - possession, membership, location, comparison, and other explicit
      relations, even when the object is an abstract phrase or quantity.
    A mathematical expression inside an assertion is evidence for a relation;
    do not reject it merely because it involves calculation.

    SUBJECT AND OBJECT:
    - Use the shortest complete source phrase that preserves the meaning.
    - Classify a persistent thing, person, place, document, concept, or
      quantity as entity.
    - Classify a named occurrence, action, transition, or state change as
      event.
    - Use entity endpoints for abstract descriptions and quantities. Do not
      require every endpoint to be a simple concrete noun.
    - Use event endpoints only for explicitly named occurrences or changes.
      Never invent an event from tense, chronology, or causality.
    - Subject and object must be different spans unless the source explicitly
      states a reflexive relation.
    - Preserve source mathematical notation and LaTeX exactly.

    PREDICATE:
    - Use one concise relation phrase of 1–5 words.
    - Preserve explicit direction, negation, temporal order, and causality.
    - Never output a sentence, clause, explanation, quotation, or predicate
      containing another subject or object.

    REJECT:
    - imperatives, questions, exercise specifications, answer requests, and
      requested operations: "Find the gradient of $f$ at $P$." → [];
    - a displayed formula or value that is only an exercise input or answer;
    - explanations or consequences not explicitly asserted by the fact;
    - a fact for which no subject, predicate, and distinct object can be
      identified without guessing.

    Examples:
    - "Alice works for Acme." → entity, works for, entity.
    - "A moving object has a velocity at any given moment." → entity, has,
      entity.
    - "The fixed point $(s, T) = (50, 8)$ sits at the center of the window."
      → entity, sits at, entity.
    - "The election preceded the appointment." → event, preceded, event.
    Return [] rather than guessing; use [] only when the fact is not an
    explicit relational assertion.
    """

    fact_text: str = dspy.InputField(
        description=(
            'One filtered source assertion. It must not be an instruction, '
            'question, exercise specification, calculation, or answer request.'
        )
    )
    triplets: list[_TripletInput] = dspy.OutputField(
        description=(
            'Concise source-grounded relations with entity/event endpoint kinds. '
            'Predicates are 1–5-word phrases, never explanatory sentences; empty if none.'
        )
    )


def _is_concise_predicate(predicate: str) -> bool:
    """Accepts only compact relation phrases at the graph boundary."""
    words = predicate.split()
    return 1 <= len(words) <= 5 and not any(
        mark in predicate for mark in '.!?;:'
    )


class _TripletDecomposer(module.Module):
    """Decomposes one source-level fact into relational evidence triplets."""

    signature = _TripletSignature
    record_name = 'triplet_extractor'

    def encode(self, fact_text: str) -> dict:
        """Builds the triplet-signature kwargs for one fact."""
        return {'fact_text': fact_text}

    def decode(self, prediction, **inputs) -> list[models.Triplet]:
        """Returns only complete triplets with compact relation phrases."""
        triplets = module.as_list(prediction.triplets)
        accepted: list[models.Triplet] = []
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
            if not _is_concise_predicate(triplet.predicate):
                logger.warning(
                    'discarding non-concise triplet predicate at index %d: %s',
                    index,
                    logs.elide(triplet.predicate),
                )
                continue
            accepted.append(
                models.Triplet(
                    subject=triplet.subject,
                    predicate=triplet.predicate,
                    object=triplet.object,
                    subject_kind=triplet.subject_kind,
                    object_kind=triplet.object_kind,
                )
            )
        return accepted


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
            request = models.FactExtractionInput(
                context_before=[
                    _fact_input(node, index)
                    for index, node in enumerate(context_before)
                ],
                target_node=_fact_input(target_node, 0),
                context_after=[
                    _fact_input(node, index)
                    for index, node in enumerate(context_after)
                ],
            )
            facts = await fact_module.aforward(request=request)
        return [
            {'text': fact['text'], 'node_positions': [node_index]}
            for fact in facts
        ]

    per_anchor = await asyncio.gather(
        *(_extract_one_anchor(index) for index in eligible_indices)
    )
    facts = [fact for anchor_facts in per_anchor for fact in anchor_facts]

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
        source = state['source'].key
        triplets = await _extract_triplets(
            nodes,
            fact_module=self._fact_module,
            triplet_module=self._triplet_module,
            source=source,
        )
        return {'triplets': triplets}
