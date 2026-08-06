r"""
Triplet extraction — one LangGraph node, one DSPy module.

The second semantic pass: it reads each atomic fact and decomposes it into
one or more (subject, predicate, object) triplets — the raw relation
inventory for the downstream entity canonicalization and relation
canonicalization passes.

Design commitments:

* ONE FACT AT A TIME. Each fact is already a self-contained standalone
  sentence (by the atomic fact pass's contract), so no surrounding context
  window is needed. Every fact is processed in isolation and concurrently
  with every other fact.

* VERBATIM SUBJECT/OBJECT. Subject and object are exact substrings lifted
  from the fact text — no normalization, no pronoun resolution, no
  abstraction. The fact already resolves referents (the atomic fact
  pass's STANDALONE rule), so the subject and object are concrete noun
  phrases already present in the text. Canonicalization into entities
  happens in the next pass.

* DOMAIN-AGNOSTIC, NO RELATION TAXONOMY. The pass targets relations
  generally: any document, any subject. The prompt does not enumerate
  relation kinds (is-a / has-property / causes / …) and carries no
  ontology vocabulary. The only criteria are that subject and object are
  verbatim substrings and the predicate is a short relation phrase.

* 1..N TRIPLETS PER FACT. A fact may contain multiple relations —
  compound assertions, conjoined properties, multi-entity claims. Each
  independent (subject, predicate, object) relationship is one triplet.
  A fact that asserts nothing decomposable into subject-predicate-object
  form (e.g., a bare existential statement) legitimately yields zero
  triplets.

* MINIMAL OUTPUT. ``models.Triplet`` carries only ``subject``,
  ``predicate``, ``object``, and ``fact_index`` — no relation kind, no
  confidence. Classification is a downstream pass's job; provenance is
  the fact index set by the entry point.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, models, recording, state

logger = logging.getLogger(__name__)


class DSPyTriplet(BaseModel):
    """One (subject, predicate, object) triplet emitted by the extractor."""

    subject: str = Field(
        description=(
            'The subject of the relation — an exact verbatim substring '
            'of the fact text. A concrete noun phrase already present in '
            'the source; never normalized, never invented.'
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
            'The object of the relation — an exact verbatim substring of '
            'the fact text. A concrete noun phrase, value, or formula '
            'already present in the source; never normalized, never '
            'invented.'
        )
    )


class Signature(dspy.Signature):
    r"""
    You are given one ATOMIC FACT — a single, self-contained sentence
    conveying exactly one piece of information. Decompose it into
    (subject, predicate, object) TRIPLETS.

    A TRIPLET is one relational assertion: a subject, a predicate that
    connects it to an object, and the object. Every triplet is ONE
    independent relationship. A fact may yield one triplet or several —
    compound assertions that state multiple relationships yield one
    triplet per relationship.

    SUBJECT AND OBJECT must be EXACT VERBATIM SUBSTRINGS of the fact
    text — lift them from the source, do not normalize or rephrase them.
    If the fact says "$f$ is continuous on $[0,1]$", the subject is "$f$"
    and the object is "continuous on $[0,1]$" — exactly as they appear.

    PREDICATE is a short verb phrase (a verb or verb+preposition) that
    captures the relation: "is", "has", "equals", "is a subset of",
    "implies", "is defined as", etc. It is NOT a full sentence.

    EXAMPLES.

    Fact: "The discriminant of $ax^2 + bx + c = 0$ is $b^2 - 4ac$."
    Triplets:
        1. subject="The discriminant of $ax^2 + bx + c = 0$"
           predicate="is"
           object="$b^2 - 4ac$"

    Fact: "A function $f$ is continuous at $c$ if $\lim_{x\to c} f(x) = f(c)$."
    Triplets:
        1. subject="$f$"
           predicate="is continuous at"
           object="$c$"
           (when $\lim_{x\to c} f(x) = f(c)$ — the condition is part of the
           definition, captured as a separate triplet:)
        2. subject="$\lim_{x\to c} f(x)$"
           predicate="equals"
           object="$f(c)$"

    Fact: "The set $\mathbb{R}$ is uncountable and has cardinality $2^{\aleph_0}$."
    Triplets:
        1. subject="$\mathbb{R}$"
           predicate="is"
           object="uncountable"
        2. subject="$\mathbb{R}$"
           predicate="has cardinality"
           object="$2^{\aleph_0}$"

    Fact: "Prove that every continuous function on $[0,1]$ is bounded."
    Triplets:
        1. subject="every continuous function on $[0,1]$"
           predicate="is"
           object="bounded"
    (The "Prove that" wrapper is an instruction framing — extract the
    underlying assertion.)

    Fact: "Is the sequence $\{3n\}_{n=1}^{\infty}$ bounded?"
    Triplets:
        1. subject="the sequence $\{3n\}_{n=1}^{\infty}$"
           predicate="is"
           object="bounded"
    (A question is interrogating a relation — extract that relation as
    if it were asserted. "Is X Y?" yields (X, is, Y). The question
    mark is part of the source wording, not part of the subject or
    object.)

    Fact: "Is the sequence $\{n\}_{n=1}^{\infty}$ convergent, and if so,
    what is its limit?"
    Triplets:
        1. subject="the sequence $\{n\}_{n=1}^{\infty}$"
           predicate="is"
           object="convergent"
    (The follow-up "and if so, what is its limit?" is a request for a
    value, not a relation — it yields no separate triplet.)

    Fact: "If the sequence $\{n\}_{n=1}^{\infty}$ is convergent,
    what is its limit?"
    Triplets:
        (none)
    (The fact's primary speech act is a value request — "what is its
    limit?" — not a relation. The "if X is convergent" clause is a
    premise taken as given, not what is being interrogated. Do not
    extract relations that appear only inside a conditional premise
    when the main question is a value request.)

    Fact: "The graph is not connected because there is no path from
    $a$ to $b$."
    Triplets:
        1. subject="The graph"
           predicate="is not"
           object="connected"
    (The second clause — "there is no path from $a$ to $b$" — is the
    REASON, not an independent relation. Do NOT extract "there" as a
    subject. The existential "there is no X" is a way of saying
    something does not exist, not a subject-predicate-object triple.
    Note: "is not" is the whole predicate here because "connected" is a
    predicate adjective. Contrast with "$G_4$ is NOT a subgraph of
    $G_1$" where the predicate is "is NOT a subgraph of" — the negation
    stays attached to the full verb phrase.)

    FACT: "The Bridges of Königsberg graph had double edges because
    there really are two bridges connecting a particular island to the
    near shore."
    Triplets:
        1. subject="The Bridges of Königsberg graph"
           predicate="had"
           object="double edges"
    (The "because" clause explains why — it is not an independent
    relation. Do not extract triplets from reason clauses that merely
    narrate background.)

    RULES:
    - VERBATIM ONLY. Subject and object must be exact substrings of the
      fact text. Never rephrase, never normalize, never invent a term
      not present in the source.
    - ONE RELATION PER TRIPLET. A fact that asserts two independent
      relationships yields two triplets. Apply the SPLIT TEST: if the
      predicate connects the subject to two objects with different
      relations, those are two triplets.
    - SHORT PREDICATE. The predicate is a verb or a short verb phrase —
      not a clause, not a sentence. It captures the relation type, not
      the full assertion.
    - COVER EVERY RELATION. Extract every (subject, predicate, object)
      relationship the fact expresses, whether it asserts it, asks
      about it, or instructs the reader about it. "Prove that X is Y"
      and "Is X Y?" both express the relation (X, is, Y) — extract it.
      The framing (imperative / interrogative / declarative) does not
      change the relation.
    - CONDITIONAL PREMISE EXCEPTION. When the fact's primary speech act
      is a value request ("what is …?", "find …", "compute …") and a
      relation appears only inside an "if" / "assuming" / "given"
      premise clause, do NOT extract that premise relation — it is a
      condition taken as given, not what the fact interrogates.
    - STANDALONE SUBJECT/OBJECT. The subject and object should each be a
      complete noun phrase that names what it is — not a dangling
      modifier, not a bare symbol with no referent.
    - EXISTENTIAL "THERE" IS A DUMMY SUBJECT. "There is no X", "there
      are Y", "there exists Z" are existential constructions — the
      real content is that X does not exist, Y are present, or Z
      exists. Do NOT extract "there" as a subject. Such clauses
      typically yield zero triplets (they don't assert a subject-
      predicate-object relation between entities).
    - REASON CLAUSES ARE NOT RELATIONS. A "because" clause explains
      why something is true — it is not an independent relation to
      extract as a separate triplet. Extract the main assertion;
      leave the reason clause alone unless it contains a distinct
      relation between named entities.
    - NEGATION TRAVELS WITH THE PREDICATE. "X is NOT a subgraph of Y"
      yields one triplet with predicate="is NOT a subgraph of" — keep
      the negation attached to the verb phrase. Do NOT split negation
      into a bare "is not" predicate with the rest of the verb phrase
      pushed into the object.
    - PROPERTY-ASCRIPTION CHECK. When a fact's structure is "X has the
      property that [long clause]" or "X have the property that [long
      clause]" — where the object is a clause describing a property
      rather than a named entity — the result is not a clean
      subject-predicate-object relationship. Skip such triplets unless
      the object can be stated as a concrete noun phrase.
    - ABSTRACT / GENERIC SUBJECTS. "Such graphs", "the resulting
      graph", "this function" — when the subject is a placeholder
      whose identity depends on the preceding sentence, it is better
      to leave the triplet out than to create an entity that will
      never be reused. If the subject cannot be stated as a concrete,
      independent noun phrase, skip the triplet.
    - PASSIVE NAMING CONSTRUCTIONS. "X are called Y", "X is known as
      Y", "we call X Y" — the subject is the named thing (X), the
      predicate is "is called" or "are called", and the object is the
      name (Y). Do not extract the naming verb as a separate relation
      or the name as a dangling entity.
    - LATEX FORMAT. Preserve LaTeX delimiters exactly as in the fact:
      `$...$` for inline, `$$...$$` for display. Never convert to
      Unicode, never strip delimiters.
    - NO DUPLICATES. Do not emit the same triplet twice.
    - Return an empty list if the fact contains no decomposable
      relationships (e.g., a bare existential statement with no
      predicate-object structure).
    """

    fact_text: str = dspy.InputField(
        description='One atomic fact — a single self-contained sentence.'
    )
    triplets: list[DSPyTriplet] = dspy.OutputField(
        description='Every (subject, predicate, object) triplet found in '
        'the fact; empty if none.'
    )


class TripletExtractor(dspy.Module):
    """Extracts (subject, predicate, object) triplets from one atomic fact.

    Args:
        language_model: The LM to run on.
    """

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__()
        self.extractor = dspy.ChainOfThought(Signature)
        self.set_lm(language_model)
        self._recorder = recorder

    async def aforward(self, fact_text: str) -> list[models.Triplet]:
        """Extract triplets from one atomic fact.

        Args:
            fact_text: The atomic fact text.

        Returns:
            The triplets found, or an empty list.
        """
        result = await self.extractor.acall(fact_text=fact_text)
        if self._recorder:
            self._recorder.record(
                'triplet_extractor',
                {'fact_text': fact_text},
                result,
            )
        triplets = [
            models.Triplet(
                subject=triplet.subject,
                predicate=triplet.predicate,
                object=triplet.object,
            )
            for triplet in (result.triplets or [])
        ]
        logger.debug(
            'triplet extractor: fact -> %d triplet(s)',
            len(triplets),
        )
        return triplets

    def forward(self, fact_text: str) -> list[models.Triplet]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(fact_text=fact_text))


# ============================================================================
# Entry point
# ============================================================================


async def extract_triplets(
    facts: list[models.AtomicFact],
    module: TripletExtractor,
    max_concurrency: int | None = None,
) -> list[models.Triplet]:
    """Extract triplets from every atomic fact.

    Each fact is processed independently and concurrently; the triplets
    are collected in document order with their ``fact_index`` set to the
    source fact's position.

    Args:
        facts: The atomic facts, in document order.
        module: The triplet extractor.
        max_concurrency: Facts in flight at once. None uses
            ``llm.MAX_CONCURRENT_CALLS``.

    Returns:
        The triplets, in document order (grouped by source fact).
    """
    if not facts:
        logger.info('triplet extractor: no facts')
        return []

    gate = llm.gate(max_concurrency)

    async def _extract_one(
        fact_index: int, fact: models.AtomicFact
    ) -> list[models.Triplet]:
        async with gate:
            triplets = await module.aforward(fact.text)
            for triplet in triplets:
                triplet.fact_index = fact_index
            return triplets

    per_fact = await asyncio.gather(
        *(_extract_one(i, fact) for i, fact in enumerate(facts))
    )
    triplets = [
        triplet for fact_triplets in per_fact for triplet in fact_triplets
    ]

    logger.info(
        'triplet extractor: %d fact(s) -> %d triplet(s)',
        len(facts),
        len(triplets),
    )
    return triplets


# ============================================================================
# LangGraph node
# ============================================================================


class TripletNode:
    """Extracts triplets from the atomic fact list.

    Runs after the atomic fact pass and before the ingestion
    persister. Reads only the ``atomic_facts`` channel; writes the
    ``triplets`` channel.

    Args:
        module: The triplet extractor.
    """

    def __init__(self, module: TripletExtractor) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Extract triplets from the atomic facts.

        Args:
            state: The pipeline state, holding the atomic facts.

        Returns:
            The ``triplets`` channel.
        """
        facts = state.get('atomic_facts', [])
        triplets = await extract_triplets(facts, module=self.module)
        return {'triplets': triplets}
