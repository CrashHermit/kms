"""Two-pass knowledge extraction: atomic facts, then triplets."""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import llm, models, module, state, walker

logger = logging.getLogger(__name__)


class _FactInput(BaseModel):
    """One atomic fact and the window nodes it was drawn from."""

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


class _FactSignature(dspy.Signature):
    r"""
    You are given a run of nodes from a document, in document order. Each
    node carries its id, its structural type, and its content. Decompose the
    window into ATOMIC FACTS.

    AN ATOMIC FACT is the smallest piece of source content worth preserving
    on its own: an explicit claim, property, relationship, event, definition,
    stated result, or factual premise. It conveys exactly ONE such unit as a
    complete standalone sentence. This is domain-neutral: the document may be
    about anything. Do not infer answers, solve exercises, or classify facts
    into kinds. Just preserve explicit subject-matter assertions.

    THE ATOMICITY TEST. A fact is atomic when it conveys exactly one unit
    of information — normally one assertion, or a source question that
    expresses a relation — with no second independent unit joined on. Apply
    the SPLIT TEST before emitting: if you can break the fact at a conjunction
    or a comma into two pieces that would each still be true of the source, it
    is not atomic — split it into those pieces. A task request such as
    "simplify this expression" is not an atomic fact merely because it is an
    instruction. When in doubt, split explicit subject-matter content.

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
    - In a word problem, separate the EXPLICIT GIVEN premises from the task.
      For "The highest recorded temperature was 57 degrees Celsius. Use
      integers to write the temperature", emit the premise "The highest
      recorded temperature was 57 degrees Celsius" and do NOT emit the
      instruction or infer an answer such as "$+57$".
    - For "Pennsylvania estimated a budget surplus of $540 million", emit that
      explicit premise as a fact even if it appears inside an exercise. For
      "Community college enrollment grew by 1,400,000 students", emit the
      stated growth. Exercise framing is transient; factual premises stated
      inside it are not automatically transient.

    RULES:
    - ONE UNIT PER FACT. One explicit assertion or one relation-bearing
      source question per fact. A sentence that makes two independent claims
      yields two facts; a passage that asserts several things yields one fact
      per assertion. Do not emit an exercise instruction as a fact.
    - STANDALONE, NOT FRAGMENTED. State every fact as a complete sentence
      that names its own subject and carries its own conditions and
      qualifiers. Resolve every "it", "this", "the former" into its
      referent. Never emit a fragment ("since $a \neq 0$", "which is
      continuous", "as above").
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
    - DURABLE, NOT TRANSITIONAL. Emit explicit subject-matter assertions and
      factual premises, including premises embedded in word problems. Do not
      emit navigation ("in this section", "as we will see"), rhetorical
      framing, formatting, an imperative task request ("simplify ...",
      "use integers to write ..."), a bare symbolic exercise, or scratch lines
      of a worked manipulation. A request to calculate is not itself a fact;
      the data stated in that request may still be facts.
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
    - FIND EVERYTHING EXPLICIT. A missed stated premise is a lost fact, but
      never invent the answer to an exercise or promote an implied result to a
      source fact. When unsure whether an explicit premise is merely exercise
      framing or subject-matter content, preserve the content and omit only
      the instruction wrapper.
    - Return an empty list if the window contains no explicit facts.
    """

    current_nodes: list[walker.WindowNode] = dspy.InputField(
        description=(
            "The window's nodes, in document order, each with its id, type, "
            'and content. Use the `id` (not `position`) when attributing '
            'facts.'
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
    facts: list[_FactInput] = dspy.OutputField(
        description='Every atomic fact found in the window; empty if none.'
    )


class _FactExtractor(module.Module):
    """Decomposes a node window into standalone atomic facts."""

    signature = _FactSignature
    record_name = 'atomic_fact_extractor'

    def encode(
        self,
        current_nodes: list[walker.WindowNode],
        context_before: str | None = None,
        context_after: str | None = None,
    ) -> dict:
        """Builds the fact-signature kwargs for one window."""
        return {
            'current_nodes': current_nodes,
            'context_before': context_before or '',
            'context_after': context_after or '',
        }

    def decode(self, prediction, **inputs) -> list[dict]:
        """Returns the atomic facts as text/node_id dicts."""
        return [
            {'text': fact.text, 'node_ids': module.as_list(fact.node_ids)}
            for fact in module.as_list(prediction.facts)
        ]


class _TripletInput(BaseModel):
    """One subject/predicate/object decomposition of a fact."""

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

    CASE NORMALIZATION. After lifting the verbatim text, lowercase
    everything EXCEPT content inside $...$ LaTeX math delimiters.
    Characters between $...$ retain their original case.  "Graph"
    becomes "graph", "Edge Set" becomes "edge set", but "$G$" stays
    "$G$" and "$V'$" stays "$V'$".  Apply this to both subject and
    object independently.

    LOCAL-VARIABLE ANNOTATION. When the subject or object is a bare
    variable — a single letter or symbol that serves only as a local
    placeholder ("$f$", "$G$", "$c$", "$a$", "$x$", "$E_1$", "$E_3$") —
    append a brief parenthetical role drawn from how the fact uses it.
    This rule applies to BOTH subject AND object equally: "$G_1$ (a
    graph)", "$E_1$ (an edge set)", "$c$ (a point)". A variable that
    appears in one triplet as subject and in another as object must
    carry the SAME annotation in both roles — decide the annotation
    from how the fact introduces it. Named entities that carry
    inherent meaning — "$\mathbb{R}$", "the derivative of $\sin x$",
    "the discriminant", "every continuous function on $[0,1]$" — need
    no annotation. The annotation is ONE short word or phrase, not a
    description.

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
        1. subject="$f$ (a function)"
           predicate="is continuous at"
           object="$c$ (a point)"
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

    Fact: "The highest recorded temperature on Earth was 57 degrees Celsius."
    Triplets:
        1. subject="The highest recorded temperature on Earth"
           predicate="was"
           object="57 degrees Celsius"
    (This is an explicit premise from a word problem. Extract the stated
    relation; do not infer or normalize the requested answer.)

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

    Fact: "The graph $G_4$ is NOT a subgraph of $G_1$, even though it looks
    like all we did is remove vertex $e$."
    Triplets:
        1. subject="$G_4$ (a graph)"
           predicate="is NOT a subgraph of"
           object="$G_1$ (a graph)"
    (The second clause — "even though it looks like all we did is remove
    vertex $e$" — is the REASON, not an independent relation. Do NOT
    extract triplets from reason clauses that merely narrate background.)

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
      fact text — never rephrase, never normalize, never invent a term
      not present in the source. The parenthetical annotation for local
      variables (see above) is the ONLY text you may add beyond the
      verbatim substring.
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
      change the relation. Explicit factual premises embedded in an exercise
      are ordinary assertions and must be decomposed; the exercise's answer
      request is not an additional relation.
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
    - NOTATION CONVENTIONS. "X are denoted by Y", "we write X for Y",
      "we use X to represent Y" — these are statements about notation,
      not about the subject matter. Skip them — they do not assert a
      relationship between entities in the domain.
    - META-DISCOURSE. "We will first consider...", "this book studies",
      "the remainder of this section is organized as follows" — these
      are statements about the book's structure, not about its content.
      Skip them.
    - TRIVIAL DEFINITIONS OF NOTATION. When a fact essentially says
      "we use the symbol X to mean Y", it is a notation convention —
      skip it. The triplet should capture the meaning, not the naming.
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
    triplets: list[_TripletInput] = dspy.OutputField(
        description='Every (subject, predicate, object) triplet found in '
        'the fact; empty if none.'
    )


class _TripletDecomposer(module.Module):
    """Decomposes one atomic fact into knowledge triplets."""

    signature = _TripletSignature
    record_name = 'triplet_extractor'

    def encode(self, fact_text: str) -> dict:
        """Builds the triplet-signature kwargs for one fact."""
        return {'fact_text': fact_text}

    def decode(self, prediction, **inputs) -> list[models.Triplet]:
        """Returns the triplets extracted from one fact."""
        return [
            models.Triplet(
                subject=triplet.subject,
                predicate=triplet.predicate,
                object=triplet.object,
            )
            for triplet in module.as_list(prediction.triplets)
        ]


async def _extract_triplets(
    nodes: list[models.ASTNode],
    fact_module: _FactExtractor,
    triplet_module: _TripletDecomposer,
    max_concurrency: int | None = None,
) -> list[models.Triplet]:
    """Extracts triplets from the node stream via fact windows.

    Walks fixed windows with context, extracts atomic facts from each,
    then decomposes every fact into triplets with the evidence node ids
    attached.

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

    triplet = config.get_settings().stages.triplet
    windows = walker.fixed_windows_with_context(
        nodes,
        triplet.window_budget,
        triplet.backward_context_budget,
        triplet.forward_context_budget,
    )
    if not windows:
        logger.info('triplet extraction: no windows')
        return []

    gate = llm.gate(max_concurrency)

    async def _extract_one_window(window: walker.Window) -> list[dict]:
        """Extracts the atomic facts of one window."""
        async with gate:
            return await fact_module.aforward(
                current_nodes=window.items,
                context_before=window.before,
                context_after=window.after,
            )

    per_window = await asyncio.gather(
        *(_extract_one_window(window) for window in windows)
    )
    facts = [fact for window_facts in per_window for fact in window_facts]

    logger.info(
        'triplet extraction: %d node(s) -> %d fact(s)',
        len(nodes),
        len(facts),
    )

    if not facts:
        return []

    async def _decompose_one(fact: dict) -> list[models.Triplet]:
        """Decomposes one fact and attaches its evidence node ids."""
        async with gate:
            triplets = await triplet_module.aforward(fact_text=fact['text'])
            for triplet in triplets:
                triplet.node_ids = list(fact['node_ids'])
            return triplets

    per_fact = await asyncio.gather(*(_decompose_one(fact) for fact in facts))
    triplets = [
        triplet for fact_triplets in per_fact for triplet in fact_triplets
    ]

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
        triplets = await _extract_triplets(
            nodes,
            fact_module=self._fact_module,
            triplet_module=self._triplet_module,
        )
        return {'triplets': triplets}
