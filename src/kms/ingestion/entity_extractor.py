r"""
Entity mention extraction — one LangGraph node, one DSPy module.

The second semantic pass over the pipeline: it reads the ``atomic_facts``
channel and extracts entity names from each fact. One DSPy call per batch of
facts (fixed window budget, same pattern as the atomic fact pass). No router
— entities are dense, and a false negative silently deletes an entity from
the graph.

Design commitments:

* NAMES ONLY. The extractor emits entity names (short phrases, 1-6 words) —
  no descriptions. The canonicalization pass later enriches each entity
  cluster with a description drawn from the full set of facts it appears in,
  where the LLM can see every mention at once.

* PER-FACT PROVENANCE. Each mention carries a ``fact_index`` — the position
  of the source fact in the ``atomic_facts`` channel — so the downstream
  canonicalization pass can map mentions back to their facts, and the
  relation pass can resolve entities to the facts they co-occur in.

* BATCHED, PARALLEL. Facts are cut into batches by character budget
  (~4 chars/token) and each batch is one DSPy call. Batches run concurrently
  through the shared ``llm.gate`` — the same pattern as the atomic fact
  extractor.

* LATEX FORMAT. Entity names keep delimited LaTeX (``$...$``, ``$$...$$``)
  exactly as in the source fact. A fact that mentions ``$x = -b \\pm
  \\sqrt{b^2-4ac} / 2a$`` as an entity keeps that string verbatim.
"""

import asyncio
import logging

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, models, state
from kms.core.recorder import Recorder

logger = logging.getLogger(__name__)

# Fixed context window for one batch (~4 chars/token). A single fact larger
# than the budget still forms a batch of its own.
WINDOW_BUDGET = 2000


# ============================================================================
# DSPy models
# ============================================================================


class DSPyEntityMention(BaseModel):
    """One entity mention emitted by the extractor for a single fact."""

    name: str = Field(
        description=(
            'The entity name as a short phrase (1-6 words). Use the exact '
            'wording from the fact — do not normalise or canonicalise. '
            'LaTeX notation with its delimiters stays as-is.'
        )
    )
    fact_index: int = Field(
        description=(
            'Which fact in the batch this mention belongs to — the '
            'index as given in the input facts list.'
        )
    )


class Signature(dspy.Signature):
    r"""
    You are given a batch of atomic facts, each with its index. For each
    fact, extract the entities it mentions.

    An ENTITY is a named thing the fact is about: a concept, an object, a
    person, a quantity, a method, a theorem, a definition, a system —
    anything you could point at and say "this fact is about that." Write
    each entity as a short name phrase (1-6 words) using the wording from
    the fact.

    LATEX FORMAT. When an entity name includes mathematical notation, keep
    it in delimited LaTeX (``$...$`` inline, ``$$...$$`` display) exactly
    as it appears in the fact. Do not strip delimiters and do not convert
    to Unicode.

    RULES:
    - Every entity worth naming gets one mention. If a fact is about the
      quadratic formula AND roots AND quadratic equations, emit all three.
    - Do not invent entities the fact does not mention. Do not skip
      entities the fact does mention.
    - Pronouns ("it", "this", "they") are not entities — use the referent
      noun phrase instead.
    - A fact may mention zero entities — return no mention for it.
    - Keep the name close to the fact's own wording. Do not normalise
      "the quadratic formula" to "quadratic formula" — the canonicalisation
      pass merges surface variants later.
    """

    facts: list[tuple[int, str]] = dspy.InputField(
        description=(
            'The batch of (fact_index, fact_text) pairs. fact_index is an '
            'integer identifying the fact in the full list; fact_text is '
            'the atomic fact text.'
        )
    )
    mentions: list[DSPyEntityMention] = dspy.OutputField(
        description='Every entity mention found across all facts in the batch.'
    )


# ============================================================================
# DSPy module
# ============================================================================


class EntityExtractor(dspy.Module):
    """Extracts entity names from a batch of atomic facts.

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
        self, facts: list[tuple[int, str]]
    ) -> list[tuple[int, str]]:
        """Extract entity names from one batch.

        Args:
            facts: List of (fact_index, fact_text) pairs.

        Returns:
            List of (fact_index, entity_name) pairs — flat, one per
            mention, in the order the LLM emitted them.
        """
        result = await self.extractor.acall(facts=facts)
        if self._recorder:
            self._recorder.record(
                'entity_extractor',
                {'facts': facts},
                result,
            )
        mentions = [(m.fact_index, m.name) for m in (result.mentions or [])]
        logger.debug(
            'entity extractor: %d fact(s) -> %d mention(s)',
            len(facts),
            len(mentions),
        )
        return mentions

    def forward(self, facts: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(self.aforward(facts=facts))


# ============================================================================
# Batch cutting
# ============================================================================


def _batch_facts(
    facts: list[models.AtomicFact], budget: int = WINDOW_BUDGET
) -> list[list[tuple[int, str]]]:
    """Cut the fact list into batches by character budget.

    Each batch is a list of (fact_index, fact_text) pairs. A batch always
    contains at least one fact — a single fact larger than the budget forms
    a batch of its own.

    Args:
        facts: The atomic facts, in document order.
        budget: The fixed character budget per batch.

    Returns:
        One batch per window, each a list of (index, text) pairs.
    """
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_size = 0
    for i, fact in enumerate(facts):
        size = len(fact.text)
        if current and current_size + size > budget:
            batches.append(current)
            current = []
            current_size = 0
        current.append((i, fact.text))
        current_size += size
    if current:
        batches.append(current)
    return batches


# ============================================================================
# Entry point
# ============================================================================


async def extract_entities(
    facts: list[models.AtomicFact],
    module: EntityExtractor,
    max_concurrency: int | None = None,
) -> list[tuple[int, str]]:
    """Extract entity mentions from the full fact list.

    Facts are cut into adjacent batches by character budget; every batch
    is processed concurrently, and the mentions are collected. The pass is
    a pure reader of ``atomic_facts`` — nothing writes back to the fact
    channel.

    Args:
        facts: The atomic facts, in document order.
        module: The entity extractor.
        max_concurrency: Batches in flight at once. None uses
            ``llm.MAX_CONCURRENT_CALLS``.

    Returns:
        List of (fact_index, entity_name) pairs, one per mention.
    """
    batches = _batch_facts(facts)
    if not batches:
        logger.info('entity extractor: no batches')
        return []

    gate = llm.gate(max_concurrency)

    async def _extract_one(
        batch: list[tuple[int, str]],
    ) -> list[tuple[int, str]]:
        async with gate:
            return await module.aforward(batch)

    per_batch = await asyncio.gather(
        *(_extract_one(batch) for batch in batches)
    )
    mentions = [
        mention for batch_result in per_batch for mention in batch_result
    ]

    logger.info(
        'entity extractor: %d fact(s) in %d batch(es) -> %d mention(s)',
        len(facts),
        len(batches),
        len(mentions),
    )
    return mentions


# ============================================================================
# LangGraph node
# ============================================================================


class EntityExtractorNode:
    """Extracts entity mentions from the atomic facts channel.

    Runs after the atomic fact pass and before the canonicalisation pass.
    Reads the ``atomic_facts`` channel; writes the ``entity_mentions``
    channel.

    Args:
        module: The entity extractor.
    """

    def __init__(self, module: EntityExtractor) -> None:
        self.module = module

    async def run(self, state: state.State) -> dict:
        """Extract entity mentions from every atomic fact.

        Args:
            state: The pipeline state, holding the ``atomic_facts``
                channel.

        Returns:
            The ``entity_mentions`` channel.
        """
        facts = state.get('atomic_facts', [])
        if not facts:
            return {}
        mentions = await extract_entities(facts, module=self.module)
        return {'entity_mentions': mentions}
