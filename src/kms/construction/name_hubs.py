"""Deterministic lexical name-hub materialization."""

import asyncio
import logging
from collections.abc import Callable

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import batching, context_window, llm, module
from kms.graph import names, queries, writer


class _LexicalDefinition(BaseModel):
    """The selected member index for one lexical cluster."""

    canonical_index: int = Field(
        description='Zero-based index of the preferred supplied surface form.'
    )


class _LexicalDefinitionInput(BaseModel):
    """Structured lexical canonical-form input."""

    surface_forms: list[str]
    kind: str


class _LexicalMembership(BaseModel):
    """The boolean decision for one deterministic candidate pair."""

    should_merge: bool


class _LexicalMembershipInput(BaseModel):
    """Structured lexical membership input."""

    left: str
    right: str
    kind: str
class _LexicalDefinitionSignature(dspy.Signature):
    r"""
    Choose one preferred written form for a fixed lexical cluster.

    The cluster has already been determined by deterministic lexical
    similarity. Do not merge, split, or reinterpret the cluster. Do not use
    semantic synonyms, broader concepts, narrower concepts, or related words
    as evidence. Preserve the phrase's token order, negation, mathematical
    notation, and meaningful case. Return one concise form that is actually
    present in the supplied surface forms whenever possible.
    """

    request: _LexicalDefinitionInput = dspy.InputField()
    result: _LexicalDefinition = dspy.OutputField()




class _LexicalMembershipSignature(dspy.Signature):
    r"""
    Decide whether two already-similar complete phrases belong to one
    lexical-form group. Return TRUE only for conservative written-form
    equivalence and FALSE otherwise. Do not judge semantic equivalence.
    """

    request: _LexicalMembershipInput = dspy.InputField()
    result: _LexicalMembership = dspy.OutputField()


class _LexicalMembershipJudge(module.Module):
    """Adjudicates deterministic lexical candidates before naming."""

    signature = _LexicalMembershipSignature
    record_name = 'lexical_membership_judge'

    def encode(self, left: str, right: str, kind: str) -> dict:
        """Build the structured lexical membership input."""
        return {
            'request': _LexicalMembershipInput(
                left=left, right=right, kind=kind
            )
        }

    def decode(self, prediction, **inputs) -> bool:
        """Return the validated lexical merge decision."""
        result = _LexicalMembership.model_validate(prediction.result)
        return result.should_merge


class _LexicalDefinitionSynthesizer(module.Module):
    """Chooses presentation text without deciding lexical membership."""

    signature = _LexicalDefinitionSignature
    record_name = 'lexical_name_synthesizer'

    def encode(self, surface_forms: list[str], kind: str) -> dict:
        """Build the structured canonical-form selector input."""
        return {
            'request': _LexicalDefinitionInput(
                surface_forms=surface_forms, kind=kind
            )
        }

    def decode(self, prediction, **inputs) -> int:
        """Return the validated supplied surface-form index."""
        index = prediction.result.canonical_index
        module.require_positions(
            [index],
            field_name='canonical_index',
            upper_bound=len(inputs['surface_forms']),
        )
        return index


async def _judged_groups(
    rows: list[dict],
    kind: str,
    *,
    language_model: dspy.LM,
    similarity_threshold: float,
    gate: asyncio.Semaphore,
    comparison_token_budget: int = 4096,
) -> list[list[dict]]:
    """Runs lexical judgment only on deterministic similarity candidates."""
    exact_pairs = [
        (left, right)
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
        if rows[left]['normalized_text'] == rows[right]['normalized_text']
    ]
    candidate_pairs = [
        pair
        for pair in names.lexical_candidate_pairs(
            rows,
            threshold=similarity_threshold,
        )
        if pair not in exact_pairs
    ]
    if not candidate_pairs:
        return names.lexical_groups_from_pairs(rows, exact_pairs)

    judge = _LexicalMembershipJudge(language_model)

    async def _judge_pair(pair: tuple[int, int]) -> tuple[int, int] | None:
        left, right = pair
        async with gate:
            decision = await judge.aforward(
                left=rows[left]['text'],
                right=rows[right]['text'],
                kind=kind,
            )
        if not isinstance(decision, bool):
            raise TypeError(
                f'lexical membership judge returned non-boolean decision: '
                f'{decision!r}'
            )
        return pair if decision else None

    def token_cost(pair: tuple[int, int]) -> int:
        left, right = pair
        return (
            context_window.estimate_text_tokens(rows[left]['text'])
            + context_window.estimate_text_tokens(rows[right]['text'])
            + context_window.estimate_text_tokens(kind)
            + 32
        )
    decisions: list[tuple[int, int] | None] = []
    wave_count = 0
    largest_wave_tokens = 0
    for wave in batching.token_batches(
        candidate_pairs,
        token_cost=token_cost,
        token_budget=comparison_token_budget,
    ):
        wave_count += 1
        largest_wave_tokens = max(
            largest_wave_tokens,
            sum(token_cost(pair) for pair in wave),
        )
        decisions.extend(await asyncio.gather(*(_judge_pair(pair) for pair in wave)))
    logger = logging.getLogger(__name__)
    logger.info(
        '%s lexical hub comparisons: %d candidate pairs, %d waves, '
        'largest wave %d estimated tokens',
        kind,
        len(candidate_pairs),
        wave_count,
        largest_wave_tokens,
    )
    accepted_pairs = exact_pairs + [
        pair for pair in decisions if pair is not None
    ]
    return names.lexical_groups_from_pairs(rows, accepted_pairs)


async def rebuild(
    kind: str,
    source: str,
    *,
    language_model: dspy.LM,
    session_factory: Callable,
    similarity_threshold: float = 0.9,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuilds source-local lexical hubs for entity or predicate names."""
    component_rows = await getattr(queries, f'all_{kind}_components')(
        session_factory, source
    )
    await writer.persist_name_occurrences(
        component_rows,
        kind,
        session_factory=session_factory,
    )
    occurrence_rows = await queries.all_name_occurrences(
        session_factory, kind, source
    )
    gate = llm.gate(max_concurrency)
    stage = getattr(config.get_settings().stages, f'{kind}_hubs')
    groups = await _judged_groups(
        occurrence_rows,
        kind,
        language_model=language_model,
        similarity_threshold=similarity_threshold,
        gate=gate,
        comparison_token_budget=stage.comparison_token_budget,
    )
    synthesizer = _LexicalDefinitionSynthesizer(language_model)

    async def _synthesize(group: list[dict]) -> dict:
        surface_forms = list(dict.fromkeys(row['text'] for row in group))
        async with gate:
            canonical_index = await synthesizer.aforward(
                surface_forms=surface_forms,
                kind=kind,
            )
        if not 0 <= canonical_index < len(surface_forms):
            raise RuntimeError(
                'lexical name synthesizer returned an invalid surface-form '
                f'index: {canonical_index}'
            )
        canonical_form = surface_forms[canonical_index]
        member_ids = [row['uuid'] for row in group]
        return {
            'uuid': names.name_hub_uuid(kind, source, member_ids),
            'canonical_form': canonical_form,
            'aliases': sorted(set(surface_forms)),
            'members': member_ids,
        }

    hubs = await asyncio.gather(*(_synthesize(group) for group in groups))
    await writer.clear_name_hubs(
        kind,
        source,
        session_factory=session_factory,
    )
    await writer.persist_name_hubs(
        kind,
        list(hubs),
        source=source,
        session_factory=session_factory,
    )
    return {
        'name_hubs': len(hubs),
        'names': len(occurrence_rows),
    }
