"""Deterministic lexical name-hub materialization."""

import asyncio
from collections.abc import Callable
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, module
from kms.graph import names, queries, writer


class _LexicalDefinition(BaseModel):
    """The selected member index for one lexical cluster."""

    canonical_index: int = Field(
        description='Zero-based index of the preferred supplied surface form.'
    )


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

    surface_forms: list[str] = dspy.InputField(
        description='The exact lexical phrases in one deterministic cluster.'
    )
    kind: str = dspy.InputField(
        description='Whether the phrases are entity names or predicate phrases.'
    )
    result: _LexicalDefinition = dspy.OutputField(
        description=(
            'The zero-based index of one supplied surface form; never invent '
            'or rewrite a form.'
        )
    )


class _LexicalMembership(BaseModel):
    """The lexical judge's decision for one deterministic candidate pair."""

    decision: Literal['Merge', 'Separate'] = Field(
        description=(
            "'Merge' when both phrases are the same written form or a "
            "conservative lexical variant; otherwise 'Separate'."
        )
    )


class _LexicalMembershipSignature(dspy.Signature):
    r"""
    Decide whether two already-similar complete phrases belong to one
    lexical-form group.

    The deterministic similarity filter has already made these two phrases
    candidates. Judge only written-form equivalence, not semantic equivalence.

    Merge spelling, punctuation, separator, whitespace, or closely related
    inflectional variants of the same phrase, such as "color"/"colour" or
    "sub-graph"/"subgraph". Keep synonyms such as "car"/"automobile",
    broader or narrower phrases such as "graph"/"subgraph", different word
    orders, and positive/negative phrases Separate. Preserve mathematical
    notation, meaningful case, quantifiers, prepositions, and negation.
    """

    left: str = dspy.InputField(
        description='The first complete lexical phrase.'
    )
    right: str = dspy.InputField(
        description='The second complete lexical phrase.'
    )
    kind: str = dspy.InputField(
        description='Whether the phrases are entity names or predicate phrases.'
    )
    result: _LexicalMembership = dspy.OutputField(
        description=(
            "'Merge' or 'Separate'; never infer a semantic relationship."
        )
    )


class _LexicalMembershipJudge(module.Module):
    """Adjudicates deterministic lexical candidates before naming."""

    signature = _LexicalMembershipSignature
    record_name = 'lexical_membership_judge'

    def encode(self, left: str, right: str, kind: str) -> dict:
        """Build the lexical membership judge inputs."""
        return {'left': left, 'right': right, 'kind': kind}

    def decode(self, prediction, **inputs) -> str:
        """Return the judge's Merge or Separate decision."""
        return prediction.result.decision


class _LexicalDefinitionSynthesizer(module.Module):
    """Chooses presentation text without deciding lexical membership."""

    signature = _LexicalDefinitionSignature
    record_name = 'lexical_name_synthesizer'

    def encode(self, surface_forms: list[str], kind: str) -> dict:
        """Build the canonical-form selector inputs."""
        return {'surface_forms': surface_forms, 'kind': kind}

    def decode(self, prediction, **inputs) -> int:
        """Return the selected supplied surface-form index."""
        return prediction.result.canonical_index


async def _judged_groups(
    rows: list[dict],
    kind: str,
    *,
    language_model: dspy.LM,
    similarity_threshold: float,
    gate: asyncio.Semaphore,
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
        if decision not in {'Merge', 'Separate'}:
            raise RuntimeError(
                f'lexical membership judge returned invalid decision: '
                f'{decision}'
            )
        return pair if decision == 'Merge' else None

    decisions = await asyncio.gather(
        *(_judge_pair(pair) for pair in candidate_pairs)
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
    groups = await _judged_groups(
        occurrence_rows,
        kind,
        language_model=language_model,
        similarity_threshold=similarity_threshold,
        gate=gate,
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


async def rebuild_meta(
    kind: str,
    *,
    language_model: dspy.LM,
    session_factory: Callable,
    similarity_threshold: float = 0.9,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuilds qualified meta lexical hubs from source-local name hubs."""
    source_hubs = await queries.all_name_hubs(session_factory, kind)
    await writer.clear_meta_name_hubs(
        kind,
        session_factory=session_factory,
    )
    if not source_hubs:
        return {'name_hubs': 0, 'source_name_hubs': 0}

    sources = {row['source'] for row in source_hubs if row.get('source')}
    if len(sources) < 2:
        raise RuntimeError(
            f'name hubs (meta/{kind}): requires at least two distinct '
            f'sources, found {len(sources)}'
        )

    gate = llm.gate(max_concurrency)
    groups = await _judged_groups(
        source_hubs,
        kind,
        language_model=language_model,
        similarity_threshold=similarity_threshold,
        gate=gate,
    )
    qualified_groups = [
        group
        for group in groups
        if len({row['source'] for row in group if row.get('source')}) >= 2
    ]
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
                'meta lexical name synthesizer returned an invalid '
                f'surface-form index: {canonical_index}'
            )
        canonical_form = surface_forms[canonical_index]
        aliases = sorted(
            {alias for row in group for alias in row.get('aliases', []) or []}
            | set(surface_forms)
        )
        member_ids = [row['uuid'] for row in group]
        return {
            'uuid': names.meta_name_hub_uuid(kind, member_ids),
            'canonical_form': canonical_form,
            'aliases': aliases,
            'members': member_ids,
            'sources': sorted({row['source'] for row in group}),
        }

    hubs = await asyncio.gather(
        *(_synthesize(group) for group in qualified_groups)
    )
    await writer.persist_meta_name_hubs(
        kind,
        list(hubs),
        session_factory=session_factory,
    )
    return {
        'name_hubs': len(hubs),
        'source_name_hubs': len(source_hubs),
    }
