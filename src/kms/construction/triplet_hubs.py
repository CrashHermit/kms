"""Materialize reusable canonical facts from canonical graph components.

Source-level triplets preserve explicit relational evidence and provenance.
This module groups exact subject/predicate/object hub memberships and asks the
LLM to synthesize a standalone learner-facing assertion for each supported
TripletHub. MetaTripletHub performs the same abstraction across sources; it
must be supported by multiple sources and must not invent consequences.
"""

import asyncio
from collections.abc import Callable

import dspy
from pydantic import BaseModel, Field

from kms.core import content, embeddings, llm, module
from kms.graph import hubs, queries, writer


class _TripletDefinition(BaseModel):
    """A reusable canonical fact synthesized from supported triplets."""

    canonical_name: str = Field(
        description='A concise canonical statement of the assertion.'
    )
    description: str = Field(
        description=(
            'A standalone 1-2 sentence learner-facing explanation of the '
            'canonical fact, supported by the supplied triplets.'
        )
    )


class _TripletDefinitionSignature(dspy.Signature):
    r"""
    Synthesize one reusable canonical fact from a fixed subject, predicate,
    and object hub tuple and the source triplets that belong to exactly that
    tuple.

    The source triplets are evidence, not the final educational abstraction.
    Produce a concise assertion and a standalone learner-facing explanation
    that can be understood without the original passage. Generalize only the
    common fact supported by the evidence. Preserve negation, conditions,
    quantifiers, mathematical notation, and other qualifiers that appear in
    the evidence.

    The tuple membership is already decided by the graph. Do not add facts,
    infer consequences, or combine the assertion with neighboring facts.
    """

    subject_hub: str = dspy.InputField(
        description='Canonical subject hub name and description.'
    )
    predicate_hub: str = dspy.InputField(
        description='Canonical predicate hub name and description.'
    )
    object_hub: str = dspy.InputField(
        description='Canonical object hub name and description.'
    )
    evidence: list[str] = dspy.InputField(
        description='Only the exact triplets assigned to this hub tuple.'
    )
    scope: str = dspy.InputField(
        description='Whether this is source-local or cross-source synthesis.'
    )
    result: _TripletDefinition = dspy.OutputField(
        description=(
            'Canonical reusable fact name and standalone learner-facing '
            'explanation.'
        )
    )


class _TripletDefinitionSynthesizer(module.Module):
    """Synthesizes searchable metadata for an exact triplet group."""

    signature = _TripletDefinitionSignature
    record_name = 'triplet_hub_synthesizer'

    def encode(
        self,
        subject_hub: str,
        predicate_hub: str,
        object_hub: str,
        evidence: list[str],
        scope: str,
    ) -> dict:
        return {
            'subject_hub': subject_hub,
            'predicate_hub': predicate_hub,
            'object_hub': object_hub,
            'evidence': evidence,
            'scope': scope,
        }

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = prediction.result
        return result.canonical_name, result.description


def _hub_context(group: dict, role: str) -> str:
    name = group[f'{role}_name']
    description = group.get(f'{role}_description')
    if description:
        return f'{name}: {description}'
    return name


def _evidence_text(group: dict) -> list[str]:
    evidence = group.get('evidence') or []
    rendered = {
        f'{item.get("subject", "")} | {item.get("predicate", "")} | '
        f'{item.get("object", "")}'
        for item in evidence
    }
    return sorted(value for value in rendered if value.strip(' |'))


def _prepare_groups(rows: list[dict], tier: str) -> list[dict]:
    groups: list[dict] = []
    for row in rows:
        source = row.get('source') if tier == 'source' else None
        subject_hub = row['subject_hub']
        predicate_hub = row['predicate_hub']
        object_hub = row['object_hub']
        group = {
            'uuid': hubs.triplet_hub_uuid(
                tier,
                source,
                subject_hub,
                predicate_hub,
                object_hub,
            ),
            'source': source,
            'subject_hub': subject_hub,
            'predicate_hub': predicate_hub,
            'object_hub': object_hub,
            'triplets': list(row.get('triplets') or []),
            'evidence': list(row.get('evidence') or []),
        }
        if tier == 'meta':
            local_hubs = set()
            for local in row.get('local_tuples') or []:
                local_hubs.add(
                    hubs.triplet_hub_uuid(
                        'source',
                        local['source'],
                        local['subject_hub'],
                        local['predicate_hub'],
                        local['object_hub'],
                    )
                )
            group['local_hubs'] = sorted(local_hubs)
            if len(set(row.get('sources') or [])) < 2:
                continue
        groups.append({**group, **row})
    return groups


async def _synthesize_groups(
    groups: list[dict],
    *,
    language_model: dspy.LM,
    max_concurrency: int | None,
    tier: str,
) -> list[dict]:
    if not groups:
        return []
    synthesizer = _TripletDefinitionSynthesizer(language_model)
    gate = llm.gate(max_concurrency)

    async def _one(group: dict) -> dict:
        async with gate:
            canonical_name, description = await synthesizer.aforward(
                subject_hub=_hub_context(group, 'subject'),
                predicate_hub=_hub_context(group, 'predicate'),
                object_hub=_hub_context(group, 'object'),
                evidence=_evidence_text(group),
                scope=(
                    'Synthesize a source-local assertion.'
                    if tier == 'source'
                    else 'Synthesize an assertion supported across sources.'
                ),
            )
        return {
            **group,
            'canonical_name': canonical_name,
            'description': description,
        }

    synthesized = await asyncio.gather(*(_one(group) for group in groups))
    texts = [
        f'{group["canonical_name"]}: {group["description"]}'
        for group in synthesized
    ]
    vectors = await embeddings.embedder().embed(
        [content.Content.from_text(text) for text in texts]
    )
    return [
        {**group, 'embedding': vector}
        for group, vector in zip(synthesized, vectors, strict=True)
    ]


async def rebuild(
    *,
    language_model: dspy.LM,
    session_factory: Callable,
    source: str | None = None,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuilds source-local TripletHub nodes for one source or all sources."""
    rows = await queries.triplet_hub_groups(
        session_factory,
        'source',
        source=source,
    )
    groups = _prepare_groups(rows, 'source')
    groups = await _synthesize_groups(
        groups,
        language_model=language_model,
        max_concurrency=max_concurrency,
        tier='source',
    )
    await writer.clear_triplet_hubs(
        'source', session_factory=session_factory, source=source
    )
    await writer.persist_triplet_hubs(
        groups,
        tier='source',
        session_factory=session_factory,
    )
    return {
        'triplet_hubs': len(groups),
        'triplets': sum(len(group['triplets']) for group in groups),
    }


async def rebuild_meta(
    *,
    language_model: dspy.LM,
    session_factory: Callable,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuilds qualified cross-source MetaTripletHub nodes."""
    rows = await queries.triplet_hub_groups(session_factory, 'meta')
    groups = _prepare_groups(rows, 'meta')
    groups = await _synthesize_groups(
        groups,
        language_model=language_model,
        max_concurrency=max_concurrency,
        tier='meta',
    )
    await writer.clear_triplet_hubs('meta', session_factory=session_factory)
    await writer.persist_triplet_hubs(
        groups,
        tier='meta',
        session_factory=session_factory,
    )
    return {
        'meta_triplet_hubs': len(groups),
        'triplets': sum(len(group['triplets']) for group in groups),
    }
