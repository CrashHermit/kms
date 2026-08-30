"""Materialize reusable canonical facts from canonical graph components.

Source-level triplets preserve explicit relational evidence and provenance.
This module groups exact subject/predicate/object hub memberships and asks the
LLM to synthesize a standalone canonical assertion for each supported
TripletHub. MetaTripletHub performs the same abstraction across sources; it
must be supported by multiple sources and must not invent consequences.
"""

import asyncio
from collections.abc import Callable

import dspy
from pydantic import BaseModel, Field

from kms.core import embeddings, identity, llm, models, module
from kms.graph import hubs, queries, writer
from kms.graph import triplets as graph_triplets


class _TripletDefinition(BaseModel):
    """A reusable canonical fact synthesized from supported triplets."""

    canonical_name: str = Field(
        description='A concise canonical statement of the assertion.'
    )
    description: str = Field(
        description=(
            'A standalone 1-2 sentence canonical explanation of the '
            'assertion, supported by the supplied triplets.'
        )
    )


class _TripletDefinitionSignature(dspy.Signature):
    r"""
    Synthesize one reusable canonical fact from a fixed subject, predicate,
    and object hub tuple and the source triplets that belong to exactly that
    tuple.

    The source triplets are evidence, not the final canonical abstraction.
    Produce a concise assertion and a standalone explanation that can be
    understood without the original passage. Generalize only the common fact
    supported by the evidence. Preserve negation, conditions, quantifiers,
    mathematical notation, and other qualifiers that appear in the evidence.

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
            'Canonical reusable fact name and standalone explanation.'
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
        return (
            module.require_text(result.canonical_name, 'canonical_name'),
            module.require_text(result.description, 'description'),
        )


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


def build_source_groups(
    triplets: list[models.Triplet],
    *,
    source: str,
    entity_assignments: list[dict],
    predicate_assignments: list[dict],
    event_assignments: list[dict] | None = None,
    entity_hubs: list[dict] | None = None,
    event_hubs: list[dict] | None = None,
    predicate_hubs: list[dict] | None = None,
) -> list[dict]:
    """Build exact source-local TripletHub groups for typed endpoints."""
    entity_map = assignment_map(entity_assignments)
    predicate_map = assignment_map(predicate_assignments)
    event_map = assignment_map(event_assignments or [])
    memberships = build_triplet_memberships(
        triplets,
        source=source,
        entity_assignments=entity_map,
        event_assignments=event_map,
        predicate_assignments=predicate_map,
    )
    grouped = exact_tuple_intersections(list(memberships))
    entity_context = {hub['uuid']: hub for hub in entity_hubs or []}
    event_context = {hub['uuid']: hub for hub in event_hubs or []}
    predicate_context = {hub['uuid']: hub for hub in predicate_hubs or []}
    component_context = entity_context | event_context
    groups = []
    for (subject_hub, predicate_hub, object_hub), indexes in sorted(
        grouped.items()
    ):
        evidence = []
        triplet_ids = []
        for index in sorted(indexes):
            triplet = triplets[index]
            for node_id in triplet.evidence_positions:
                triplet_id = graph_triplets.triplet_uuid(
                    source,
                    node_id,
                    triplet.subject,
                    triplet.predicate,
                    triplet.object,
                )
                triplet_ids.append(triplet_id)
                evidence.append(
                    {
                        'uuid': triplet_id,
                        'subject': triplet.subject,
                        'predicate': triplet.predicate,
                        'object': triplet.object,
                    }
                )
        groups.append(
            {
                'uuid': hubs.triplet_hub_uuid(
                    'source', source, subject_hub, predicate_hub, object_hub
                ),
                'source': source,
                'subject_hub': subject_hub,
                'predicate_hub': predicate_hub,
                'object_hub': object_hub,
                'triplets': triplet_ids,
                'evidence': evidence,
                'subject_name': component_context.get(subject_hub, {}).get(
                    'canonical_name', subject_hub
                ),
                'subject_description': component_context.get(subject_hub, {}).get(
                    'description'
                ),
                'predicate_name': predicate_context.get(predicate_hub, {}).get(
                    'canonical_name', predicate_hub
                ),
                'predicate_description': predicate_context.get(
                    predicate_hub, {}
                ).get('description'),
                'object_name': component_context.get(object_hub, {}).get(
                    'canonical_name', object_hub
                ),
                'object_description': component_context.get(object_hub, {}).get(
                    'description'
                ),
            }
        )
    return groups


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
        [text for text in texts]
    )
    return [
        {**group, 'embedding': vector}
        for group, vector in zip(synthesized, vectors, strict=True)
    ]


def assignment_map(assignments: list[dict]) -> dict[str, tuple[str, ...]]:
    """Converts writer-shaped component assignments into grouped memberships."""
    grouped: dict[str, set[str]] = {}
    for assignment in assignments:
        grouped.setdefault(assignment['component'], set()).add(
            assignment['hub']
        )
    return {
        component: tuple(sorted(hubs)) for component, hubs in grouped.items()
    }


def build_triplet_memberships(
    triplets: list[models.Triplet],
    *,
    source: str,
    entity_assignments: dict[str, tuple[str, ...]],
    predicate_assignments: dict[str, tuple[str, ...]],
    event_assignments: dict[str, tuple[str, ...]] | None = None,
) -> tuple[models.TripletMembership, ...]:
    """Build role-ordered memberships for typed extracted triplets."""
    event_assignments = event_assignments or {}
    memberships = []
    for index, triplet in enumerate(triplets):
        subject_hubs: set[str] = set()
        object_hubs: set[str] = set()
        predicate_hubs: set[str] = set()
        for node_id in triplet.evidence_positions:
            subject_id = (
                identity.entity_uuid(source, node_id, triplet.subject)
                if triplet.subject_kind is models.NodeKind.ENTITY
                else identity.event_uuid(source, node_id, triplet.subject)
            )
            object_id = (
                identity.entity_uuid(source, node_id, triplet.object)
                if triplet.object_kind is models.NodeKind.ENTITY
                else identity.event_uuid(source, node_id, triplet.object)
            )
            subject_hubs.update(
                (entity_assignments if triplet.subject_kind is models.NodeKind.ENTITY
                 else event_assignments).get(subject_id, ())
            )
            object_hubs.update(
                (entity_assignments if triplet.object_kind is models.NodeKind.ENTITY
                 else event_assignments).get(object_id, ())
            )
            predicate_hubs.update(
                predicate_assignments.get(triplet.predicate_uuids[node_id], ())
            )
        memberships.append(models.TripletMembership(
            triplet_index=index,
            source=source,
            subject_hubs=tuple(sorted(subject_hubs)),
            predicate_hubs=tuple(sorted(predicate_hubs)),
            object_hubs=tuple(sorted(object_hubs)),
        ))
    return tuple(memberships)


def exact_tuple_intersections(
    memberships: list[models.TripletMembership],
) -> dict[tuple[str, str, str], set[int]]:
    """Returns triplets having each exact ordered hub tuple membership."""
    result: dict[tuple[str, str, str], set[int]] = {}
    for membership in memberships:
        for subject in membership.subject_hubs:
            for predicate in membership.predicate_hubs:
                for object_ in membership.object_hubs:
                    result.setdefault((subject, predicate, object_), set()).add(
                        membership.triplet_index
                    )
    return result


async def build_source(
    *,
    language_model: dspy.LM,
    source: str,
    triplets: list[models.Triplet],
    entity_assignments: list[dict],
    predicate_assignments: list[dict],
    event_assignments: list[dict] | None = None,
    entity_hubs: list[dict] | None = None,
    event_hubs: list[dict] | None = None,
    predicate_hubs: list[dict] | None = None,
    max_concurrency: int | None = None,
) -> dict:
    """Build source-local TripletHubs from typed endpoint memberships."""
    groups = build_source_groups(
        triplets,
        source=source,
        entity_assignments=entity_assignments,
        event_assignments=event_assignments,
        predicate_assignments=predicate_assignments,
        entity_hubs=entity_hubs,
        event_hubs=event_hubs,
        predicate_hubs=predicate_hubs,
    )
    groups = await _synthesize_groups(
        groups,
        language_model=language_model,
        max_concurrency=max_concurrency,
        tier='source',
    )
    return {
        'triplet_hubs': len(groups),
        'triplets': sum(len(group['triplets']) for group in groups),
        'hubs': groups,
    }


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
