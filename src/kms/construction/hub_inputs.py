"""Build source-local semantic hub inputs in memory."""

from typing import Literal

from kms.core import identity, models, state


def _component_uuid(
    kind: Literal['entity', 'predicate'],
    source: str,
    node_id: int,
    triplet_uuid: str,
    name: str,
) -> str:
    if kind == 'entity':
        return identity.entity_uuid(source, node_id, name)
    return identity.predicate_uuid(triplet_uuid)


def build_hub_components(
    *,
    kind: Literal['entity', 'predicate'],
    source: str,
    triplets: list[models.Triplet],
    descriptions: dict[int, dict[str, str | None]],
    embeddings: dict[int, dict[str, list[float]]],
) -> tuple[models.HubComponent, ...]:
    """Builds local occurrence records from enrichment outputs."""
    components: list[models.HubComponent] = []
    for triplet in triplets:
        terms = (
            (triplet.subject, triplet.object)
            if kind == 'entity'
            else (triplet.predicate,)
        )
        for node_id in triplet.node_ids:
            for name in dict.fromkeys(terms):
                description = descriptions.get(node_id, {}).get(name)
                embedding = embeddings.get(node_id, {}).get(name)
                if embedding is None:
                    raise ValueError(
                        f'{kind} {name!r} at node {node_id} lacks embedding'
                    )
                triplet_uuid = triplet.occurrence_uuids.get(node_id)
                if triplet_uuid is None:
                    raise ValueError(
                        f'triplet occurrence for node {node_id} lacks uuid'
                    )
                components.append(
                    models.HubComponent(
                        uuid=_component_uuid(
                            kind, source, node_id, triplet_uuid, name
                        ),
                        source=source,
                        node_id=node_id,
                        name=name,
                        description=description,
                        embedding=list(embedding),
                    )
                )
    return tuple(components)


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


def _component_record(
    component: models.HubComponent | dict,
) -> models.HubRecord:
    if isinstance(component, dict):
        return models.HubRecord(
            uuid=component['uuid'],
            name=component.get('canonical_name', component.get('name', '')),
            description=component.get('description'),
            embedding=list(component['embedding']),
            source=component.get('source') or '',
            aliases=tuple(component.get('aliases') or []),
        )
    return models.HubRecord(
        uuid=component.uuid,
        name=component.name,
        description=component.description,
        embedding=component.embedding,
        source=component.source,
    )


class HubInputNode:
    """Builds both source-local hub assignment bundles in memory."""

    async def run(self, current_state: dict) -> dict:
        """Builds entity and predicate bundles from the canonical bundle."""
        bundle = state.to_construction_bundle(current_state)
        source = bundle.source.key or ''
        entity_bundle = models.HubBuildBundle(
            source=source,
            components=tuple(
                _component_record(component)
                for component in bundle.entity_hub_components
            ),
            candidate_hubs=tuple(
                _component_record(component)
                for component in bundle.entity_hub_records
            ),
        )
        predicate_bundle = models.HubBuildBundle(
            source=source,
            components=tuple(
                _component_record(component)
                for component in bundle.predicate_hub_components
            ),
            candidate_hubs=tuple(
                _component_record(component)
                for component in bundle.predicate_hub_records
            ),
        )
        bundle.entity_hub_bundle = entity_bundle
        bundle.predicate_hub_bundle = predicate_bundle
        return {
            'entity_hub_bundle': entity_bundle,
            'predicate_hub_bundle': predicate_bundle,
            'construction_bundle': bundle,
        }


def build_triplet_memberships(
    triplets: list[models.Triplet],
    *,
    source: str,
    entity_assignments: dict[str, tuple[str, ...]],
    predicate_assignments: dict[str, tuple[str, ...]],
) -> tuple[models.TripletMembership, ...]:
    """Builds role-ordered canonical memberships for extracted triplets."""
    memberships = []
    for index, triplet in enumerate(triplets):
        subject_hubs: set[str] = set()
        object_hubs: set[str] = set()
        predicate_hubs: set[str] = set()
        for node_id in triplet.node_ids:
            subject_hubs.update(
                entity_assignments.get(
                    identity.entity_uuid(source, node_id, triplet.subject), ()
                )
            )
            object_hubs.update(
                entity_assignments.get(
                    identity.entity_uuid(source, node_id, triplet.object), ()
                )
            )
            predicate_hubs.update(
                predicate_assignments.get(
                    triplet.occurrence_uuids[node_id],
                    (),
                )
            )
        memberships.append(
            models.TripletMembership(
                triplet_index=index,
                source=source,
                subject_hubs=tuple(sorted(subject_hubs)),
                predicate_hubs=tuple(sorted(predicate_hubs)),
                object_hubs=tuple(sorted(object_hubs)),
            )
        )
    return tuple(memberships)


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
