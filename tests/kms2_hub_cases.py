"""Explicit typed backend fixtures shared by hub reduction tests and smoke."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

HUB_CASES = (
    ('SourceEntityHub', 'source_semantic'),
    ('SourceEventHub', 'source_semantic'),
    ('SourcePredicateHub', 'source_semantic'),
    ('SourceStatementHub', 'source_semantic'),
    ('SourceProcedureHub', 'source_semantic'),
    ('SourceTripletHub', 'source_semantic'),
    ('GlobalEntityHub', 'global_semantic'),
    ('GlobalEventHub', 'global_semantic'),
    ('GlobalPredicateHub', 'global_semantic'),
    ('GlobalStatementHub', 'global_semantic'),
    ('GlobalProcedureHub', 'global_semantic'),
    ('GlobalTripletHub', 'global_semantic'),
)


@dataclass(frozen=True)
class HubFixture:
    """Concrete input models and worker state for one hub type."""

    stem: str
    scope: str
    model_module: Any
    node_class: type
    backend_items: list[Any]
    state: dict[str, Any]
    backend_uuids: list[str]
    expected_aliases: list[str]


def build_hub_fixture(
    stem: str,
    *,
    count: int = 10,
    long_field: tuple[str, str] | None = None,
) -> HubFixture:
    """Build valid backend objects; no budgets or token counters are included.

    ``long_field`` names a member field and supplies its value, enabling
    whole-record budget boundary tests while retaining valid backend metadata.
    ``backend_items`` is a member list for ten community hubs and a singleton
    group list for triplet hubs. ``state`` is directly usable by synthesis_worker.
    """
    scope = dict(HUB_CASES)[stem]
    model_module = import_module(f'kms2.core.model.{scope}.{_snake(stem)}')
    node_module = import_module(f'kms2.node.{scope}.{_snake(stem)}')
    node_class = getattr(node_module, f'{stem}Node')
    ordinal_key = f'{_snake(stem)}_synthesis_ordinal'
    if stem.endswith('TripletHub'):
        group = _triplet_group(model_module, stem, long_field)
        group_key = f'{_snake(stem)}_group'
        state = {ordinal_key: 7, group_key: group}
        uuids = [
            group.subject_hub_uuid,
            group.predicate_hub_uuid,
            group.object_hub_uuid,
            *(
                group.triplet_uuids
                if stem == 'SourceTripletHub'
                else group.global_triplet_uuids
            ),
        ]
        if stem == 'GlobalTripletHub':
            uuids.extend(
                item.source_triplet_hub_uuid for item in group.evidence
            )
        return HubFixture(
            stem, scope, model_module, node_class, [group], state, uuids, []
        )

    member_type = getattr(model_module, f'{stem}Member')
    items: list[Any] = []
    uuids = []
    aliases = []
    for index in range(count):
        values = {}
        for field_name in member_type.model_fields:
            if field_name == 'uuid':
                value = f'opaque-{stem}-{index}'
                uuids.append(value)
            elif field_name == 'description':
                value = f'{stem} evidence record {index} preserves distinct condition.'
            elif field_name in {'name', 'canonical_name', 'predicate'}:
                value = f'{stem} name {index}'
            else:
                raise AssertionError(
                    f'No fixture value for {stem}.{field_name}'
                )
            if (
                long_field is not None
                and field_name == long_field[0]
                and index == 0
            ):
                value = long_field[1]
            if stem == 'GlobalStatementHub':
                if field_name == 'description':
                    aliases.append(value)
            elif field_name in {'name', 'canonical_name', 'predicate'}:
                aliases.append(value)
            values[field_name] = value
        items.append(member_type(**values))
    community_key = f'{_snake(stem)}_community'
    state = {ordinal_key: 7, community_key: items}
    return HubFixture(
        stem, scope, model_module, node_class, items, state, uuids, aliases
    )


def _triplet_group(
    model_module: Any, stem: str, long_field: tuple[str, str] | None
) -> Any:
    role_type = getattr(model_module, f'{stem}Role')
    role_values = [
        role_type(
            name=f'{stem} {role} name', description=f'{stem} {role} context'
        )
        for role in ('subject', 'predicate', 'object')
    ]
    if long_field is not None and long_field[0] == 'role_description':
        role_values[0] = role_type(
            name=role_values[0].name, description=long_field[1]
        )
    if stem == 'SourceTripletHub':
        evidence_type = getattr(model_module, f'{stem}Evidence')
        evidence = [
            evidence_type(
                triplet_uuid=f'source-triplet-{index}',
                fact_text=(
                    long_field[1]
                    if long_field is not None
                    and long_field[0] == 'fact_text'
                    and index == 0
                    else f'Fact {index} directs alpha-{index} to beta-{index}.'
                ),
                subject=f'alpha-{index}',
                predicate='directs',
                object=f'beta-{index}',
            )
            for index in range(5)
        ]
        return getattr(model_module, f'{stem}Group')(
            subject_hub_uuid='subject-uuid',
            subject_hub=role_values[0],
            predicate_hub_uuid='predicate-uuid',
            predicate_hub=role_values[1],
            object_hub_uuid='object-uuid',
            object_hub=role_values[2],
            triplet_uuids=[f'source-triplet-{index}' for index in range(5)],
            evidence=evidence,
        )
    evidence_type = getattr(model_module, f'{stem}Evidence')
    evidence = [
        evidence_type(
            global_triplet_uuid=f'global-triplet-{index}',
            source_triplet_hub_uuid=f'source-hub-{index}',
            canonical_name=f'Relation {index}',
            description=(
                long_field[1]
                if long_field is not None
                and long_field[0] == 'source_triplet_description'
                and index == 0
                else f'Global relation evidence {index}.'
            ),
        )
        for index in range(5)
    ]
    return getattr(model_module, f'{stem}Group')(
        subject_hub_uuid='subject-uuid',
        subject_hub=role_values[0],
        predicate_hub_uuid='predicate-uuid',
        predicate_hub=role_values[1],
        object_hub_uuid='object-uuid',
        object_hub=role_values[2],
        global_triplet_uuids=[f'global-triplet-{index}' for index in range(5)],
        evidence=evidence,
    )


def _snake(value: str) -> str:
    """Convert the explicit CamelCase case name to its module basename."""
    chars = []
    for index, character in enumerate(value):
        if character.isupper() and index:
            chars.append('_')
        chars.append(character.lower())
    return ''.join(chars)
