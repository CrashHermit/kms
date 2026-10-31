"""Explicit typed backend fixtures shared by hub reduction tests and smoke."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

HUB_CASES = (
    ('SourceStatementHub', 'source_semantic'),
    ('SourceProcedureHub', 'source_semantic'),
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
    """Build valid member inputs and worker state for one retained hub type.

    ``long_field`` supplies one member value for whole-record budget tests.
    """
    scope = dict(HUB_CASES)[stem]
    model_module = import_module(f'kms2.core.model.{scope}.{_snake(stem)}')
    node_module = import_module(f'kms2.node.{scope}.{_snake(stem)}')
    node_class = getattr(node_module, f'{stem}Node')
    ordinal_key = f'{_snake(stem)}_synthesis_ordinal'

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
            elif field_name == 'content':
                value = f'{stem} complete reconstructed evidence record {index} preserves distinct condition.'
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
            if field_name in {'name', 'canonical_name'}:
                aliases.append(value)
            values[field_name] = value
        items.append(member_type(**values))
    community_key = f'{_snake(stem)}_community'
    state = {ordinal_key: 7, community_key: items}
    return HubFixture(
        stem, scope, model_module, node_class, items, state, uuids, aliases
    )


def _snake(value: str) -> str:
    """Convert the explicit CamelCase case name to its module basename."""
    chars = []
    for index, character in enumerate(value):
        if character.isupper() and index:
            chars.append('_')
        chars.append(character.lower())
    return ''.join(chars)
