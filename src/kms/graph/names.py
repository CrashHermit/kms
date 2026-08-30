"""Lexical occurrence and lexical-hub graph conventions."""

import unicodedata
from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

ENTITY_NAME_LABEL = 'EntityName'
EVENT_NAME_LABEL = 'EventName'
PREDICATE_NAME_LABEL = 'PredicateName'
LOCAL_ENTITY_NAME_HUB_LABEL = 'LocalEntityNameHub'
LOCAL_EVENT_NAME_HUB_LABEL = 'LocalEventNameHub'
LOCAL_PREDICATE_NAME_HUB_LABEL = 'LocalPredicateNameHub'
GLOBAL_ENTITY_NAME_HUB_LABEL = 'GlobalEntityNameHub'
GLOBAL_EVENT_NAME_HUB_LABEL = 'GlobalEventNameHub'
GLOBAL_PREDICATE_NAME_HUB_LABEL = 'GlobalPredicateNameHub'

_NAME_LABELS = {
    'entity': ENTITY_NAME_LABEL,
    'event': EVENT_NAME_LABEL,
    'predicate': PREDICATE_NAME_LABEL,
}
_LOCAL_NAME_HUB_LABELS = {
    'entity': LOCAL_ENTITY_NAME_HUB_LABEL,
    'event': LOCAL_EVENT_NAME_HUB_LABEL,
    'predicate': LOCAL_PREDICATE_NAME_HUB_LABEL,
}
_GLOBAL_NAME_HUB_LABELS = {
    'entity': GLOBAL_ENTITY_NAME_HUB_LABEL,
    'event': GLOBAL_EVENT_NAME_HUB_LABEL,
    'predicate': GLOBAL_PREDICATE_NAME_HUB_LABEL,
}


def name_label(kind: str) -> str:
    """Returns the lexical occurrence label for one component kind."""
    try:
        return _NAME_LABELS[kind]
    except KeyError as error:
        raise ValueError(f'unknown name kind: {kind}') from error


def name_hub_label(kind: str, tier: str = 'local') -> str:
    """Returns the local or global lexical hub label."""
    if tier in {'local', 'source'}:
        labels = _LOCAL_NAME_HUB_LABELS
    elif tier in {'global', 'meta'}:
        labels = _GLOBAL_NAME_HUB_LABELS
    else:
        raise ValueError(f'unknown name-hub tier: {tier}')
    try:
        return labels[kind]
    except KeyError as error:
        raise ValueError(f'unknown name kind: {kind}') from error


def local_name_hub_label(kind: str) -> str:
    """Returns the local lexical hub label."""
    return name_hub_label(kind, tier='local')


def global_name_hub_label(kind: str) -> str:
    """Returns the global lexical hub label."""
    return name_hub_label(kind, tier='global')


def normalize_text(text: str) -> str:
    """Normalizes harmless lexical formatting while preserving meaning.

    Case, punctuation, mathematical notation, and token order are preserved.
    Correction and formatting are upstream responsibilities; this function is
    not an OCR repair pass.
    """
    normalized = unicodedata.normalize('NFKC', text)
    return ' '.join(normalized.split())


def name_uuid(kind: str, component_uuid: str) -> str:
    """Returns the stable lexical occurrence id for one component."""
    return uuid5(NAMESPACE_URL, f'{kind}#name#{component_uuid}').hex


def name_properties(
    kind: str,
    source: str,
    component_uuid: str,
    text: str,
    node_position: int,
) -> dict:
    """Builds a lexical occurrence row for an Entity or Predicate."""
    return {
        'uuid': name_uuid(kind, component_uuid),
        'source': nodes.source_uuid(source),
        'node_position': node_position,
        'text': text,
        'normalized_text': normalize_text(text),
        'component_uuid': component_uuid,
    }


def name_hub_uuid(kind: str, source: str, member_ids: list[str]) -> str:
    """Returns a stable id for a source-local lexical cluster."""
    identity = '|'.join(sorted(member_ids))
    return uuid5(
        NAMESPACE_URL,
        f'{source}#{kind}_name_hub#{identity}',
    ).hex


def global_name_hub_uuid(kind: str, member_ids: list[str]) -> str:
    """Returns a stable id for a cross-source lexical cluster."""
    identity = '|'.join(sorted(member_ids))
    return uuid5(
        NAMESPACE_URL,
        f'meta#{kind}_name_hub#{identity}',
    ).hex


def global_name_hub_properties(
    kind: str,
    canonical_form: str,
    aliases: list[str],
    sources: list[str],
    *,
    hub_id: str,
) -> dict:
    """Builds properties for a cross-source lexical hub."""
    return {
        'uuid': hub_id,
        'canonical_form': canonical_form,
        'normalized_form': normalize_text(canonical_form),
        'aliases': aliases,
        'sources': sorted(set(sources)),
        'kind': kind,
    }


def name_hub_properties(
    kind: str,
    source: str,
    canonical_form: str,
    aliases: list[str],
    *,
    hub_id: str,
) -> dict:
    """Builds properties for a source-local lexical hub."""
    return {
        'uuid': hub_id,
        'source': nodes.source_uuid(source),
        'canonical_form': canonical_form,
        'normalized_form': normalize_text(canonical_form),
        'aliases': aliases,
        'kind': kind,
    }


_NEGATION_TOKENS = frozenset({'no', 'not', 'never', 'without'})


def _levenshtein_distance(left: str, right: str) -> int:
    """Returns insertion/deletion/substitution distance for two phrases."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def lexical_similarity(left: str, right: str) -> float:
    """Returns ordered character similarity for two normalized phrases."""
    if left == right:
        return 1.0
    left_negation = set(left.split()) & _NEGATION_TOKENS
    right_negation = set(right.split()) & _NEGATION_TOKENS
    if left_negation != right_negation:
        return 0.0
    longest = max(len(left), len(right))
    return 1.0 - (_levenshtein_distance(left, right) / longest)


def lexical_candidate_pairs(
    rows: list[dict],
    *,
    threshold: float = 0.9,
) -> list[tuple[int, int]]:
    """Returns deterministic candidate pairs for lexical adjudication."""
    normalized = [row['normalized_text'] for row in rows]
    return [
        (left, right)
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
        if lexical_similarity(normalized[left], normalized[right]) >= threshold
    ]


def lexical_groups_from_pairs(
    rows: list[dict],
    pairs: list[tuple[int, int]],
) -> list[list[dict]]:
    """Builds groups from already-accepted lexical candidate pairs."""
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(rows))}
    for left, right in pairs:
        adjacency[left].append(right)
        adjacency[right].append(left)

    groups: list[list[dict]] = []
    visited: set[int] = set()
    for start in range(len(rows)):
        if start in visited:
            continue
        stack = [start]
        group: list[dict] = []
        while stack:
            index = stack.pop()
            if index in visited:
                continue
            visited.add(index)
            group.append(rows[index])
            stack.extend(adjacency[index])
        groups.append(group)
    return groups


def lexical_groups(
    rows: list[dict],
    *,
    threshold: float = 0.9,
) -> list[list[dict]]:
    """Groups lexical occurrences by deterministic phrase similarity.

    This helper exposes the pre-judge deterministic grouping used by tests and
    callers that only need candidate components. Production hub rebuilds use
    the candidate pairs followed by an LLM Merge/Separate judge.
    """
    return lexical_groups_from_pairs(
        rows,
        lexical_candidate_pairs(rows, threshold=threshold),
    )
