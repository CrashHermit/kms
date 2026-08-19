"""Pure source-local selection of canonical construction knowledge."""

from kms.core import models


def _knowledge_for_members(
    members: list[int], index: models.KnowledgeIndex
) -> models.Knowledge:
    """Selects assertions supported by at least one supplied member node."""
    member_ids = frozenset(members)
    selected = [
        assertion
        for assertion in index.assertions
        if assertion.evidence_node_ids & member_ids
    ]
    return models.Knowledge(
        assertions=tuple(sorted(selected, key=lambda item: item.uuid))
    )


def _check_source(
    source: models.Source, index: models.KnowledgeIndex
) -> None:
    """Rejects accidental cross-source knowledge selection."""
    if source.key != index.source:
        raise ValueError(
            f'knowledge index source {index.source!r} does not match '
            f'bundle source {source.key!r}'
        )


def knowledge_for_statement(
    bundle: models.ConstructionBundle,
    statement: models.Statement,
    index: models.KnowledgeIndex | None = None,
) -> models.Knowledge:
    """Selects canonical assertions supported by statement members."""
    index = index or bundle.knowledge_index
    if index is None:
        return models.Knowledge()
    _check_source(bundle.source, index)
    return _knowledge_for_members(statement.members, index)


def knowledge_for_procedure(
    bundle: models.ConstructionBundle,
    procedure: models.Procedure,
    index: models.KnowledgeIndex | None = None,
) -> models.Knowledge:
    """Selects canonical assertions supported by procedure members."""
    index = index or bundle.knowledge_index
    if index is None:
        return models.Knowledge()
    _check_source(bundle.source, index)
    return _knowledge_for_members(procedure.members, index)
