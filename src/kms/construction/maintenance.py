"""Rebuild disposable graph knowledge from durable source records."""

from collections.abc import Callable
from typing import Any

from kms.construction import (
    local_entity_hubs,
    local_predicate_hubs,
    local_procedure_hubs,
    local_statement_hubs,
)


async def rebuild_source(
    source: str,
    *,
    session_factory: Callable,
    entity_language_model: Any,
    entity_adjudicator: Any,
    entity_synthesizer: Any,
    predicate_language_model: Any,
    predicate_adjudicator: Any,
    predicate_synthesizer: Any,
    statement_adjudicator: Any,
    statement_synthesizer: Any,
    procedure_adjudicator: Any,
    procedure_synthesizer: Any,
    max_concurrency: int | None = None,
) -> dict[str, dict]:
    """Rebuild all source-local derived hubs for one source.

    The operation reads persisted components, statements, and procedures. It
    clears and replaces only their derived source-local hubs and membership
    relationships; source nodes, assertions, instructions, procedures, and
    provenance remain durable.

    Args:
        source: The source key to rebuild.
        session_factory: Async callable returning Neo4j sessions.
        entity_language_model: Model used by entity triplet-hub synthesis.
        entity_adjudicator: Entity mention adjudicator.
        entity_synthesizer: Entity hub synthesizer.
        predicate_language_model: Model used by predicate triplet synthesis.
        predicate_adjudicator: Predicate mention adjudicator.
        predicate_synthesizer: Predicate hub synthesizer.
        statement_adjudicator: Statement hub adjudicator.
        statement_synthesizer: Statement hub synthesizer.
        procedure_adjudicator: Procedure hub adjudicator.
        procedure_synthesizer: Procedure hub synthesizer.
        max_concurrency: Optional limit for hub-building calls.

    Returns:
        Counts returned by each source-local rebuild stage.
    """
    entity_result = await local_entity_hubs.rebuild(
        source,
        language_model=entity_language_model,
        adjudicator=entity_adjudicator,
        synthesizer=entity_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    predicate_result = await local_predicate_hubs.rebuild(
        source,
        language_model=predicate_language_model,
        adjudicator=predicate_adjudicator,
        synthesizer=predicate_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    statement_result = await local_statement_hubs.rebuild(
        source,
        session_factory=session_factory,
        adjudicator=statement_adjudicator,
        synthesizer=statement_synthesizer,
    )
    procedure_result = await local_procedure_hubs.rebuild(
        source,
        session_factory=session_factory,
        adjudicator=procedure_adjudicator,
        synthesizer=procedure_synthesizer,
    )
    return {
        'entity': entity_result,
        'predicate': predicate_result,
        'statement': statement_result,
        'procedure': procedure_result,
    }


