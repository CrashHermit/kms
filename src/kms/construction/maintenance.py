"""Rebuild disposable graph knowledge from durable source records."""

from collections.abc import Callable
from typing import Any

import dspy

from kms.construction import (
    local_entity_hubs,
    local_event_hubs,
    local_predicate_hubs,
    local_procedure_hubs,
    local_statement_hubs,
    triplet_hubs,
)


async def rebuild_source(
    source: str,
    *,
    session_factory: Callable,
    entity_language_model: Any,
    entity_synthesizer: Any,
    predicate_language_model: Any,
    predicate_synthesizer: Any,
    event_synthesizer: Any,
    triplet_language_model: dspy.LM,
    statement_synthesizer: Any,
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
        entity_language_model: Model used by entity hub synthesis.
        entity_synthesizer: Entity hub synthesizer.
        predicate_language_model: Model used by predicate hub synthesis.
        predicate_synthesizer: Predicate hub synthesizer.
        event_synthesizer: Event hub synthesizer.
        triplet_language_model: Model used by triplet hub synthesis.
        statement_synthesizer: Statement hub synthesizer.
        procedure_synthesizer: Procedure hub synthesizer.
        max_concurrency: Optional limit for hub-building calls.

    Returns:
        Counts returned by each source-local rebuild stage.
    """
    entity_result = await local_entity_hubs.rebuild(
        source,
        language_model=entity_language_model,
        synthesizer=entity_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    predicate_result = await local_predicate_hubs.rebuild(
        source,
        language_model=predicate_language_model,
        synthesizer=predicate_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    event_result = await local_event_hubs.rebuild(
        source,
        synthesizer=event_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    triplet_result = await triplet_hubs.rebuild(
        language_model=triplet_language_model,
        source=source,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    statement_result = await local_statement_hubs.rebuild(
        source,
        session_factory=session_factory,
        synthesizer=statement_synthesizer,
        max_concurrency=max_concurrency,
    )
    procedure_result = await local_procedure_hubs.rebuild(
        source,
        session_factory=session_factory,
        synthesizer=procedure_synthesizer,
        max_concurrency=max_concurrency,
    )
    return {
        'entity': entity_result,
        'predicate': predicate_result,
        'event': event_result,
        'triplet': triplet_result,
        'statement': statement_result,
        'procedure': procedure_result,
    }
