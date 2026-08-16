"""Rebuild disposable graph knowledge from durable source records."""

from collections.abc import Callable
from typing import Any

from kms.construction import (
    entity_hubs,
    predicate_hubs,
    procedure_hubs,
    statement_hubs,
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
    entity_result = await entity_hubs.rebuild_source(
        source,
        language_model=entity_language_model,
        adjudicator=entity_adjudicator,
        synthesizer=entity_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    predicate_result = await predicate_hubs.rebuild_source(
        source,
        language_model=predicate_language_model,
        adjudicator=predicate_adjudicator,
        synthesizer=predicate_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    statement_result = await statement_hubs.rebuild(
        source,
        session_factory=session_factory,
        adjudicator=statement_adjudicator,
        synthesizer=statement_synthesizer,
    )
    procedure_result = await procedure_hubs.rebuild(
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


async def rebuild_meta(
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
    include_statement_learning: bool = False,
    include_procedure_learning: bool = False,
) -> dict[str, dict]:
    """Rebuild cross-source semantic and learning hubs manually.

    Statement and procedure meta hubs are opt-in because they are a separate
    cross-source learning layer.

    Args:
        session_factory: Async callable returning Neo4j sessions.
        entity_language_model: Model used by entity meta-hub synthesis.
        entity_adjudicator: Entity hub adjudicator.
        entity_synthesizer: Entity hub synthesizer.
        predicate_language_model: Model used by predicate meta synthesis.
        predicate_adjudicator: Predicate hub adjudicator.
        predicate_synthesizer: Predicate hub synthesizer.
        statement_adjudicator: Statement hub adjudicator.
        statement_synthesizer: Statement hub synthesizer.
        procedure_adjudicator: Procedure hub adjudicator.
        procedure_synthesizer: Procedure hub synthesizer.
        max_concurrency: Optional limit for hub-building calls.

    Returns:
        Counts returned by each meta-hub rebuild stage.
    """
    entity_result = await entity_hubs.rebuild_meta(
        language_model=entity_language_model,
        adjudicator=entity_adjudicator,
        synthesizer=entity_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    predicate_result = await predicate_hubs.rebuild_meta(
        language_model=predicate_language_model,
        adjudicator=predicate_adjudicator,
        synthesizer=predicate_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    result = {'entity': entity_result, 'predicate': predicate_result}
    if include_statement_learning:
        result['statement'] = await statement_hubs.rebuild_meta(
            session_factory=session_factory,
            adjudicator=statement_adjudicator,
            synthesizer=statement_synthesizer,
        )
    if include_procedure_learning:
        result['procedure'] = await procedure_hubs.rebuild_meta(
            session_factory=session_factory,
            adjudicator=procedure_adjudicator,
            synthesizer=procedure_synthesizer,
        )
    return result
