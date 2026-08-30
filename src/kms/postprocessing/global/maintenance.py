"""Coordinate cross-source global graph derivation."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

entity_hubs = import_module('kms.postprocessing.global.entity_hubs')
predicate_hubs = import_module('kms.postprocessing.global.predicate_hubs')
event_hubs = import_module('kms.postprocessing.global.event_hubs')
procedure_hubs = import_module('kms.postprocessing.global.procedure_hubs')
statement_hubs = import_module('kms.postprocessing.global.statement_hubs')


async def rebuild(
    *,
    session_factory: Callable,
    entity_language_model: Any,
    entity_adjudicator: Any,
    entity_synthesizer: Any,
    event_language_model: Any,
    event_adjudicator: Any,
    event_synthesizer: Any,
    predicate_language_model: Any,
    predicate_adjudicator: Any,
    predicate_synthesizer: Any,
    statement_adjudicator: Any,
    statement_synthesizer: Any,
    procedure_adjudicator: Any,
    procedure_synthesizer: Any,
    max_concurrency: int | None = None,
    include_statements: bool = False,
    include_procedures: bool = False,
) -> dict[str, dict]:
    """Rebuild global entity and predicate hubs and optional learning hubs."""
    entity_result = await entity_hubs.rebuild_global(
        language_model=entity_language_model,
        adjudicator=entity_adjudicator,
        synthesizer=entity_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    predicate_result = await predicate_hubs.rebuild_global(
        language_model=predicate_language_model,
        adjudicator=predicate_adjudicator,
        synthesizer=predicate_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    event_result = await event_hubs.align_global_hubs(
        language_model=event_language_model,
        adjudicator=event_adjudicator,
        synthesizer=event_synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
    result = {
        'entity': entity_result,
        'event': event_result,
        'predicate': predicate_result,
    }
    if include_statements:
        result['statement'] = await statement_hubs.rebuild(
            session_factory=session_factory,
            adjudicator=statement_adjudicator,
            synthesizer=statement_synthesizer,
        )
    if include_procedures:
        result['procedure'] = await procedure_hubs.rebuild(
            session_factory=session_factory,
            adjudicator=procedure_adjudicator,
            synthesizer=procedure_synthesizer,
        )
    return result
