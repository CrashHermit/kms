"""
Graph representation of the procedural layer — the ``:Procedure`` containers and their ``:Event``
step chains (see ``docs/UNIFIED-KG.md``, "Edges (math-first)").

An entity's ``procedures`` are its *derivations* — a proof, a solution, a physics derivation: the
procedural half of the bi-modal graph — so they are reified out of the entity (where the older schema
kept them as JSON-string blobs) into real graph structure: one ``:Procedure`` per derivation, hung off
its entity via ``(:Entity)-[:HAS_PROCEDURE]->(:Procedure)``, and one ``:Event`` per step, threaded
``(:Procedure)-[:FIRST]->(:Event)-[:THEN]->(:Event)-…``. The entity keeps only its *statement*
structure (its own ``bodylist``); the doing lives here. This mirrors ``graph.nodes``/``graph.entities``
and stays free of the neo4j driver (pure mapping) — the driver lives in ``graph.db``, the writes in
``graph.writer``.

Representation: a procedure carries the base ``:Procedure`` label and its ``type`` (proof / solution /
derivation / …) as a **property** — the same "kind = label, type = property" rule the entity layer
follows, for the same reason: the procedure finder induces the type per book, so a per-kind label set
would grow without bound (``docs/GENERALIZATION.md``). Events likewise carry only ``:Event``, their
``action`` role a property. A procedure with no step decomposition (a solution, typically) is
persisted as a bare container carrying its ``contents`` and no event chain.

Identity: deterministic uuid5s, disjoint from node/entity/hub uuids by their segment. A procedure keys
on ``(source, entity id, type, procedure index)`` — the type keeps a proof #0 distinct from a solution
#0 on the same entity — and an event on the procedure key plus its step index. So re-persisting a book
MERGEs onto the same procedure/event vertices instead of duplicating them.

Event provenance: a ``bodylist`` step is a content slice, not tied to a specific source node id, so no
``(:Event)-[:DERIVED_FROM]->(:Node)`` edge is drawn here — provenance is reachable transitively
(event → procedure → entity → its member ``:Node`` s). A precise step→node link is later work.
"""

from collections.abc import Iterator
from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph.entities import entity_uuid
from kms.graph.nodes import source_uuid

PROCEDURE_LABEL = 'Procedure'
EVENT_LABEL = 'Event'


def procedure_uuid(source: str, entity_id: int, kind: str, index: int) -> str:
    """Stable, deterministic vertex key for a procedure: uuid5 over ``(source, entity id, type,
    procedure index)``. The ``procedure#`` segment keeps it disjoint from node/entity/hub uuids; the
    type disambiguates a proof from a solution on the same entity id."""
    return uuid5(
        NAMESPACE_URL, f'{source}#procedure#{entity_id}#{kind}#{index}'
    ).hex


def event_uuid(
    source: str, entity_id: int, kind: str, proc_index: int, step_index: int
) -> str:
    """Stable, deterministic vertex key for one procedure step: the procedure key plus the step's
    position. The ``event#`` segment keeps it disjoint from every other uuid namespace."""
    return uuid5(
        NAMESPACE_URL,
        f'{source}#event#{entity_id}#{kind}#{proc_index}#{step_index}',
    ).hex


def procedure_events(
    entity: models.Entity, source: str
) -> Iterator[tuple[str, models.BodySegment]]:
    """Yield ``(event_uuid, step)`` for every step of every procedure of the entity — the reified
    ``:Event`` identities. The single place the ``:Event`` key is derived from the (procedure index,
    step index) scheme, so a consumer (e.g. the step-level ``:USES`` builder or the event
    conceptualization) reuses it instead of reconstructing the same uuid."""
    for kind, proc_index, procedure in _derivations(entity):
        for step_index, step in enumerate(procedure.steps):
            yield (
                event_uuid(source, entity.id, kind, proc_index, step_index),
                step,
            )


def _derivations(
    entity: models.Entity,
) -> list[tuple[str, int, models.Procedure]]:
    """The entity's procedures as ``(type, index, procedure)`` tuples, in the order the procedure
    finder emitted them. The index is per-type (proof #0, #1, …; solution #0, …), so a re-run keys
    the same procedure even if another type is added before it; the uuid folds in the type, so the
    index spaces don't collide."""
    rows: list[tuple[str, int, models.Procedure]] = []
    counts: dict[str, int] = {}
    for procedure in entity.procedures:
        kind = procedure.type
        index = counts.get(kind, 0)
        counts[kind] = index + 1
        rows.append((kind, index, procedure))
    return rows


def procedure_properties(
    source: str,
    entity_id: int,
    kind: str,
    index: int,
    procedure: models.Procedure,
) -> dict:
    """The Neo4j property map for one procedure: its stable uuid, the source link, the ``type``, its
    per-type ``index``, ``contents`` as a native string array, and ``generated`` — true when the
    procedure finder *created* the derivation (Math-LLM completion of a task that showed none)
    rather than extracting it from the page, so a query can tell page truth from model output. Empty
    contents is dropped; ``generated`` is written only when true."""
    props = {
        'uuid': procedure_uuid(source, entity_id, kind, index),
        'source': source_uuid(source),
        'type': kind,
        'index': index,
        'contents': procedure.contents or None,
        'generated': procedure.generated or None,
    }
    return {key: value for key, value in props.items() if value is not None}


def event_properties(
    source: str,
    entity_id: int,
    kind: str,
    proc_index: int,
    step_index: int,
    step: models.BodySegment,
) -> dict:
    """The Neo4j property map for one procedure step: its stable uuid, the source link, the ``action``
    (the tactic role, a property since roles are open), the step ``text``, and its ordinal ``index``."""
    return {
        'uuid': event_uuid(source, entity_id, kind, proc_index, step_index),
        'source': source_uuid(source),
        'action': step.action,
        'text': step.description,
        'index': step_index,
    }


def procedure_rows(entities: list[models.Entity], source: str) -> list[dict]:
    """Every procedure's property map across the overlay, one flat list — procedures carry a single
    ``:Procedure`` label (their ``type`` is a property, not a per-kind label), so one batched MERGE
    writes them all."""
    return [
        procedure_properties(source, entity.id, kind, index, procedure)
        for entity in entities
        for kind, index, procedure in _derivations(entity)
    ]


def event_rows(entities: list[models.Entity], source: str) -> list[dict]:
    """Every step's property map across the overlay, one flat list — events carry a single ``:Event``
    label (their ``action`` is a property, not a per-type label), so one batched MERGE writes them all."""
    return [
        event_properties(source, entity.id, kind, index, step_index, step)
        for entity in entities
        for kind, index, procedure in _derivations(entity)
        for step_index, step in enumerate(procedure.steps)
    ]


def has_procedure_pairs(
    entities: list[models.Entity], source: str
) -> list[dict]:
    """The ``{entity, procedure}`` uuid pairs for the ``:HAS_PROCEDURE`` edges — one per (entity,
    derivation), hanging each procedure off the declarative entity it derives."""
    return [
        {
            'entity': entity_uuid(source, entity.id),
            'procedure': procedure_uuid(source, entity.id, kind, index),
        }
        for entity in entities
        for kind, index, _ in _derivations(entity)
    ]


def first_pairs(entities: list[models.Entity], source: str) -> list[dict]:
    """The ``{procedure, event}`` uuid pairs for the ``:FIRST`` edges — each procedure to its opening
    step. Only procedures with at least one step appear; a stepless one (a bare solution) has none."""
    pairs: list[dict] = []
    for entity in entities:
        for kind, index, procedure in _derivations(entity):
            if procedure.steps:
                pairs.append(
                    {
                        'procedure': procedure_uuid(
                            source, entity.id, kind, index
                        ),
                        'event': event_uuid(source, entity.id, kind, index, 0),
                    }
                )
    return pairs


def then_pairs(entities: list[models.Entity], source: str) -> list[dict]:
    """The ``{from, to}`` uuid pairs for the ``:THEN`` chain — consecutive steps within each procedure.
    A procedure of fewer than two steps contributes none; the chain never crosses procedures."""
    pairs: list[dict] = []
    for entity in entities:
        for kind, index, procedure in _derivations(entity):
            for step_index in range(len(procedure.steps) - 1):
                pairs.append(
                    {
                        'from': event_uuid(
                            source, entity.id, kind, index, step_index
                        ),
                        'to': event_uuid(
                            source, entity.id, kind, index, step_index + 1
                        ),
                    }
                )
    return pairs
