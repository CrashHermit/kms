"""
Graph representation of the procedural layer — the ``:Procedure`` containers and their ``:Event``
step chains (see ``docs/UNIFIED-KG.md``, "Edges (math-first)").

A Theorem's ``proofs`` and a Problem's ``solutions`` are *derivations* — the procedural half of the
bi-modal graph — so they are reified out of the entity (where the older schema kept them as JSON-string
blobs) into real graph structure: one ``:Procedure`` per proof/solution, hung off its entity via
``(:Entity)-[:HAS_PROCEDURE]->(:Procedure)``, and one ``:Event`` per ``bodylist`` step, threaded
``(:Procedure)-[:FIRST]->(:Event)-[:THEN]->(:Event)-…``. The entity keeps only its *statement*
structure (its own ``bodylist``); the doing lives here. This mirrors ``graph.nodes``/``graph.entities``:
the vocabulary is ``core.ProcedureType`` (it invents no kinds) and it stays free of the neo4j driver
(pure mapping) — the driver lives in ``graph.db``, the writes in ``graph.writer``.

Representation: every procedure carries the base ``:Procedure`` label AND its per-kind label
(``:Procedure:Proof`` / ``:Procedure:Solution``); math-first the kind is a label, like ``:Entity:Theorem``
(``docs/UNIFIED-KG.md``, the math-first note). Events carry only ``:Event`` (the step's ``action`` — the
tactic role — is a property, not a label, since roles are open). A solution has no ``bodylist`` in the
model, so a ``:Procedure:Solution`` carries its ``contents`` but no event chain; proofs get both.

Identity: deterministic uuid5s, disjoint from node/entity/hub uuids by their segment. A procedure keys
on ``(source, entity id, kind, procedure index)`` — kind keeps a Thm's proof #0 distinct from a Prob's
solution #0 on the (rare) shared entity — and an event on the procedure key plus its step index. So
re-persisting a book MERGEs onto the same procedure/event vertices instead of duplicating them.

Event provenance: a ``bodylist`` step is a content slice, not tied to a specific source node id, so no
``(:Event)-[:DERIVED_FROM]->(:Node)`` edge is drawn here — provenance is reachable transitively
(event → procedure → entity → its member ``:Node`` s). A precise step→node link is later work.
"""

from collections.abc import Iterator
from uuid import NAMESPACE_URL, uuid5

from kms.core.models import BodySegment, Entity, ProcedureType
from kms.graph.entities import entity_uuid
from kms.graph.nodes import source_uuid

PROCEDURE_LABEL = "Procedure"
EVENT_LABEL = "Event"


def procedure_uuid(source: str, entity_id: int, kind: str, index: int) -> str:
    """Stable, deterministic vertex key for a procedure: uuid5 over ``(source, entity id, kind,
    procedure index)``. The ``procedure#`` segment keeps it disjoint from node/entity/hub uuids; kind
    disambiguates a proof from a solution on the same entity id."""
    return uuid5(NAMESPACE_URL, f"{source}#procedure#{entity_id}#{kind}#{index}").hex


def event_uuid(source: str, entity_id: int, kind: str, proc_index: int, step_index: int) -> str:
    """Stable, deterministic vertex key for one procedure step: the procedure key plus the step's
    position. The ``event#`` segment keeps it disjoint from every other uuid namespace."""
    return uuid5(NAMESPACE_URL, f"{source}#event#{entity_id}#{kind}#{proc_index}#{step_index}").hex


def procedure_label(kind: str) -> str:
    """The per-kind label for a procedure (``ProcedureType.PROOF`` -> ``"Proof"``), applied ALONGSIDE
    the base ``:Procedure`` label. The ProcedureType values are single lowercase words, so capitalizing
    yields a valid Neo4j label."""
    return kind.capitalize()


def proof_events(entity: Entity, source: str) -> Iterator[tuple[str, BodySegment]]:
    """Yield ``(event_uuid, step)`` for every step of every proof of the entity — the reified
    ``:Event`` identities. The single place the ``:Event`` key is derived from the (proof index, step
    index) scheme, so a consumer (e.g. the step-level ``:USES`` builder) reuses it instead of
    reconstructing the same uuid."""
    for proc_index, proof in enumerate(entity.proofs):
        for step_index, step in enumerate(proof.bodylist):
            yield (
                event_uuid(source, entity.id, ProcedureType.PROOF.value, proc_index, step_index),
                step,
            )


def _derivations(entity: Entity) -> list[tuple[str, int, list[str], list[BodySegment]]]:
    """The entity's procedures as ``(kind, index, contents, bodylist)`` tuples, in a stable order:
    proofs first (Theorem-only), then solutions (Problem-only). A solution carries no bodylist, so its
    step list is empty. The index is per-kind (proof #0, #1, …; solution #0, …); the uuid folds in the
    kind, so the two index spaces don't collide."""
    rows: list[tuple[str, int, list[str], list[BodySegment]]] = []
    for i, proof in enumerate(entity.proofs):
        rows.append((ProcedureType.PROOF.value, i, proof.contents, proof.bodylist))
    for i, solution in enumerate(entity.solutions):
        rows.append((ProcedureType.SOLUTION.value, i, solution.contents, []))
    return rows


def procedure_properties(
    source: str, entity_id: int, kind: str, index: int, contents: list[str]
) -> dict:
    """The Neo4j property map for one procedure: its stable uuid, the source link, the ``type`` (kind),
    its per-kind ``index``, and ``contents`` as a native string array. Empty contents is dropped."""
    props = {
        "uuid": procedure_uuid(source, entity_id, kind, index),
        "source": source_uuid(source),
        "type": kind,
        "index": index,
        "contents": contents or None,
    }
    return {key: value for key, value in props.items() if value is not None}


def event_properties(
    source: str, entity_id: int, kind: str, proc_index: int, step_index: int, step: BodySegment
) -> dict:
    """The Neo4j property map for one procedure step: its stable uuid, the source link, the ``action``
    (the tactic role, a property since roles are open), the step ``text``, and its ordinal ``index``."""
    return {
        "uuid": event_uuid(source, entity_id, kind, proc_index, step_index),
        "source": source_uuid(source),
        "action": step.action,
        "text": step.description,
        "index": step_index,
    }


def procedure_batches(entities: list[Entity], source: str) -> dict[str, list[dict]]:
    """Group every procedure's property map by its per-kind label, so each label is one batched
    MERGE (mirrors ``entity_batches``)."""
    batches: dict[str, list[dict]] = {}
    for entity in entities:
        for kind, index, contents, _ in _derivations(entity):
            batches.setdefault(procedure_label(kind), []).append(
                procedure_properties(source, entity.id, kind, index, contents)
            )
    return batches


def event_rows(entities: list[Entity], source: str) -> list[dict]:
    """Every step's property map across the overlay, one flat list — events carry a single ``:Event``
    label (their ``action`` is a property, not a per-type label), so one batched MERGE writes them all."""
    return [
        event_properties(source, entity.id, kind, index, step_index, step)
        for entity in entities
        for kind, index, _, bodylist in _derivations(entity)
        for step_index, step in enumerate(bodylist)
    ]


def has_procedure_pairs(entities: list[Entity], source: str) -> list[dict]:
    """The ``{entity, procedure}`` uuid pairs for the ``:HAS_PROCEDURE`` edges — one per (entity,
    derivation), hanging each procedure off the declarative entity it derives."""
    return [
        {
            "entity": entity_uuid(source, entity.id),
            "procedure": procedure_uuid(source, entity.id, kind, index),
        }
        for entity in entities
        for kind, index, _, _ in _derivations(entity)
    ]


def first_pairs(entities: list[Entity], source: str) -> list[dict]:
    """The ``{procedure, event}`` uuid pairs for the ``:FIRST`` edges — each procedure to its opening
    step. Only procedures with at least one step (proofs) appear; a stepless solution has none."""
    pairs: list[dict] = []
    for entity in entities:
        for kind, index, _, bodylist in _derivations(entity):
            if bodylist:
                pairs.append(
                    {
                        "procedure": procedure_uuid(source, entity.id, kind, index),
                        "event": event_uuid(source, entity.id, kind, index, 0),
                    }
                )
    return pairs


def then_pairs(entities: list[Entity], source: str) -> list[dict]:
    """The ``{from, to}`` uuid pairs for the ``:THEN`` chain — consecutive steps within each procedure.
    A procedure of fewer than two steps contributes none; the chain never crosses procedures."""
    pairs: list[dict] = []
    for entity in entities:
        for kind, index, _, bodylist in _derivations(entity):
            for step_index in range(len(bodylist) - 1):
                pairs.append(
                    {
                        "from": event_uuid(source, entity.id, kind, index, step_index),
                        "to": event_uuid(source, entity.id, kind, index, step_index + 1),
                    }
                )
    return pairs
