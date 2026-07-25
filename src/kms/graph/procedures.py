"""
Graph representation of the procedural layer — the ``:Procedure`` containers and their ``:Act``
step chains (see ``docs/SCHEMA.md``).

A block's derivations are *procedures* — the procedural half of the bi-modal graph — so they are
reified out of the entity (where an older schema kept them as JSON-string blobs) into real graph
structure: one ``:Procedure`` per derivation, hung off its entity via
``(:Entity)-[:HAS_PROCEDURE]->(:Procedure)``, and one ``:Act`` per step, threaded
``(:Procedure)-[:FIRST]->(:Act)-[:THEN]->(:Act)-…``. The entity keeps only its statement; the doing
lives here. This mirrors ``graph.nodes``/``graph.entities``: pure mapping, free of the neo4j driver
(the driver lives in ``graph.db``, the writes in ``graph.writer``).

Representation: a procedure carries the bare ``:Procedure`` label, with NO per-kind label and no
``type`` property — proof / solution / derivation is derivable from the owning entity's ``type``,
so storing it would duplicate a neighbour's fact (``docs/SCHEMA.md``, principle 5). An ``:Act``
likewise carries only its verbatim ``text`` and its ordinal ``index``: the old ``action`` tactic
role was AutoMathKG's closed nine-value taxonomy, was written and never read, and the concept layer
supersedes it with open, cross-corpus-linkable tags.

Decomposition is UNIVERSAL — every procedure has steps, whatever it derives. The old schema
restricted its step list to Theorems and Definitions, which left every solution stepless.

Identity: deterministic uuid5s, disjoint from node/entity uuids by their segment. A procedure keys
on ``(source, entity id, procedure index)`` and an act on the procedure key plus its step index. So
re-persisting a book MERGEs onto the same procedure/act vertices instead of duplicating them.

Provenance: a procedure is found as its own span, so it carries ``members`` and gets real
``(:Procedure)-[:DERIVED_FROM]->(:Node)`` edges. An ``:Act`` does not: its text is a sub-node slice
that need not align to node boundaries, so its provenance is transitive (act → procedure → node).
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.entities import entity_uuid

PROCEDURE_LABEL = 'Procedure'
ACT_LABEL = 'Act'


def procedure_uuid(source: str, entity_id: int, index: int) -> str:
    """Stable, deterministic vertex key for a procedure: uuid5 over ``(source, entity id,
    procedure index)``. The ``procedure#`` segment keeps it disjoint from every other uuid
    namespace. An entity's procedures are indexed from 0 in document order."""
    return uuid5(NAMESPACE_URL, f'{source}#procedure#{entity_id}#{index}').hex


def act_uuid(
    source: str, entity_id: int, procedure_index: int, step_index: int
) -> str:
    """Stable, deterministic vertex key for one procedure step: the procedure key plus the step's
    position. The ``act#`` segment keeps it disjoint from every other uuid namespace."""
    return uuid5(
        NAMESPACE_URL,
        f'{source}#act#{entity_id}#{procedure_index}#{step_index}',
    ).hex


def procedure_properties(
    source: str, entity_id: int, procedure: models.Procedure
) -> dict:
    """The Neo4j property map for one procedure: its stable uuid, the source link, its ordinal
    ``index``, and ``contents`` as a native string array. Empty contents is dropped. No ``type``
    — see the module docstring."""
    props = {
        'uuid': procedure_uuid(source, entity_id, procedure.index),
        'source': nodes.source_uuid(source),
        'index': procedure.index,
        'contents': procedure.contents or None,
    }
    return {key: value for key, value in props.items() if value is not None}


def act_properties(
    source: str,
    entity_id: int,
    procedure_index: int,
    step_index: int,
    text: str,
) -> dict:
    """The Neo4j property map for one procedure step: its stable uuid, the source link, the
    verbatim step ``text``, and its ordinal ``index``."""
    return {
        'uuid': act_uuid(source, entity_id, procedure_index, step_index),
        'source': nodes.source_uuid(source),
        'text': text,
        'index': step_index,
    }


def procedure_rows(entities: list[models.Entity], source: str) -> list[dict]:
    """Every procedure's property map across the overlay, one flat list. Procedures carry a single
    ``:Procedure`` label, so one batched MERGE writes them all."""
    return [
        procedure_properties(source, entity.id, procedure)
        for entity in entities
        for procedure in entity.procedures
    ]


def act_rows(entities: list[models.Entity], source: str) -> list[dict]:
    """Every step's property map across the overlay, one flat list — acts carry a single ``:Act``
    label, so one batched MERGE writes them all."""
    return [
        act_properties(source, entity.id, procedure.index, step_index, text)
        for entity in entities
        for procedure in entity.procedures
        for step_index, text in enumerate(procedure.steps)
    ]


def has_procedure_pairs(
    entities: list[models.Entity], source: str
) -> list[dict]:
    """The ``{entity, procedure}`` uuid pairs for the ``:HAS_PROCEDURE`` edges — one per (entity,
    procedure), hanging each derivation off the block it derives."""
    return [
        {
            'entity': entity_uuid(source, entity.id),
            'procedure': procedure_uuid(source, entity.id, procedure.index),
        }
        for entity in entities
        for procedure in entity.procedures
    ]


def procedure_member_pairs(
    entities: list[models.Entity], source: str
) -> list[dict]:
    """The ``{procedure, node}`` uuid pairs for the procedures' ``:DERIVED_FROM`` edges — one per
    (procedure, member), so a derivation links to every source chunk it was built from. Possible
    because the group finder detects a procedure as its own span, giving it member node ids."""
    return [
        {
            'procedure': procedure_uuid(source, entity.id, procedure.index),
            'node': nodes.node_uuid(source, member),
        }
        for entity in entities
        for procedure in entity.procedures
        for member in procedure.members
    ]


def first_pairs(entities: list[models.Entity], source: str) -> list[dict]:
    """The ``{procedure, act}`` uuid pairs for the ``:FIRST`` edges — each procedure to its opening
    step. A procedure that decomposed into nothing contributes none."""
    return [
        {
            'procedure': procedure_uuid(source, entity.id, procedure.index),
            'act': act_uuid(source, entity.id, procedure.index, 0),
        }
        for entity in entities
        for procedure in entity.procedures
        if procedure.steps
    ]


def then_pairs(entities: list[models.Entity], source: str) -> list[dict]:
    """The ``{from, to}`` uuid pairs for the ``:THEN`` chain — consecutive steps within each
    procedure. A procedure of fewer than two steps contributes none; the chain never crosses
    procedures."""
    return [
        {
            'from': act_uuid(source, entity.id, procedure.index, step_index),
            'to': act_uuid(source, entity.id, procedure.index, step_index + 1),
        }
        for entity in entities
        for procedure in entity.procedures
        for step_index in range(len(procedure.steps) - 1)
    ]
