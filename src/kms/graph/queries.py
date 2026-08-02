"""
Read-side queries over the graph — the counterpart to ``writer``.

``writer`` is the write half of the I/O layer (persist nodes, statements,
procedures, equations, variables). This module is the read half: named,
parameterised queries that return plain data, used by construction passes
that must look back at the graph — the concept pass's search-before-create
dedup, the relation pass's living-schema canonicalisation, and anything
else that needs to see what is already there.

Every function takes the injected ``session_factory`` (the same callable
the persister receives — an async context manager with
``run(query, **params)``) and returns plain values, never driver objects,
so the queries are unit-testable against a scripted session and the neo4j
driver stays quarantined in ``db``. Scoping by book uses the raw source
identity and maps it through ``nodes.source_uuid``, matching ``writer``.
"""

from collections.abc import Callable
from dataclasses import dataclass

from kms.graph import nodes

# The concept tier's vertex label. The concept mapping module (which owns
# this label, its uuid scheme, and its property map) lands with the concept
# pass; until then the label lives here so the dedup query has a target.
CONCEPT_LABEL = 'Concept'


@dataclass(slots=True)
class ConceptDatum:
    """One existing concept read back for dedup.

    ``description`` is the grounding text the concept was built from (the
    representative fact or typed meaning), which is what gets embedded for
    similarity matching against new mentions.
    """

    uuid: str
    name: str
    description: str | None = None
    source: str | None = None


async def relation_types(
    session_factory: Callable,
    *,
    source: str | None = None,
) -> list[str]:
    """Every distinct relation type present in the graph.

    The living schema's ground truth: relation types are whatever edges the
    graph actually contains. The relation pass's write-time canonicalisation
    compares a proposed type against this list and reuses the existing name
    when one matches.

    Args:
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
        source: The book identity to scope to, or None for the whole graph.

    Returns:
        The distinct relation type names, sorted.
    """
    params: dict[str, str] = {}
    if source is not None:
        params['source'] = nodes.source_uuid(source)
        cypher = (
            'MATCH (n {source: $source})-[r]->() '
            'RETURN DISTINCT type(r) AS type ORDER BY type'
        )
    else:
        cypher = (
            'MATCH ()-[r]->() RETURN DISTINCT type(r) AS type ORDER BY type'
        )

    async with session_factory() as session:
        result = await session.run(cypher, **params)
        records = await result.all()
    return [record['type'] for record in records]


async def existing_concepts(
    session_factory: Callable,
    *,
    source: str | None = None,
) -> list[ConceptDatum]:
    """Every concept currently in the graph.

    The search-before-create target for the concept pass: before a new
    concept is minted from a fact mention, the pass compares the mention
    against these (embedding the ``description``/``name`` on the fly) and
    merges into an existing one when similar enough.

    Args:
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
        source: The book identity to scope to, or None for all books (the
            cross-book dedup case).

    Returns:
        The concepts, sorted by name.
    """
    params: dict[str, str] = {}
    if source is not None:
        params['source'] = nodes.source_uuid(source)
        cypher = (
            f'MATCH (c:{CONCEPT_LABEL} {{source: $source}}) '
            f'RETURN c.uuid AS uuid, c.name AS name, '
            f'c.description AS description, c.source AS source '
            f'ORDER BY c.name'
        )
    else:
        cypher = (
            f'MATCH (c:{CONCEPT_LABEL}) '
            f'RETURN c.uuid AS uuid, c.name AS name, '
            f'c.description AS description, c.source AS source '
            f'ORDER BY c.name'
        )

    async with session_factory() as session:
        result = await session.run(cypher, **params)
        records = await result.all()
    return [
        ConceptDatum(
            uuid=record['uuid'],
            name=record['name'],
            description=record.get('description'),
            source=record.get('source'),
        )
        for record in records
    ]
