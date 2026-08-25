"""Neo4j constraint and index statements for the graph schema."""

from collections.abc import Callable

from kms import config
from kms.graph import (
    entities,
    hubs,
    instructions,
    learning,
    local_procedure_hubs,
    local_statement_hubs,
    names,
    nodes,
    predicates,
    procedures,
    statements,
    triplets,
)


def schema_statements() -> list[str]:
    """Returns the idempotent constraint and index statements.

    Every vertex label gets a unique-uuid constraint and the vector
    labels get configured-dimension cosine similarity indexes.
    """
    dimension = config.get_settings().embeddings.dimension
    return [
        'MATCH (h:StatementHub) REMOVE h:StatementHub '
        'SET h:LocalStatementHub',
        'MATCH (h:ProcedureHub) REMOVE h:ProcedureHub '
        'SET h:LocalProcedureHub',
        'MATCH (h:MetaStatementHub) REMOVE h:MetaStatementHub '
        'SET h:GlobalStatementHub',
        'MATCH (h:MetaProcedureHub) REMOVE h:MetaProcedureHub '
        'SET h:GlobalProcedureHub',
        'MATCH (h:EntityHub) REMOVE h:EntityHub SET h:LocalEntityHub',
        'MATCH (h:PredicateHub) REMOVE h:PredicateHub '
        'SET h:LocalPredicateHub',
        'MATCH (h:MetaEntityHub) REMOVE h:MetaEntityHub '
        'SET h:GlobalEntityHub',
        'MATCH (h:MetaPredicateHub) REMOVE h:MetaPredicateHub '
        'SET h:GlobalPredicateHub',
        'MATCH (h:TripletHub) REMOVE h:TripletHub '
        'SET h:LocalTripletHub',
        'MATCH (h:MetaTripletHub) REMOVE h:MetaTripletHub '
        'SET h:GlobalTripletHub',
        'MATCH (h:EntityNameHub) REMOVE h:EntityNameHub '
        'SET h:LocalEntityNameHub',
        'MATCH (h:PredicateNameHub) REMOVE h:PredicateNameHub '
        'SET h:LocalPredicateNameHub',
        'MATCH (h:MetaEntityNameHub) REMOVE h:MetaEntityNameHub '
        'SET h:GlobalEntityNameHub',
        'MATCH (h:MetaPredicateNameHub) REMOVE h:MetaPredicateNameHub '
        'SET h:GlobalPredicateNameHub',
        f'CREATE CONSTRAINT node_uuid IF NOT EXISTS '
        f'FOR (n:{nodes.NODE_LABEL}) REQUIRE n.uuid IS UNIQUE',
        f'CREATE CONSTRAINT source_uuid IF NOT EXISTS '
        f'FOR (s:{nodes.SOURCE_LABEL}) REQUIRE s.uuid IS UNIQUE',
        f'CREATE CONSTRAINT statement_uuid IF NOT EXISTS '
        f'FOR (s:{statements.STATEMENT_LABEL}) REQUIRE s.uuid IS UNIQUE',
        f'CREATE CONSTRAINT procedure_uuid IF NOT EXISTS '
        f'FOR (p:{procedures.PROCEDURE_LABEL}) REQUIRE p.uuid IS UNIQUE',
        f'CREATE CONSTRAINT instruction_uuid IF NOT EXISTS '
        f'FOR (i:{instructions.INSTRUCTION_LABEL}) REQUIRE i.uuid IS UNIQUE',
        f'CREATE CONSTRAINT triplet_uuid IF NOT EXISTS '
        f'FOR (t:{triplets.TRIPLET_LABEL}) REQUIRE t.uuid IS UNIQUE',
        f'CREATE CONSTRAINT entity_uuid IF NOT EXISTS '
        f'FOR (e:{entities.ENTITY_LABEL}) REQUIRE e.uuid IS UNIQUE',
        f'CREATE CONSTRAINT predicate_uuid IF NOT EXISTS '
        f'FOR (p:{predicates.PREDICATE_LABEL}) REQUIRE p.uuid IS UNIQUE',
        f'CREATE CONSTRAINT entity_name_uuid IF NOT EXISTS '
        f'FOR (n:{names.ENTITY_NAME_LABEL}) REQUIRE n.uuid IS UNIQUE',
        f'CREATE CONSTRAINT predicate_name_uuid IF NOT EXISTS '
        f'FOR (n:{names.PREDICATE_NAME_LABEL}) REQUIRE n.uuid IS UNIQUE',
        f'CREATE CONSTRAINT entity_name_hub_uuid IF NOT EXISTS '
        f'FOR (h:{names.ENTITY_NAME_HUB_LABEL}) REQUIRE h.uuid IS UNIQUE',
        f'CREATE CONSTRAINT predicate_name_hub_uuid IF NOT EXISTS '
        f'FOR (h:{names.PREDICATE_NAME_HUB_LABEL}) REQUIRE h.uuid IS UNIQUE',
        f'CREATE INDEX node_source IF NOT EXISTS '
        f'FOR (n:{nodes.NODE_LABEL}) ON (n.source)',
        f'CREATE INDEX entity_source IF NOT EXISTS '
        f'FOR (e:{entities.ENTITY_LABEL}) ON (e.source)',
        f'CREATE INDEX predicate_source IF NOT EXISTS '
        f'FOR (p:{predicates.PREDICATE_LABEL}) ON (p.source)',
        f'CREATE INDEX entity_name_text IF NOT EXISTS '
        f'FOR (n:{names.ENTITY_NAME_LABEL}) ON (n.normalized_text)',
        f'CREATE INDEX predicate_name_text IF NOT EXISTS '
        f'FOR (n:{names.PREDICATE_NAME_LABEL}) ON (n.normalized_text)',
        f'CREATE INDEX entity_name_hub_source IF NOT EXISTS '
        f'FOR (h:{names.ENTITY_NAME_HUB_LABEL}) ON (h.source)',
        f'CREATE INDEX entity_name_hub_form IF NOT EXISTS '
        f'FOR (h:{names.ENTITY_NAME_HUB_LABEL}) ON (h.canonical_form)',
        f'CREATE INDEX entity_name_hub_normalized IF NOT EXISTS '
        f'FOR (h:{names.ENTITY_NAME_HUB_LABEL}) ON (h.normalized_form)',
        f'CREATE INDEX predicate_name_hub_source IF NOT EXISTS '
        f'FOR (h:{names.PREDICATE_NAME_HUB_LABEL}) ON (h.source)',
        f'CREATE INDEX predicate_name_hub_form IF NOT EXISTS '
        f'FOR (h:{names.PREDICATE_NAME_HUB_LABEL}) ON (h.canonical_form)',
        f'CREATE INDEX predicate_name_hub_normalized IF NOT EXISTS '
        f'FOR (h:{names.PREDICATE_NAME_HUB_LABEL}) ON (h.normalized_form)',
        f'CREATE CONSTRAINT meta_entity_name_hub_uuid IF NOT EXISTS '
        f'FOR (h:{names.META_ENTITY_NAME_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE INDEX meta_entity_name_hub_form IF NOT EXISTS '
        f'FOR (h:{names.META_ENTITY_NAME_HUB_LABEL}) ON (h.canonical_form)',
        f'CREATE INDEX meta_entity_name_hub_normalized IF NOT EXISTS '
        f'FOR (h:{names.META_ENTITY_NAME_HUB_LABEL}) ON (h.normalized_form)',
        f'CREATE CONSTRAINT meta_predicate_name_hub_uuid IF NOT EXISTS '
        f'FOR (h:{names.META_PREDICATE_NAME_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE INDEX meta_predicate_name_hub_form IF NOT EXISTS '
        f'FOR (h:{names.META_PREDICATE_NAME_HUB_LABEL}) ON (h.canonical_form)',
        f'CREATE INDEX meta_predicate_name_hub_normalized IF NOT EXISTS '
        f'FOR (h:{names.META_PREDICATE_NAME_HUB_LABEL}) ON (h.normalized_form)',
        f'CREATE VECTOR INDEX node_content IF NOT EXISTS '
        f'FOR (n:{nodes.NODE_LABEL}) ON (n.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE INDEX statement_source IF NOT EXISTS '
        f'FOR (s:{statements.STATEMENT_LABEL}) ON (s.source)',
        f'CREATE VECTOR INDEX statement_embedding IF NOT EXISTS '
        f'FOR (s:{statements.STATEMENT_LABEL}) ON (s.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE INDEX triplet_source IF NOT EXISTS '
        f'FOR (t:{triplets.TRIPLET_LABEL}) ON (t.source)',
        f'CREATE VECTOR INDEX entity_embedding IF NOT EXISTS '
        f'FOR (e:{entities.ENTITY_LABEL}) ON (e.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE VECTOR INDEX predicate_embedding IF NOT EXISTS '
        f'FOR (p:{predicates.PREDICATE_LABEL}) ON (p.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT entity_hub_uuid IF NOT EXISTS '
        f'FOR (h:{hubs.ENTITY_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE INDEX entity_hub_source IF NOT EXISTS '
        f'FOR (h:{hubs.ENTITY_HUB_LABEL}) ON (h.source)',
        f'CREATE VECTOR INDEX entity_hub_embedding IF NOT EXISTS '
        f'FOR (h:{hubs.ENTITY_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT predicate_hub_uuid IF NOT EXISTS '
        f'FOR (h:{hubs.PREDICATE_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE INDEX predicate_hub_source IF NOT EXISTS '
        f'FOR (h:{hubs.PREDICATE_HUB_LABEL}) ON (h.source)',
        f'CREATE VECTOR INDEX predicate_hub_embedding IF NOT EXISTS '
        f'FOR (h:{hubs.PREDICATE_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT meta_entity_hub_uuid IF NOT EXISTS '
        f'FOR (h:{hubs.META_ENTITY_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE VECTOR INDEX meta_entity_hub_embedding IF NOT EXISTS '
        f'FOR (h:{hubs.META_ENTITY_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT meta_predicate_hub_uuid IF NOT EXISTS '
        f'FOR (h:{hubs.META_PREDICATE_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE VECTOR INDEX meta_predicate_hub_embedding IF NOT EXISTS '
        f'FOR (h:{hubs.META_PREDICATE_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT triplet_hub_uuid IF NOT EXISTS '
        f'FOR (h:{hubs.TRIPLET_HUB_LABEL}) REQUIRE h.uuid IS UNIQUE',
        f'CREATE INDEX triplet_hub_source IF NOT EXISTS '
        f'FOR (h:{hubs.TRIPLET_HUB_LABEL}) ON (h.source)',
        f'CREATE VECTOR INDEX triplet_hub_embedding IF NOT EXISTS '
        f'FOR (h:{hubs.TRIPLET_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT meta_triplet_hub_uuid IF NOT EXISTS '
        f'FOR (h:{hubs.META_TRIPLET_HUB_LABEL}) REQUIRE h.uuid IS UNIQUE',
        f'CREATE VECTOR INDEX meta_triplet_hub_embedding IF NOT EXISTS '
        f'FOR (h:{hubs.META_TRIPLET_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE VECTOR INDEX procedure_embedding IF NOT EXISTS '
        f'FOR (p:{procedures.PROCEDURE_LABEL}) ON (p.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT global_statement_hub_uuid IF NOT EXISTS '
        f'FOR (h:{local_statement_hubs.GLOBAL_STATEMENT_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE VECTOR INDEX global_statement_hub_embedding IF NOT EXISTS '
        f'FOR (h:{local_statement_hubs.GLOBAL_STATEMENT_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT global_procedure_hub_uuid IF NOT EXISTS '
        f'FOR (h:{local_procedure_hubs.GLOBAL_PROCEDURE_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE VECTOR INDEX global_procedure_hub_embedding IF NOT EXISTS '
        f'FOR (h:{local_procedure_hubs.GLOBAL_PROCEDURE_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT local_statement_hub_uuid IF NOT EXISTS '
        f'FOR (h:{local_statement_hubs.LOCAL_STATEMENT_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE INDEX local_statement_hub_source IF NOT EXISTS '
        f'FOR (h:{local_statement_hubs.LOCAL_STATEMENT_HUB_LABEL}) ON (h.source)',
        f'CREATE VECTOR INDEX local_statement_hub_embedding IF NOT EXISTS '
        f'FOR (h:{local_statement_hubs.LOCAL_STATEMENT_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT local_procedure_hub_uuid IF NOT EXISTS '
        f'FOR (h:{local_procedure_hubs.LOCAL_PROCEDURE_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE INDEX local_procedure_hub_source IF NOT EXISTS '
        f'FOR (h:{local_procedure_hubs.LOCAL_PROCEDURE_HUB_LABEL}) ON (h.source)',
        f'CREATE VECTOR INDEX local_procedure_hub_embedding IF NOT EXISTS '
        f'FOR (h:{local_procedure_hubs.LOCAL_PROCEDURE_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT card_uuid IF NOT EXISTS '
        f'FOR (c:{learning.CARD_LABEL}) REQUIRE c.uuid IS UNIQUE',
        f'CREATE INDEX card_hub_uuid IF NOT EXISTS '
        f'FOR (c:{learning.CARD_LABEL}) ON (c.hub_uuid)',
        f'CREATE INDEX card_due IF NOT EXISTS '
        f'FOR (c:{learning.CARD_LABEL}) ON (c.fsrs_due)',
        f'CREATE INDEX card_status IF NOT EXISTS '
        f'FOR (c:{learning.CARD_LABEL}) ON (c.status)',
        f'CREATE CONSTRAINT review_uuid IF NOT EXISTS '
        f'FOR (r:{learning.REVIEW_LABEL}) REQUIRE r.uuid IS UNIQUE',
        f'CREATE INDEX review_card_uuid IF NOT EXISTS '
        f'FOR (r:{learning.REVIEW_LABEL}) ON (r.card_uuid)',
        f'CREATE INDEX review_timestamp IF NOT EXISTS '
        f'FOR (r:{learning.REVIEW_LABEL}) ON (r.timestamp)',
    ]


_VECTOR_INDEX_NAMES = {
    'node_content',
    'statement_embedding',
    'entity_embedding',
    'predicate_embedding',
    'entity_hub_embedding',
    'predicate_hub_embedding',
    'meta_entity_hub_embedding',
    'meta_predicate_hub_embedding',
    'triplet_hub_embedding',
    'meta_triplet_hub_embedding',
    'procedure_embedding',
    'meta_statement_hub_embedding',
    'meta_procedure_hub_embedding',
    'statement_hub_embedding',
    'procedure_hub_embedding',
}


async def _drop_stale_vector_indexes(session) -> None:
    """Drop known vector indexes whose dimension no longer matches."""
    result = await session.run(
        'SHOW VECTOR INDEXES YIELD name, options '
        'RETURN name, options'
    )
    for row in await result.data():
        name = row.get('name')
        if name not in _VECTOR_INDEX_NAMES:
            continue
        options = row.get('options') or {}
        index_config = options.get('indexConfig') or {}
        dimensions = index_config.get('vector.dimensions')
        if dimensions != config.get_settings().embeddings.dimension:
            await session.run(f'DROP INDEX `{name}` IF EXISTS')

async def ensure_schema(session_factory: Callable) -> None:
    """Applies schema after removing stale known vector indexes."""
    async with session_factory() as session:
        await _drop_stale_vector_indexes(session)
        for statement in schema_statements():
            await session.run(statement)
