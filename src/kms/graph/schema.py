from collections.abc import Callable

from kms.graph import (
    entity_hubs,
    instructions,
    nodes,
    procedures,
    statements,
    triplets,
)


def schema_statements() -> list[str]:
    return [
        f'CREATE CONSTRAINT node_uuid IF NOT EXISTS '
        f'FOR (n:{nodes.NODE_LABEL}) REQUIRE n.uuid IS UNIQUE',
        f'CREATE CONSTRAINT source_uuid IF NOT EXISTS '
        f'FOR (s:{nodes.SOURCE_LABEL}) REQUIRE s.uuid IS UNIQUE',
        f'CREATE CONSTRAINT statement_uuid IF NOT EXISTS '
        f'FOR (s:{statements.STATEMENT_LABEL}) REQUIRE s.uuid IS UNIQUE',
        f'CREATE CONSTRAINT procedure_uuid IF NOT EXISTS '
        f'FOR (p:{procedures.PROCEDURE_LABEL}) REQUIRE p.uuid IS UNIQUE',
        f'CREATE CONSTRAINT step_uuid IF NOT EXISTS '
        f'FOR (s:{procedures.STEP_LABEL}) REQUIRE s.uuid IS UNIQUE',
        f'CREATE CONSTRAINT instruction_uuid IF NOT EXISTS '
        f'FOR (i:{instructions.INSTRUCTION_LABEL}) REQUIRE i.uuid IS UNIQUE',
        f'CREATE CONSTRAINT triplet_uuid IF NOT EXISTS '
        f'FOR (t:{triplets.TRIPLET_LABEL}) REQUIRE t.uuid IS UNIQUE',
        f'CREATE INDEX node_source IF NOT EXISTS '
        f'FOR (n:{nodes.NODE_LABEL}) ON (n.source)',
        f'CREATE VECTOR INDEX node_content IF NOT EXISTS '
        f'FOR (n:{nodes.NODE_LABEL}) ON (n.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: 1024, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE INDEX statement_source IF NOT EXISTS '
        f'FOR (s:{statements.STATEMENT_LABEL}) ON (s.source)',
        f'CREATE INDEX triplet_source IF NOT EXISTS '
        f'FOR (t:{triplets.TRIPLET_LABEL}) ON (t.source)',
        f'CREATE VECTOR INDEX triplet_embedding IF NOT EXISTS '
        f'FOR (t:{triplets.TRIPLET_LABEL}) ON (t.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: 1024, '
        f'`vector.similarity_function`: "cosine"}}}}',
        f'CREATE CONSTRAINT entity_hub_uuid IF NOT EXISTS '
        f'FOR (h:{entity_hubs.ENTITY_HUB_LABEL}) '
        f'REQUIRE h.uuid IS UNIQUE',
        f'CREATE VECTOR INDEX entity_hub_embedding IF NOT EXISTS '
        f'FOR (h:{entity_hubs.ENTITY_HUB_LABEL}) ON (h.embedding) '
        f'OPTIONS {{indexConfig: {{`vector.dimensions`: 1024, '
        f'`vector.similarity_function`: "cosine"}}}}',
    ]


async def ensure_schema(session_factory: Callable) -> None:
    async with session_factory() as session:
        for statement in schema_statements():
            await session.run(statement)
