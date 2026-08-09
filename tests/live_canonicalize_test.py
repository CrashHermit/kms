"""Live test: full-rebuild canonicalization against Neo4j.

Reads all :Entity and :Predicate spokes from Neo4j, clusters them,
synthesises definitions, deletes the old canonical layer, and writes
fresh :EntityHub/:PredicateHub + :Definition + :CANONICAL edges.

No incremental merge, no cross-batch adjudication — full rebuild
from spokes.

Requires NEO4J_URI/USERNAME/PASSWORD and EMBEDDING_API_KEY (for
definition embedding) and DEEPSEEK_API_KEY (for definition synthesis
LLM calls).

Run from the repo root with:
    .venv/bin/python tests/live_canonicalize_test.py
"""

import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


async def main() -> None:
    from kms.core import llm
    from kms.graph import db
    from kms.ingestion.canonicalizer import rebuild

    if not db.is_configured():
        print('Neo4j not configured — stopping.')
        return

    language_model = llm.text_lm()

    def _sf():
        return db.session()

    threshold = 0.85

    try:
        result = await rebuild(
            threshold=threshold,
            language_model=language_model,
            session_factory=_sf,
            entity_kind=True,
            predicate_kind=True,
            rebuild_triplets=True,
        )

        print(f'\n{"=" * 60}')
        print('SUMMARY')
        print('=' * 60)

        for kind in ('entity', 'predicate'):
            info = result.get(kind, {})
            print(f'\n  {kind}:')
            print(f'    Clusters: {info.get("clusters", 0)}')
            print(f'    Spokes:   {info.get("spokes", 0)}')

        # Quick verification query
        print(f'\n{"=" * 60}')
        print('NEO4J CHECK')
        print('=' * 60)

        async with db.session() as s:
            for label in (
                'EntityHub', 'PredicateHub', 'Definition',
                'TripletHub', 'FactHub',
            ):
                r = await s.run(
                    f'MATCH (n:`{label}`) RETURN count(n) AS cnt'
                )
                cnt = (await r.single())['cnt']
                print(f'  :{label:<16} {cnt}')

            r = await s.run(
                'MATCH ()-[r:CANONICAL]->() RETURN count(r) AS cnt'
            )
            cnt = (await r.single())['cnt']
            print(f'  [:CANONICAL]          {cnt}')

            # Show a few hubs with their definitions
            r = await s.run(
                'MATCH (h:EntityHub)-[:HAS_DEFINITION]->(d:Definition) '
                'RETURN h.display_name AS name, d.text AS definition '
                'LIMIT 5'
            )
            print('\n  Sample entity hubs:')
            async for rec in r:
                print(f'    {rec["name"]}: '
                      f'{rec["definition"][:100]}...')

    finally:
        await db.close_driver()

    print(f'\n{"=" * 60}')
    print('Done.')
    print('=' * 60)


if __name__ == '__main__':
    asyncio.run(main())
