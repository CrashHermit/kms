"""Live test: TripletHub + FactHub construction over canonical hubs.

Reads the canonical hub graph, groups triplets by (subj_hub, pred_hub,
obj_hub), builds :TripletHub + :FactHub + edges, and verifies.

Run from the repo root with:
    .venv/bin/python tests/live_triplet_hub_test.py
"""

import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


async def main() -> None:
    from kms.graph import db, schema, writer
    from kms.ingestion.triplet_hub_builder import build_triplet_hubs

    if not db.is_configured():
        print('Neo4j not configured — stopping.')
        return

    source = 'combinatorics_graph_theory_page'

    def _sf():
        return db.session()

    await schema.ensure_schema(_sf)

    try:
        groups = await build_triplet_hubs(source=source, session_factory=_sf)

        if groups:
            await writer.persist_triplet_hubs(
                groups, source, session_factory=_sf
            )

        print(f'\n{"=" * 60}')
        print('RESULTS')
        print('=' * 60)

        async with db.session() as s:
            for label in ('TripletHub', 'FactHub'):
                r = await s.run(
                    f'MATCH (n:`{label}`) RETURN count(n) AS cnt'
                )
                cnt = (await r.single())['cnt']
                print(f'  :{label:<16} {cnt}')

            for edge in ('CANONICAL_SUBJECT', 'CANONICAL_PREDICATE',
                         'CANONICAL_OBJECT', 'HAS_FACT', 'SUPPORTED_BY'):
                r = await s.run(
                    f'MATCH ()-[r:`{edge}`]->() RETURN count(r) AS cnt'
                )
                cnt = (await r.single())['cnt']
                print(f'  [:{edge:<20}] {cnt}')

            # Show a few FactHub texts
            r = await s.run(
                'MATCH (fh:FactHub) '
                'RETURN fh.text AS text, '
                'size(fh.embedding) AS dims '
                'LIMIT 5'
            )
            print('\n  Sample assertions:')
            async for rec in r:
                print(f'    [{rec["dims"]}-dim] {rec["text"]}')

            # Spot-check: resolve one TripletHub to its hubs
            r = await s.run(
                'MATCH (th:TripletHub) '
                'MATCH (sh:EntityHub)-[:CANONICAL_SUBJECT]->(th) '
                'MATCH (ph:PredicateHub)-[:CANONICAL_PREDICATE]->(th) '
                'MATCH (oh:EntityHub)-[:CANONICAL_OBJECT]->(th) '
                'MATCH (th)-[:HAS_FACT]->(fh:FactHub) '
                'RETURN sh.display_name AS subj, '
                'ph.display_name AS pred, '
                'oh.display_name AS obj, '
                'fh.text AS assertion '
                'LIMIT 3'
            )
            print('\n  Resolved assertions:')
            async for rec in r:
                print(f'    ({rec["subj"]}) --[{rec["pred"]}]--> '
                      f'({rec["obj"]})')
                print(f'      → "{rec["assertion"]}"')

    finally:
        await db.close_driver()

    print(f'\n{"=" * 60}')
    print('Done.')
    print('=' * 60)


if __name__ == '__main__':
    asyncio.run(main())
