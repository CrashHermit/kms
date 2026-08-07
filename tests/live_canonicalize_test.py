"""Live test: DIAL-KG–style four-phase canonicalization against Neo4j.

Runs on the combinatorics graph-theory page embeddings already persisted
by live_neo4j_pipeline_test.py.  Reads uncanonicalized :Entity and
:Predicate spokes from Neo4j, clusters them, aligns against any existing
hubs (first run = all new hubs), and writes :EntityHub/:PredicateHub +
:Definition + :CANONICAL edges.

Requires NEO4J_URI/USERNAME/PASSWORD and OPENROUTER_API_KEY (for
definition embedding) and DEEPSEEK_API_KEY (for adjudication +
definition synthesis LLM calls).

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
    from kms.ingestion.canonicalizer import run_canonicalization

    if not db.is_configured():
        print('Neo4j not configured — stopping.')
        return

    source = 'combinatorics_graph_theory_page'
    language_model = llm.text_lm()

    def _sf():
        return db.session()

    threshold = 0.85

    try:
        result = await run_canonicalization(
            source=source,
            threshold=threshold,
            language_model=language_model,
            session_factory=_sf,
            cross_source=False,
            entity_kind=True,
            predicate_kind=True,
        )

        print(f'\n{"=" * 60}')
        print('SUMMARY')
        print('=' * 60)

        for kind in ('entity', 'predicate'):
            info = result.get(kind, {})
            print(f'\n  {kind}:')
            print(f'    Merged into existing hubs: {info["merged"]}')
            print(f'    New hubs created:         {info["new_hubs"]}')
            print(f'    Review-flagged:           {info["review_flagged"]}')
            print(f'    Total spokes processed:   {info["total_spokes"]}')

        # Quick verification query
        print(f'\n{"=" * 60}')
        print('NEO4J CHECK')
        print('=' * 60)

        async with db.session() as s:
            for label in ('EntityHub', 'PredicateHub', 'Definition'):
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
