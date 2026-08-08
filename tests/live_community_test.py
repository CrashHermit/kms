"""Live test: community detection + summary synthesis on the canonical
hub graph.

Runs after canonicalization.  Reads the EntityHub / PredicateHub graph
from Neo4j, finds connected communities, synthesises summary paragraphs,
and persists :Community nodes.

Requires NEO4J_URI/USERNAME/PASSWORD, DEEPSEEK_API_KEY (LLM summaries),
and OPENROUTER_API_KEY (summary embedding).

Run from the repo root with:
    .venv/bin/python tests/live_community_test.py
"""

import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


async def main() -> None:
    from kms.core import llm
    from kms.graph import db, schema, writer
    from kms.ingestion.community_builder import build_communities

    if not db.is_configured():
        print('Neo4j not configured — stopping.')
        return

    source = 'combinatorics_graph_theory_page'
    language_model = llm.text_lm()

    def _sf():
        return db.session()

    # Ensure schema includes Community constraint + index
    await schema.ensure_schema(_sf)

    try:
        communities = await build_communities(
            source=source,
            language_model=language_model,
            session_factory=_sf,
        )

        if communities:
            await writer.persist_communities(
                communities, source, session_factory=_sf
            )

        print(f'\n{"=" * 60}')
        print('RESULTS')
        print('=' * 60)
        print(f'{len(communities)} community(ies) built')

        # Verify
        async with db.session() as s:
            r = await s.run(
                'MATCH (c:Community) RETURN count(c) AS cnt'
            )
            cnt = (await r.single())['cnt']
            print(f'\n  :Community nodes: {cnt}')

            r = await s.run(
                'MATCH ()-[r:HAS_MEMBER]->() RETURN count(r) AS cnt'
            )
            cnt = (await r.single())['cnt']
            print(f'  :HAS_MEMBER edges:  {cnt}')

            r = await s.run(
                'MATCH ()-[r:COMMUNITY_EVIDENCE]->() RETURN count(r) AS cnt'
            )
            cnt = (await r.single())['cnt']
            print(f'  :COMMUNITY_EVIDENCE: {cnt}')

            # Show summaries
            r = await s.run(
                'MATCH (c:Community) '
                'RETURN c.summary_text AS summary, '
                'size(c.summary_embedding) AS dims '
                'LIMIT 5'
            )
            print('\n  Community summaries:')
            async for rec in r:
                print()
                print('    [' + str(rec['dims']) + '-dim embedding]')
                print('    ' + rec['summary'])

    finally:
        await db.close_driver()

    print(f'\n{"=" * 60}')
    print('Done.')
    print('=' * 60)


if __name__ == '__main__':
    asyncio.run(main())
