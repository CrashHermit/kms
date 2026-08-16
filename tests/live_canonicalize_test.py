import asyncio

from kms.construction import canonicalizer
from kms.construction.canonicalizer import rebuild
from kms.construction.triplet_extractor import (
    TripletNode,
    _FactExtractor,
    _TripletDecomposer,
)
from kms.core import embeddings, llm, models
from kms.graph import db, schema, writer

CONTENT = """\
Here both $G_2$ and $G_3$ are subgraphs of $G_1$. But only $G_2$ is an \
*induced* subgraph. Every edge in $G_1$ that connects vertices in $G_2$ is \
also an edge in $G_2$. In $G_3$, the edge $\\{a, b\\}$ is in $E_1$ but not \
$E_3$, even though vertices $a$ and $b$ are in $V_3$.

The graph $G_4$ is NOT a subgraph of $G_1$, even though it looks like all we \
did is remove vertex $e$. The reason is that in $E_4$ we have the edge \
$\\{c, f\\}$, but this is not an element of $E_1$, so we don't have the \
required $E_4 \\subseteq E_1$.

Back to some basic graph theory definitions. Notice that all the graphs we \
have drawn above have the property that no pair of vertices is connected \
more than once, and no vertex is connected to itself. Graphs like these are \
sometimes called **simple**, although we will just call them *graphs*.

We say that $G' = (V', E')$ is a **subgraph** of $G = (V, E)$, and write \
$G' \\subseteq G$, provided $V' \\subseteq V$ and $E' \\subseteq E$.

We say that $G' = (V', E')$ is an **induced subgraph** of $G = (V, E)$ \
provided $V' \\subseteq V$ and every edge in $E$ whose vertices are still \
in $V'$ is also an edge in $E'$.

Notice that every induced subgraph is also an ordinary subgraph, but not \
conversely. Think of a subgraph as the result of deleting some vertices and \
edges from the larger graph. For the subgraph to be an induced subgraph, \
we can still delete vertices, but now we only delete those edges that \
included the deleted vertices.

The graphs above are also **connected**: you can get from any vertex to any \
other vertex by following some path of edges. A graph that is not connected \
can be thought of as two separate graphs drawn close together."""

SOURCE = 'graph_theory_page'


async def main():
    if not db.is_configured():
        print('Neo4j not configured.')
        return

    def _session():
        return db.session()

    async with _session() as session:
        await session.run('MATCH (n) DETACH DELETE n')
    await schema.ensure_schema(_session)

    lm = llm.module_lm('canonicalizer')
    nodes = [models.ASTNode(id=0, type='paragraph', content=CONTENT)]
    triplet_node = TripletNode(
        fact_module=_FactExtractor(lm), triplet_module=_TripletDecomposer(lm)
    )
    result = await triplet_node.run({'nodes': nodes, 'source': SOURCE})
    triplets = result.get('triplets', [])

    print(f'{len(triplets)} triplets:')
    for t in triplets:
        print(f'  {t.subject} | {t.predicate} | {t.object}')

    enricher = canonicalizer.ComponentEnricher(language_model=lm)
    (
        entity_descriptions,
        predicate_descriptions,
    ) = await canonicalizer.enrich_components(nodes, triplets, enricher)

    (
        entity_embeddings,
        predicate_embeddings,
    ) = await canonicalizer.embed_components(
        entity_descriptions,
        predicate_descriptions,
        embeddings.embedder(),
    )

    await writer.persist_nodes(nodes, SOURCE, session_factory=_session)
    await writer.persist_assertions(
        triplets,
        SOURCE,
        session_factory=_session,
        entity_descriptions=entity_descriptions,
        predicate_descriptions=predicate_descriptions,
        entity_embeddings=entity_embeddings,
        predicate_embeddings=predicate_embeddings,
    )
    await writer.persist_chain(nodes, SOURCE, session_factory=_session)

    print('\n' + '=' * 60)
    print('CANONICALIZER (maintenance rebuild)')
    print('=' * 60)

    result = await rebuild(
        'entity',
        language_model=lm,
        session_factory=_session,
        source=SOURCE,
    )
    print(f'\nRecords: {result["records"]}')
    print(f'Clusters: {result["clusters"]}')

    async with _session() as session:
        hubs = await session.run(
            'MATCH (h:EntityHub) '
            'RETURN h.canonical_name AS name, h.description AS description '
            'ORDER BY name'
        )
        records = [r async for r in hubs]
        print(f'\n{len(records)} EntityHub(s):')
        for r in records:
            print(f'  {r["name"]}')
            print(f'    {r["description"]}')
    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
