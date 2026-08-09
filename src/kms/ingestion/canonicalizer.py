"""
Entity and predicate canonicalization — full rebuild from spokes.

Every source ingestion triggers a complete rebuild of the canonical
layer: read all spokes across all sources, cluster by embedding
similarity, synthesise definitions, delete the old canonical layer,
and write fresh hubs, definitions, and CANONICAL edges.

The canonical layer is fully derived from immutable spokes — there is
no incremental merge and no cross-batch adjudication.  The rebuild is
deterministic for a given set of spokes and threshold.

Design commitments:

* CRASH ON MISSING EMBEDDINGS — every spoke must carry an embedding.
* PURE-MATH CLUSTERING — all-pairs cosine similarity, no LLM in the
  cluster path.
* ONE LLM CALL PER DEFINITION — definition synthesis is per cluster,
  run concurrently.
* ATOMIC REPLACEMENT — the old canonical layer is wiped before the
  new one is written, so the graph is never in a mixed state.
"""

import asyncio
import logging
from collections import Counter

import dspy

from kms.core import embeddings
from kms.core import llm as llm_config

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy module — definition synthesis
# ============================================================================


class DefinitionSignature(dspy.Signature):
    """
    You are given several descriptions of the same concept from different
    contexts in a document. Each description was written independently by a
    reader who only saw one occurrence. Write a single canonical 1-2
    sentence definition that synthesises what this concept IS.

    The canonical definition should:
    - Capture the essential meaning shared across all the descriptions
    - Include any important detail that appears in only one description
    - Be readable standalone — someone reading only the definition should
      understand the concept
    - Use LaTeX with $ delimiters for any mathematical notation

    Never invent information that is not supported by at least one of the
    input descriptions.
    """

    concept_name: str = dspy.InputField(
        description='The name of the concept being defined.'
    )
    descriptions: list[str] = dspy.InputField(
        description='The individual descriptions to synthesise.'
    )
    definition: str = dspy.OutputField(
        description='1-2 sentence canonical definition.'
    )


class DefinitionSynthesizer(dspy.Module):
    """Synthesises multiple descriptions into one canonical definition.

    Args:
        language_model: The LM to run on.
    """

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(DefinitionSignature)
        self.set_lm(language_model)

    async def aforward(
        self, concept_name: str, descriptions: list[str]
    ) -> str:
        """Synthesize a canonical definition asynchronously.

        Args:
            concept_name: The name of the concept being defined.
            descriptions: The source descriptions to synthesize from.

        Returns:
            The synthesized canonical definition.
        """
        result = await self.synthesizer.acall(
            concept_name=concept_name, descriptions=descriptions
        )
        logger.debug(
            'definition synthesizer: %d descriptions -> 1 definition',
            len(descriptions),
        )
        return result.definition

    def forward(self, concept_name: str, descriptions: list[str]) -> str:
        """Synthesize a canonical definition synchronously.

        Wraps :meth:`aforward` in an asyncio event loop.

        Args:
            concept_name: The name of the concept being defined.
            descriptions: The source descriptions to synthesize from.

        Returns:
            The synthesized canonical definition.
        """
        return asyncio.run(self.aforward(concept_name, descriptions))


# ============================================================================
# Clustering
# ============================================================================


def _cluster(spokes: list[dict], threshold: float) -> list[list[dict]]:
    """Cluster spokes by all-pairs cosine similarity → connected components.

    Args:
        spokes: One dict per spoke, each with ``uuid`` and ``embedding``.
        threshold: Minimum cosine similarity for two spokes to be
            considered the same concept.

    Returns:
        One list of spoke dicts per cluster.

    Raises:
        ValueError: If any spoke is missing its embedding.
    """
    for spoke in spokes:
        if spoke.get('embedding') is None:
            display = spoke.get('name') or spoke.get('predicate')
            raise ValueError(
                f'Spoke {spoke["uuid"]} ({display}) has no embedding. '
                f'Run the entity_embedding stage first.'
            )

    count = len(spokes)
    vectors = [spoke['embedding'] for spoke in spokes]

    adjacency: dict[int, list[int]] = {i: [] for i in range(count)}
    for i in range(count):
        for j in range(i + 1, count):
            if (
                embeddings.cosine_similarity(vectors[i], vectors[j])
                >= threshold
            ):
                adjacency[i].append(j)
                adjacency[j].append(i)

    visited: set[int] = set()
    clusters: list[list[dict]] = []
    for i in range(count):
        if i not in visited:
            component: list[dict] = []
            stack = [i]
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    component.append(spokes[vertex])
                    stack.extend(adjacency[vertex])
            clusters.append(component)

    return clusters


# ============================================================================
# Helpers — centroid, display name, description collection
# ============================================================================


def _most_frequent(values: list[str]) -> str:
    counts = Counter(values)
    max_count = max(counts.values())
    candidates = [v for v, c in counts.items() if c == max_count]
    return min(candidates, key=len)


def _display_name(cluster: list[dict], name_key: str) -> str:
    return _most_frequent([spoke[name_key] for spoke in cluster])


def _centroid_embedding(cluster: list[dict]) -> list[float]:
    dim = len(cluster[0]['embedding'])
    centroid = [0.0] * dim
    for spoke in cluster:
        for i, v in enumerate(spoke['embedding']):
            centroid[i] += v
    count = len(cluster)
    return [v / count for v in centroid]


def _collect_descriptions(cluster: list[dict]) -> list[str]:
    return list(
        {spoke['description'] for spoke in cluster if spoke.get('description')}
    )


# ============================================================================
# Definition synthesis
# ============================================================================


async def _synthesize_definition(
    cluster: list[dict],
    name_key: str,
    synthesizer: DefinitionSynthesizer,
) -> str:
    """Create a canonical definition for one cluster.

    Args:
        cluster: One cluster's spoke dicts.
        name_key: ``'name'`` or ``'predicate'``.
        synthesizer: The definition-writing LLM module.

    Returns:
        The definition text.
    """
    display = _display_name(cluster, name_key)
    descriptions = _collect_descriptions(cluster)

    if len(descriptions) >= 2:
        return await synthesizer.aforward(display, descriptions)
    elif len(descriptions) == 1:
        return descriptions[0]
    else:
        return display


async def _embed_text(text: str) -> list[float] | None:
    """Embed a single string, returning None when no embedder is
    configured."""
    if not embeddings.is_configured():
        return None
    embedder = embeddings.embedder()
    return (await embedder.embed([text]))[0]


async def _synthesize_all(
    clusters: list[list[dict]],
    name_key: str,
    synthesizer: DefinitionSynthesizer,
    *,
    max_concurrency: int | None = None,
) -> list[dict]:
    """Synthesize a canonical definition for every cluster, concurrently.

    Args:
        clusters: One list of spoke dicts per cluster.
        name_key: ``'name'`` for entities, ``'predicate'`` for predicates.
        synthesizer: The definition-writing LLM module.
        max_concurrency: Max LLM calls in flight. None uses
            ``llm.MAX_CONCURRENT_CALLS``.

    Returns:
        One dict per cluster:
        ``{display_name, definition_text, definition_embedding}``.
    """
    gate = llm_config.gate(max_concurrency)

    async def _one(cluster: list[dict]) -> dict:
        async with gate:
            display = _display_name(cluster, name_key)
            def_text = await _synthesize_definition(
                cluster, name_key, synthesizer
            )
            def_embedding = await _embed_text(def_text)
            return {
                'display_name': display,
                'definition_text': def_text,
                'definition_embedding': def_embedding,
            }

    definitions = await asyncio.gather(
        *(_one(cluster) for cluster in clusters)
    )
    logger.info(
        'synthesized %d %s definition(s)',
        len(definitions),
        name_key,
    )
    return list(definitions)


# ============================================================================
# Write helpers
# ============================================================================


async def _write_entity_hubs(
    clusters: list[list[dict]],
    definitions: list[dict],
    session_factory,
) -> None:
    """Upsert :EntityHub, :Definition, CANONICAL, and HAS_DEFINITION.

    Uses the deterministic-source-majority convention: each cluster's
    source is the most common source among its spokes.
    """
    from kms.graph import queries, writer
    from kms.graph.definitions import definition_rows, has_definition_pairs
    from kms.graph.entity_hubs import entity_hub_uuid

    hub_defs: list[dict] = []
    hub_rows: list[dict] = []
    canonical_pairs: list[dict] = []
    now = writer.utcnow_iso()

    for cluster, definition in zip(clusters, definitions, strict=True):
        spoke_uuids = [s['uuid'] for s in cluster]
        # Majority source
        sources = [s.get('source', 'unknown') for s in cluster]
        source = Counter(sources).most_common(1)[0][0]
        hub_uuid = entity_hub_uuid(source, spoke_uuids)

        hub_defs.append({
            'hub_uuid': hub_uuid,
            'definition_text': definition['definition_text'],
            'definition_embedding': definition['definition_embedding'],
        })

        from kms.graph.entity_hubs import entity_hub_properties
        hub_rows.append(entity_hub_properties(
            source, spoke_uuids,
            display_name=definition['display_name'],
        ))

        for spoke in cluster:
            canonical_pairs.append({'entity': spoke['uuid'], 'hub': hub_uuid})

    if not hub_rows:
        return

    async with session_factory() as session:
        await session.run(queries.MERGE_ENTITY_HUBS, rows=hub_rows, now=now)

        def_rows_list = definition_rows(hub_defs)
        if def_rows_list:
            await session.run(
                queries.MERGE_DEFINITIONS, rows=def_rows_list, now=now
            )

        if canonical_pairs:
            await session.run(
                queries.MERGE_CANONICAL_ENTITY,
                pairs=canonical_pairs,
                now=now,
            )

        has_def_pairs_list = has_definition_pairs(hub_defs)
        if has_def_pairs_list:
            await session.run(
                queries.MERGE_HAS_DEFINITION,
                pairs=has_def_pairs_list,
                now=now,
            )


async def _write_predicate_hubs(
    clusters: list[list[dict]],
    definitions: list[dict],
    session_factory,
) -> None:
    """Upsert :PredicateHub, :Definition, CANONICAL, and HAS_DEFINITION."""
    from kms.graph import queries
    from kms.graph import writer as w
    from kms.graph.definitions import definition_rows, has_definition_pairs
    from kms.graph.predicate_hubs import predicate_hub_uuid

    hub_defs: list[dict] = []
    hub_rows: list[dict] = []
    canonical_pairs: list[dict] = []
    now = w.utcnow_iso()

    for cluster, definition in zip(clusters, definitions, strict=True):
        spoke_uuids = [s['uuid'] for s in cluster]
        sources = [s.get('source', 'unknown') for s in cluster]
        source = Counter(sources).most_common(1)[0][0]
        hub_uuid = predicate_hub_uuid(source, spoke_uuids)

        hub_defs.append({
            'hub_uuid': hub_uuid,
            'definition_text': definition['definition_text'],
            'definition_embedding': definition['definition_embedding'],
        })

        from kms.graph.predicate_hubs import predicate_hub_properties
        hub_rows.append(predicate_hub_properties(
            source, spoke_uuids,
            display_name=definition['display_name'],
        ))

        for spoke in cluster:
            canonical_pairs.append(
                {'predicate': spoke['uuid'], 'hub': hub_uuid}
            )

    if not hub_rows:
        return

    async with session_factory() as session:
        await session.run(
            queries.MERGE_PREDICATE_HUBS, rows=hub_rows, now=now
        )

        def_rows_list = definition_rows(hub_defs)
        if def_rows_list:
            await session.run(
                queries.MERGE_DEFINITIONS, rows=def_rows_list, now=now
            )

        if canonical_pairs:
            await session.run(
                queries.MERGE_CANONICAL_PREDICATE,
                pairs=canonical_pairs,
                now=now,
            )

        has_def_pairs_list = has_definition_pairs(hub_defs)
        if has_def_pairs_list:
            await session.run(
                queries.MERGE_HAS_DEFINITION,
                pairs=has_def_pairs_list,
                now=now,
            )


# ============================================================================
# TripletHub + FactHub rebuild
# ============================================================================


async def _rebuild_triplet_hubs(session_factory) -> None:
    """Rebuild :TripletHub and :FactHub from the fresh canonical layer.

    Reads every :Triplet, resolves to its canonical hubs, groups by
    (subj_hub, pred_hub, obj_hub), and writes :TripletHub + :FactHub.
    """
    from collections import defaultdict

    from kms.core import embeddings as emb
    from kms.graph import queries, triplet_hubs, writer

    hub_triplets = await queries.all_canonical_triplets(session_factory)
    print(f'  {len(hub_triplets)} canonical triplet(s) at hub level')

    if not hub_triplets:
        return

    # Group by (subj_hub, pred_hub, obj_hub)
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for ht in hub_triplets:
        key = (ht['subj_hub'], ht['pred_hub'], ht['obj_hub'])
        groups[key].append(ht)

    print(
        f'  {len(groups)} unique canonical assertion(s) '
        f'({len(hub_triplets)} total triplet(s))'
    )

    # Build group dicts
    result: list[dict] = []
    texts_to_embed: list[str] = []
    embed_indices: list[int] = []

    for i, ((subj, pred, obj), triplets) in enumerate(groups.items()):
        first = triplets[0]
        subj_name = first['subj_name']
        pred_name = first['pred_name']
        obj_name = first['obj_name']

        # Majority source
        sources = [
            t.get('source', 'unknown') for t in triplets
            if t.get('source')
        ]
        source = (
            Counter(sources).most_common(1)[0][0]
            if sources else 'global'
        )
        fact_text = f'{subj_name} {pred_name} {obj_name}'

        result.append({
            'triplet_hub_uuid': triplet_hubs.triplet_hub_uuid(
                source, subj, pred, obj
            ),
            'subj_hub': subj,
            'pred_hub': pred,
            'obj_hub': obj,
            'triplet_uuids': [t['triplet_uuid'] for t in triplets],
            'fact_text': fact_text,
        })
        texts_to_embed.append(fact_text)
        embed_indices.append(i)

    # Embed assertion texts
    if emb.is_configured():
        embedder = emb.embedder()
        vectors = await embedder.embed(texts_to_embed)
        for idx, vector in zip(embed_indices, vectors, strict=True):
            result[idx]['fact_embedding'] = vector
        print(f'  {len(vectors)} assertion text(s) embedded')

    # Write
    await writer.persist_triplet_hubs(
        result, session_factory=session_factory
    )


# ============================================================================
# Public entry point
# ============================================================================


async def rebuild(
    threshold: float,
    language_model: dspy.LM,
    session_factory,
    *,
    entity_kind: bool = True,
    predicate_kind: bool = True,
    rebuild_triplets: bool = True,
) -> dict:
    """Rebuild the entire canonical layer from all spokes.

    Args:
        threshold: Minimum cosine similarity for clustering.
        language_model: The LM for definition synthesis.
        session_factory: Neo4j session factory.
        entity_kind: Whether to canonicalize entities.
        predicate_kind: Whether to canonicalize predicates.
        rebuild_triplets: Whether to rebuild TripletHubs + FactHubs
            after the entity/predicate canonical layer.

    Returns:
        A dict with keys ``entity`` and ``predicate``, each a dict
        ``{clusters, spokes}``.
    """
    from kms.graph import queries

    synthesizer = DefinitionSynthesizer(language_model)
    result: dict = {}

    # ==================================================================
    # ENTITIES
    # ==================================================================
    if entity_kind:
        print('=' * 60)
        print('ENTITY CANONICALIZATION')
        print('=' * 60)

        spokes = await queries.all_entity_spokes(session_factory)
        print(f'\nPhase 1 — Read: {len(spokes)} entity spoke(s)')

        if spokes:
            clusters = _cluster(spokes, threshold)
            print(
                f'Phase 2 — Cluster: {len(clusters)} cluster(s) '
                f'from {len(spokes)} spoke(s) at threshold {threshold}'
            )

            print('Phase 3 — Synthesize definitions...')
            definitions = await _synthesize_all(
                clusters, 'name', synthesizer
            )

            print('Phase 4 — Delete old canonical layer...')
            await queries.delete_canonical_layer(session_factory)

            print('Phase 5 — Write new entity hubs...')
            await _write_entity_hubs(
                clusters, definitions, session_factory
            )

            result['entity'] = {
                'clusters': len(clusters),
                'spokes': len(spokes),
            }
        else:
            result['entity'] = {'clusters': 0, 'spokes': 0}

    # ==================================================================
    # PREDICATES
    # ==================================================================
    if predicate_kind:
        print(f'\n{"=" * 60}')
        print('PREDICATE CANONICALIZATION')
        print('=' * 60)

        spokes = await queries.all_predicate_spokes(session_factory)
        print(f'\nPhase 1 — Read: {len(spokes)} predicate spoke(s)')

        if spokes:
            clusters = _cluster(spokes, threshold)
            print(
                f'Phase 2 — Cluster: {len(clusters)} cluster(s) '
                f'from {len(spokes)} spoke(s) at threshold {threshold}'
            )

            print('Phase 3 — Synthesize definitions...')
            definitions = await _synthesize_all(
                clusters, 'predicate', synthesizer
            )

            print('Phase 4 — Delete old canonical layer...')
            await queries.delete_canonical_layer(session_factory)

            print('Phase 5 — Write new predicate hubs...')
            await _write_predicate_hubs(
                clusters, definitions, session_factory
            )

            result['predicate'] = {
                'clusters': len(clusters),
                'spokes': len(spokes),
            }
        else:
            result['predicate'] = {'clusters': 0, 'spokes': 0}

    # ==================================================================
    # TRIPLET HUBS + FACT HUBS
    # ==================================================================
    if rebuild_triplets:
        print(f'\n{"=" * 60}')
        print('TRIPLET HUB + FACT HUB REBUILD')
        print('=' * 60)
        await _rebuild_triplet_hubs(session_factory)

    return result
