import asyncio
import logging
from collections import Counter

import dspy

from kms.core import embeddings
from kms.core import llm as llm_config

logger = logging.getLogger(__name__)


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
    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(DefinitionSignature)
        self.set_lm(language_model)

    async def aforward(
        self, concept_name: str, descriptions: list[str]
    ) -> str:
        result = await self.synthesizer.acall(
            concept_name=concept_name, descriptions=descriptions
        )
        logger.debug(
            'definition synthesizer: %d descriptions -> 1 definition',
            len(descriptions),
        )
        return result.definition

    def forward(self, concept_name: str, descriptions: list[str]) -> str:
        return asyncio.run(self.aforward(concept_name, descriptions))


def _cluster(spokes: list[dict], threshold: float) -> list[list[dict]]:
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


async def _synthesize_definition(
    cluster: list[dict],
    name_key: str,
    synthesizer: DefinitionSynthesizer,
) -> str:
    display = _display_name(cluster, name_key)
    descriptions = _collect_descriptions(cluster)

    if len(descriptions) >= 2:
        return await synthesizer.aforward(display, descriptions)
    elif len(descriptions) == 1:
        return descriptions[0]
    else:
        return display


async def _embed_text(text: str) -> list[float] | None:
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


class AdjudicationSignature(dspy.Signature):
    """
    You are verifying whether an entity or predicate description belongs
    to a given concept cluster.

    You are shown:
    - A concept name (the most common surface form in the cluster)
    - Several sample descriptions of the concept from different contexts
    - A target description to verify

    Decide whether the target description refers to the SAME concept.
    Different concepts sometimes share vocabulary — focus on what the
    descriptions MEAN, not just what words they use.  When the target
    describes a genuinely different concept, answer No.
    """

    concept_name: str = dspy.InputField(
        description='The most common surface form in the cluster.'
    )
    sample_descriptions: list[str] = dspy.InputField(
        description='3-5 descriptions from different contexts that '
        'represent the concept.'
    )
    target_description: str = dspy.InputField(
        description='The description to verify — does it belong to '
        'this concept?'
    )
    is_same_concept: bool = dspy.OutputField(
        description='True if the target describes the same concept as '
        'the samples.'
    )


class Adjudicator(dspy.Module):
    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(AdjudicationSignature)
        self.set_lm(language_model)

    async def aforward(
        self,
        concept_name: str,
        sample_descriptions: list[str],
        target_description: str,
    ) -> bool:
        result = await self.judge.acall(
            concept_name=concept_name,
            sample_descriptions=sample_descriptions,
            target_description=target_description,
        )
        return result.is_same_concept

    def forward(
        self,
        concept_name: str,
        sample_descriptions: list[str],
        target_description: str,
    ) -> bool:
        return asyncio.run(
            self.aforward(
                concept_name, sample_descriptions, target_description
            )
        )


async def _verify_and_refine(
    clusters: list[list[dict]],
    name_key: str,
    adjudicator: Adjudicator,
    threshold: float,
    *,
    auto_accept: float = 0.85,
    auto_reject: float = 0.65,
    max_concurrency: int | None = None,
) -> list[list[dict]]:
    gate = llm_config.gate(max_concurrency)
    refined: list[list[dict]] = []
    rejected_pool: list[dict] = []
    llm_calls = 0

    for cluster in clusters:
        if len(cluster) <= 3:
            refined.append(cluster)
            continue

        centroid = _centroid_embedding(cluster)
        display = _display_name(cluster, name_key)
        samples = _pick_samples(cluster, centroid, n=5)

        kept: list[dict] = []
        to_check: list[dict] = []

        for spoke in cluster:
            sim = embeddings.cosine_similarity(
                spoke['embedding'], centroid
            )
            if sim >= auto_accept:
                kept.append(spoke)
            elif sim < auto_reject:
                rejected_pool.append(spoke)
            else:
                to_check.append(spoke)

        if not to_check:
            kept.extend(to_check)
            refined.append(kept)
            continue

        async def _check_one(
            spoke: dict,
            _display: str = display,
            _samples: list[str] = samples,
        ) -> tuple[dict, bool]:
            async with gate:
                desc = spoke.get('description') or ''
                if not desc:
                    return spoke, True
                match = await adjudicator.aforward(
                    concept_name=_display,
                    sample_descriptions=_samples,
                    target_description=desc,
                )
                return spoke, match

        results = await asyncio.gather(
            *(_check_one(spoke) for spoke in to_check)
        )
        for spoke, matched in results:
            if matched:
                kept.append(spoke)
            else:
                rejected_pool.append(spoke)

        llm_calls += len(to_check)
        refined.append(kept)
    if rejected_pool:
        reject_clusters = _cluster(rejected_pool, threshold)
        refined.extend(reject_clusters)
        logger.info(
            'verification: %d spoke(s) rejected, '
            'formed %d new cluster(s)',
            len(rejected_pool),
            len(reject_clusters),
        )

    logger.info(
        'verification: %d LLM call(s) across %d cluster(s)',
        llm_calls,
        len(clusters),
    )
    return refined


def _pick_samples(
    cluster: list[dict], centroid: list[float], n: int
) -> list[str]:
    described = [
        spoke
        for spoke in cluster
        if spoke.get('description')
    ]
    if not described:
        return []
    ranked = sorted(
        described,
        key=lambda s: embeddings.cosine_similarity(
            s['embedding'], centroid
        ),
        reverse=True,
    )
    return [s['description'] for s in ranked[:n]]


async def _write_hubs(
    clusters: list[list[dict]],
    definitions: list[dict],
    kind: str,
    session_factory,
) -> None:
    from kms.graph import hubs, queries
    from kms.graph import writer as w
    from kms.graph.definitions import definition_rows, has_definition_pairs

    hub_defs: list[dict] = []
    hub_rows_list: list[dict] = []
    now = w.utcnow_iso()

    merge_hubs = (
        queries.MERGE_ENTITY_HUBS
        if kind == 'entity'
        else queries.MERGE_PREDICATE_HUBS
    )
    merge_canonical = (
        queries.MERGE_CANONICAL_ENTITY
        if kind == 'entity'
        else queries.MERGE_CANONICAL_PREDICATE
    )

    for cluster, definition in zip(clusters, definitions, strict=True):
        spoke_uuids = [s['uuid'] for s in cluster]
        sources = [s.get('source', 'unknown') for s in cluster]
        source = Counter(sources).most_common(1)[0][0]
        hub_uuid_val = hubs.hub_uuid(source, spoke_uuids, kind)

        hub_defs.append({
            'hub_uuid': hub_uuid_val,
            'definition_text': definition['definition_text'],
            'definition_embedding': definition['definition_embedding'],
        })

        hub_rows_list.append(hubs.hub_properties(
            source, spoke_uuids, kind,
            display_name=definition['display_name'],
        ))

    canon_pairs = hubs.canonical_pairs(clusters, kind)

    if not hub_rows_list:
        return

    async with session_factory() as session:
        await session.run(merge_hubs, rows=hub_rows_list, now=now)
        def_rows_list = definition_rows(hub_defs)
        if def_rows_list:
            await session.run(
                queries.MERGE_DEFINITIONS, rows=def_rows_list, now=now
            )
        if canon_pairs:
            await session.run(
                merge_canonical, pairs=canon_pairs, now=now
            )
        has_def_pairs = has_definition_pairs(hub_defs)
        if has_def_pairs:
            await session.run(
                queries.MERGE_HAS_DEFINITION,
                pairs=has_def_pairs,
                now=now,
            )


async def _rebuild_triplet_hubs(session_factory) -> None:
    from collections import defaultdict

    from kms.core import embeddings as emb
    from kms.graph import queries, triplet_hubs, writer

    hub_triplets = await queries.all_canonical_triplets(session_factory)
    print(f'  {len(hub_triplets)} canonical triplet(s) at hub level')

    if not hub_triplets:
        return
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for ht in hub_triplets:
        key = (ht['subj_hub'], ht['pred_hub'], ht['obj_hub'])
        groups[key].append(ht)

    print(
        f'  {len(groups)} unique canonical assertion(s) '
        f'({len(hub_triplets)} total triplet(s))'
    )
    result: list[dict] = []
    texts_to_embed: list[str] = []
    embed_indices: list[int] = []

    for i, ((subj, pred, obj), triplets) in enumerate(groups.items()):
        first = triplets[0]
        subj_name = first['subj_name']
        pred_name = first['pred_name']
        obj_name = first['obj_name']
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
    if emb.is_configured():
        embedder = emb.embedder()
        vectors = await embedder.embed(texts_to_embed)
        for idx, vector in zip(embed_indices, vectors, strict=True):
            result[idx]['fact_embedding'] = vector
        print(f'  {len(vectors)} assertion text(s) embedded')
    await writer.persist_triplet_hubs(
        result, session_factory=session_factory
    )


async def rebuild(
    threshold: float,
    language_model: dspy.LM,
    session_factory,
    *,
    entity_kind: bool = True,
    predicate_kind: bool = True,
    rebuild_triplets: bool = True,
    adjudicate: bool = True,
    coarse_threshold: float | None = None,
) -> dict:
    from kms.graph import queries

    synthesizer = DefinitionSynthesizer(language_model)
    adjudicator = Adjudicator(language_model) if adjudicate else None
    coarse = (
        coarse_threshold
        if coarse_threshold is not None
        else threshold - 0.15 if adjudicate else threshold
    )
    result: dict = {}
    if entity_kind:
        print('=' * 60)
        print('ENTITY CANONICALIZATION')
        print('=' * 60)

        spokes = await queries.all_entity_spokes(session_factory)
        print(f'\nPhase 1 — Read: {len(spokes)} entity spoke(s)')

        if spokes:
            clusters = _cluster(spokes, coarse)
            print(
                f'Phase 2 — Coarse cluster: {len(clusters)} cluster(s) '
                f'from {len(spokes)} spoke(s) at threshold {coarse}'
            )

            if adjudicator:
                print('Phase 3 — Verify clusters (LLM)...')
                clusters = await _verify_and_refine(
                    clusters, 'name', adjudicator, threshold
                )
                print(
                    f'  {len(clusters)} cluster(s) after verification'
                )
                phase = 4
            else:
                phase = 3

            print(f'Phase {phase} — Synthesize definitions...')
            definitions = await _synthesize_all(
                clusters, 'name', synthesizer
            )

            print(f'Phase {phase + 1} — Delete old canonical layer...')
            await queries.delete_canonical_layer(session_factory)

            print(f'Phase {phase + 2} — Write new entity hubs...')
            await _write_hubs(
                clusters, definitions, 'entity', session_factory
            )

            result['entity'] = {
                'clusters': len(clusters),
                'spokes': len(spokes),
            }
        else:
            result['entity'] = {'clusters': 0, 'spokes': 0}
    if predicate_kind:
        print(f'\n{"=" * 60}')
        print('PREDICATE CANONICALIZATION')
        print('=' * 60)

        spokes = await queries.all_predicate_spokes(session_factory)
        print(f'\nPhase 1 — Read: {len(spokes)} predicate spoke(s)')

        if spokes:
            clusters = _cluster(spokes, coarse)
            print(
                f'Phase 2 — Coarse cluster: {len(clusters)} cluster(s) '
                f'from {len(spokes)} spoke(s) at threshold {coarse}'
            )

            if adjudicator:
                print('Phase 3 — Verify clusters (LLM)...')
                clusters = await _verify_and_refine(
                    clusters, 'predicate', adjudicator, threshold
                )
                print(
                    f'  {len(clusters)} cluster(s) after verification'
                )
                phase = 4
            else:
                phase = 3

            print(f'Phase {phase} — Synthesize definitions...')
            definitions = await _synthesize_all(
                clusters, 'predicate', synthesizer
            )

            print(f'Phase {phase + 1} — Delete old canonical layer...')
            await queries.delete_canonical_layer(session_factory)

            print(f'Phase {phase + 2} — Write new predicate hubs...')
            await _write_hubs(
                clusters, definitions, 'predicate', session_factory
            )

            result['predicate'] = {
                'clusters': len(clusters),
                'spokes': len(spokes),
            }
        else:
            result['predicate'] = {'clusters': 0, 'spokes': 0}
    if rebuild_triplets:
        print(f'\n{"=" * 60}')
        print('TRIPLET HUB + FACT HUB REBUILD')
        print('=' * 60)
        await _rebuild_triplet_hubs(session_factory)

    return result

