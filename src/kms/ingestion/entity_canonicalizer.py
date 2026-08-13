import asyncio
import logging
from collections import Counter
from collections.abc import Callable

import dspy
from pydantic import BaseModel, Field

from kms.core import content, embeddings, llm
from kms.graph import queries, writer

logger = logging.getLogger(__name__)


class _DefinitionInput(BaseModel):
    canonical_name: str = Field(
        description='The canonical name that best represents this entity.'
    )
    description: str = Field(
        description='A concise 1-2 sentence description of what this entity is.'
    )


class _DefinitionSignature(dspy.Signature):
    r"""
    You are given several surface forms of the same entity from different
    contexts in a technical document, together with the full triplet
    assertions where they appear. Write a canonical name and a 1-2 sentence
    description of what this entity IS.

    The surface forms include parenthetical role annotations added during
    extraction. The canonical name should be the most informative concept
    name — prefer general terms over specific notation: "graph" not
    "$G = (V, E)$ (a graph)", "vertex set" not "$V$ (a vertex set)".
    If the entity is a specific named object rather than a general
    concept, keep the most informative surface form. Preserve LaTeX
    with $ delimiters for mathematical notation.

    The description should be a STANDALONE TEXTBOOK-STYLE DEFINITION:
    - Write as if for a reference work — the reader has no access to
      the source document
    - Strip out all source-specific references: never mention "in the
      given contexts", "$G_1$", "the document", "the passage", or any
      example-specific objects that only exist in the source
    - State only what the entity IS in general — not what it does in a
      particular passage
    - Use LaTeX with $ delimiters for any mathematical notation
    - Be readable standalone — someone reading only the definition
      should understand the entity completely

    Never invent information not supported by at least one of the input
    surface forms or their contexts.
    """

    surface_forms: list[str] = dspy.InputField(
        description='Every surface form that belongs to this entity cluster.'
    )
    contexts: list[str] = dspy.InputField(
        description='The full triplet assertions each surface form appears in.'
    )
    result: _DefinitionInput = dspy.OutputField(
        description='Canonical name and description for this entity.'
    )


class _DefinitionSynthesizer(dspy.Module):
    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.synthesizer = dspy.Predict(_DefinitionSignature)
        self.set_lm(language_model)

    async def aforward(
        self, surface_forms: list[str], contexts: list[str]
    ) -> tuple[str, str]:
        result = await self.synthesizer.acall(
            surface_forms=surface_forms, contexts=contexts
        )
        definition = result.result
        return definition.canonical_name, definition.description

    def forward(
        self, surface_forms: list[str], contexts: list[str]
    ) -> tuple[str, str]:
        return asyncio.run(self.aforward(surface_forms, contexts))


def _cluster(entities: list[dict], threshold: float) -> list[list[dict]]:
    count = len(entities)
    adjacency: dict[int, list[int]] = {i: [] for i in range(count)}
    for i in range(count):
        for j in range(i + 1, count):
            if (
                embeddings.cosine_similarity(
                    entities[i]['embedding'], entities[j]['embedding']
                )
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
                    component.append(entities[vertex])
                    stack.extend(adjacency[vertex])
            clusters.append(component)
    return clusters


async def _synthesize_definitions(
    clusters: list[list[dict]],
    synthesizer: _DefinitionSynthesizer,
    max_concurrency: int | None = None,
) -> list[dict]:
    gate = llm.gate(max_concurrency)

    async def _one(cluster: list[dict]) -> dict:
        surface_forms = [e['name'] for e in cluster]
        contexts = []
        for e in cluster:
            for t in e.get('triplet_texts', []):
                if t not in contexts:
                    contexts.append(t)
        async with gate:
            canonical_name, description = await synthesizer.aforward(
                surface_forms, contexts
            )
        return {
            'canonical_name': canonical_name,
            'description': description,
        }

    results = await asyncio.gather(*(_one(cluster) for cluster in clusters))
    logger.info(
        'entity canonicalizer: %d definition(s) synthesized',
        len(results),
    )
    return list(results)


async def rebuild(
    threshold: float,
    language_model: dspy.LM,
    session_factory: Callable,
) -> dict:
    spokes = await queries.all_entity_spokes(session_factory)
    logger.info('entity canonicalizer: %d triplet(s) read', len(spokes))

    if not spokes:
        return {'clusters': 0, 'entities': 0}

    entity_map: dict[str, dict] = {}
    for spoke in spokes:
        triplet_text = (
            f'{spoke["subject"]} | {spoke["predicate"]} | {spoke["object"]}'
        )
        for role in ('subject', 'object'):
            name = spoke[role]
            if name not in entity_map:
                entity_map[name] = {
                    'name': name,
                    'subject_triplets': [],
                    'object_triplets': [],
                    'triplet_texts': [],
                    'sources': [],
                }
            entry = entity_map[name]
            if role == 'subject':
                entry['subject_triplets'].append(spoke['triplet_uuid'])
            else:
                entry['object_triplets'].append(spoke['triplet_uuid'])
            if triplet_text not in entry['triplet_texts']:
                entry['triplet_texts'].append(triplet_text)
            if spoke.get('source'):
                entry['sources'].append(spoke['source'])

    entities = list(entity_map.values())
    logger.info(
        'entity canonicalizer: %d unique entity string(s)',
        len(entities),
    )

    if not embeddings.is_configured():
        logger.warning('entity canonicalizer: no embedding key, skipping')
        return {'clusters': 0, 'entities': len(entities)}

    embedder = embeddings.embedder()
    texts = [
        f'{e["name"]} : {e["triplet_texts"][0]}'
        if e['triplet_texts']
        else e['name']
        for e in entities
    ]
    vectors = await embedder.embed(
        [content.Content.from_text(text) for text in texts]
    )
    for entity, vector in zip(entities, vectors, strict=True):
        entity['embedding'] = vector

    clusters = _cluster(entities, threshold)
    logger.info(
        'entity canonicalizer: %d cluster(s) at threshold %.2f',
        len(clusters),
        threshold,
    )

    synthesizer = _DefinitionSynthesizer(language_model)
    definitions = await _synthesize_definitions(clusters, synthesizer)

    hub_texts = [
        f'{definition["canonical_name"]}: {definition["description"]}'
        for definition in definitions
    ]
    hub_vectors = await embedder.embed(
        [content.Content.from_text(text) for text in hub_texts]
    )

    hubs: list[dict] = []
    for index, (cluster, definition) in enumerate(
        zip(clusters, definitions, strict=True)
    ):
        sources = []
        for e in cluster:
            sources.extend(e.get('sources', []))
        counts = Counter(sources)
        source = counts.most_common(1)[0][0] if counts else 'unknown'

        hubs.append(
            {
                'source': source,
                'canonical_name': definition['canonical_name'],
                'description': definition['description'],
                'embedding': hub_vectors[index],
                'subject_spokes': [
                    {'triplet_uuid': u}
                    for e in cluster
                    for u in e.get('subject_triplets', [])
                ],
                'object_spokes': [
                    {'triplet_uuid': u}
                    for e in cluster
                    for u in e.get('object_triplets', [])
                ],
            }
        )

    await writer.persist_entity_hubs(hubs, session_factory=session_factory)
    logger.info('entity canonicalizer: %d hub(s) persisted', len(hubs))

    return {'clusters': len(clusters), 'entities': len(entities)}
