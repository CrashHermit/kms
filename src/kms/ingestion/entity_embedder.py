import logging

from kms.core import embeddings, state

logger = logging.getLogger(__name__)


def _embed_text(name: str, description: str | None) -> str:
    if description:
        return f'{name}: {description}'
    return name


async def embed_descriptions(
    entity_descs: dict[int, list[dict]],
    predicate_descs: dict[int, list[dict]],
) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    if not embeddings.is_configured():
        logger.info('entity embedder: no embedding key configured, skipping')
        return entity_descs, predicate_descs
    texts: list[str] = []
    entity_entries: list[tuple[int, int]] = []
    predicate_entries: list[tuple[int, int]] = []

    for node_id, entries in entity_descs.items():
        for index, entry in enumerate(entries):
            texts.append(_embed_text(entry['name'], entry.get('description')))
            entity_entries.append((node_id, index))

    for node_id, entries in predicate_descs.items():
        for index, entry in enumerate(entries):
            texts.append(
                _embed_text(entry['predicate'], entry.get('description'))
            )
            predicate_entries.append((node_id, index))

    if not texts:
        return entity_descs, predicate_descs

    embedder = embeddings.embedder()
    vectors = await embedder.embed(texts)

    n_entities = len(entity_entries)
    entity_vectors = vectors[:n_entities]
    predicate_vectors = vectors[n_entities:]
    result_entities: dict[int, list[dict]] = {
        node_id: [dict(entry) for entry in entries]
        for node_id, entries in entity_descs.items()
    }
    for (node_id, index), vector in zip(
        entity_entries, entity_vectors, strict=True
    ):
        result_entities[node_id][index]['embedding'] = vector

    result_predicates: dict[int, list[dict]] = {
        node_id: [dict(entry) for entry in entries]
        for node_id, entries in predicate_descs.items()
    }
    for (node_id, index), vector in zip(
        predicate_entries, predicate_vectors, strict=True
    ):
        result_predicates[node_id][index]['embedding'] = vector

    logger.info(
        'entity embedder: %d entity + %d predicate embedding(s)',
        n_entities,
        len(predicate_entries),
    )
    return result_entities, result_predicates


class EntityEmbedderNode:
    async def run(self, state: state.State) -> dict:
        entity_descs = state.get('node_entity_descriptions', {})
        predicate_descs = state.get('node_predicate_descriptions', {})
        enriched_entities, enriched_predicates = await embed_descriptions(
            entity_descs, predicate_descs
        )
        return {
            'node_entity_descriptions': enriched_entities,
            'node_predicate_descriptions': enriched_predicates,
        }

