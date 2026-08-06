"""
Entity and predicate embedding — one LangGraph node.

Runs after entity enrichment and before the ingestion persister. Reads the
per-node entity and predicate description dicts, computes an embedding for
each entry by concatenating the name/predicate with its description
(``"{name}: {description}"``), and writes the enriched dicts back with an
optional ``embedding`` field.

When no embedding API key is configured the stage is a no-op: the dicts
pass through unchanged and no ``embedding`` field appears on any vertex.
"""

import logging

from kms.core import embeddings, state

logger = logging.getLogger(__name__)


def _embed_text(name: str, description: str | None) -> str:
    """The text to vectorise for a single entity or predicate.

    Args:
        name: The entity name or predicate text.
        description: The enricher's description, or None.

    Returns:
        ``"{name}: {description}"`` when a description is present, otherwise
        the bare name.
    """
    if description:
        return f'{name}: {description}'
    return name


async def embed_descriptions(
    entity_descs: dict[int, list[dict]],
    predicate_descs: dict[int, list[dict]],
) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    """Compute embeddings for every entity and predicate description.

    All texts are collected into one batch and sent to the embedder in a
    single request (or as few as the batch size allows), so the latency
    cost is constant regardless of description count.

    Args:
        entity_descs: The per-node entity descriptions from the enricher.
        predicate_descs: The per-node predicate descriptions from the
            enricher.

    Returns:
        New dicts with an ``embedding`` field on each entry. When no
        embedding API key is configured, the inputs are returned unchanged.
    """
    if not embeddings.is_configured():
        logger.info(
            'entity embedder: no embedding key configured, skipping'
        )
        return entity_descs, predicate_descs

    # Single batch: entities then predicates, so vectors split cleanly.
    texts: list[str] = []
    entity_entries: list[tuple[int, int]] = []  # (node_id, idx)
    predicate_entries: list[tuple[int, int]] = []

    for node_id, entries in entity_descs.items():
        for idx, entry in enumerate(entries):
            texts.append(
                _embed_text(
                    entry['name'], entry.get('description')
                )
            )
            entity_entries.append((node_id, idx))

    for node_id, entries in predicate_descs.items():
        for idx, entry in enumerate(entries):
            texts.append(
                _embed_text(
                    entry['predicate'], entry.get('description')
                )
            )
            predicate_entries.append((node_id, idx))

    if not texts:
        return entity_descs, predicate_descs

    embedder = embeddings.embedder()
    vectors = await embedder.embed(texts)

    n_entities = len(entity_entries)
    entity_vectors = vectors[:n_entities]
    predicate_vectors = vectors[n_entities:]

    # Shallow-copy the dicts so we don't mutate the originals.
    result_entities: dict[int, list[dict]] = {
        node_id: [dict(entry) for entry in entries]
        for node_id, entries in entity_descs.items()
    }
    for (node_id, idx), vector in zip(
        entity_entries, entity_vectors, strict=True
    ):
        result_entities[node_id][idx]['embedding'] = vector

    result_predicates: dict[int, list[dict]] = {
        node_id: [dict(entry) for entry in entries]
        for node_id, entries in predicate_descs.items()
    }
    for (node_id, idx), vector in zip(
        predicate_entries, predicate_vectors, strict=True
    ):
        result_predicates[node_id][idx]['embedding'] = vector

    logger.info(
        'entity embedder: %d entity + %d predicate embedding(s)',
        n_entities,
        len(predicate_entries),
    )
    return result_entities, result_predicates


# ============================================================================
# LangGraph node
# ============================================================================


class EntityEmbedderNode:
    """Computes embeddings for entity and predicate descriptions.

    Runs after entity enrichment, before the ingestion persister. Reads
    ``node_entity_descriptions`` and ``node_predicate_descriptions``;
    writes them back with an ``embedding`` field on every entry.

    When no embedding API key is configured, the stage is a no-op.
    """

    async def run(self, state: state.State) -> dict:
        """Embed entity and predicate descriptions.

        Args:
            state: The pipeline state.

        Returns:
            The enriched description channels.
        """
        entity_descs = state.get('node_entity_descriptions', {})
        predicate_descs = state.get('node_predicate_descriptions', {})
        enriched_entities, enriched_predicates = (
            await embed_descriptions(entity_descs, predicate_descs)
        )
        return {
            'node_entity_descriptions': enriched_entities,
            'node_predicate_descriptions': enriched_predicates,
        }
