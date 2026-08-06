"""
Entity and predicate canonicalization — clustering by embedding similarity
followed by LLM definition synthesis.

Runs post-ingestion: reads the existing ``:Entity`` and ``:Predicate``
spokes from Neo4j, clusters them by cosine similarity of their
name+description embeddings, writes one ``:EntityHub``/``:PredicateHub``
per cluster, and synthesises a canonical ``:Definition`` from the cluster's
descriptions.

The hub is empty (like ``:Triplet``). Content lives on ``:Definition``.

Design commitments:

* PURE EMBEDDING SIMILARITY. No name-grouping prepass — the embedding
  already encodes both what the thing is called (name) and what it means
  (description). Two entities with different surface forms but the same
  referent will have similar embeddings.

* CONNECTED COMPONENTS. All-pairs cosine similarity above a threshold
  forms an adjacency graph; transitive closure (connected components)
  determines the final clusters.

* CRASH ON MISSING EMBEDDINGS. Every spoke must carry an embedding.
  Spokes without one raise immediately — no fallback, no silent singleton.
  The entity_embedding pipeline stage is a prerequisite.

* ONE LLM CALL PER MULTI-DESCRIPTION CLUSTER. Clusters with 2+ distinct
  descriptions get a synthesised canonical definition. Single-description
  clusters reuse the description verbatim. Empty-description clusters
  fall back to the bare display name.
"""

import asyncio
import logging
from collections import Counter

import dspy
from pydantic import BaseModel, Field

from kms.core import embeddings

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy module
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
        """Synthesise a canonical definition.

        Args:
            concept_name: The display name of the concept.
            descriptions: The individual descriptions from each spoke.

        Returns:
            The synthesised canonical definition text.
        """
        result = await self.synthesizer.acall(
            concept_name=concept_name, descriptions=descriptions
        )
        logger.debug(
            'definition synthesizer: %d descriptions -> 1 definition',
            len(descriptions),
        )
        return result.definition

    def forward(
        self, concept_name: str, descriptions: list[str]
    ) -> str:
        """Sync forward for DSPy optimisers."""
        return asyncio.run(
            self.aforward(concept_name, descriptions)
        )


# ============================================================================
# Clustering
# ============================================================================


def _cluster(
    spokes: list[dict], threshold: float
) -> list[list[dict]]:
    """Cluster spokes by all-pairs cosine similarity → connected components.

    Every spoke must carry an ``embedding`` field (list[float]). Spokes
    without embeddings raise ``ValueError`` immediately.

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

    # Adjacency list from cosine similarity
    adjacency: dict[int, list[int]] = {i: [] for i in range(count)}
    for i in range(count):
        for j in range(i + 1, count):
            if (
                embeddings.cosine_similarity(vectors[i], vectors[j])
                >= threshold
            ):
                adjacency[i].append(j)
                adjacency[j].append(i)

    # Connected components via DFS
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
# Definition synthesis
# ============================================================================


def _most_frequent(values: list[str]) -> str:
    """The most common value in *values*; shortest wins ties.

    Args:
        values: Non-empty list of strings.

    Returns:
        The most frequent value.
    """
    counts = Counter(values)
    max_count = max(counts.values())
    candidates = [
        value for value, count in counts.items() if count == max_count
    ]
    return min(candidates, key=len)


def _display_name(
    cluster: list[dict], name_key: str
) -> str:
    """The display name for a cluster — most frequent surface form.

    Args:
        cluster: One cluster's spoke dicts.
        name_key: ``'name'`` for entities, ``'predicate'`` for predicates.

    Returns:
        The most frequent surface form in the cluster.
    """
    return _most_frequent(
        [spoke[name_key] for spoke in cluster]
    )


async def _define_cluster(
    cluster: list[dict],
    name_key: str,
    source: str,
    synthesizer: DefinitionSynthesizer,
    hub_uuid: str,
) -> dict:
    """Synthesise a canonical definition for one cluster, then embed it.

    Args:
        cluster: One cluster's spoke dicts.
        name_key: ``'name'`` or ``'predicate'``.
        source: The stable book identity (unused; reserved for future
            cross-source canonicalization).
        synthesizer: The definition-writing LLM module.
        hub_uuid: The pre-computed hub uuid for this cluster.

    Returns:
        ``{hub_uuid, display_name, definition_text,
          definition_embedding}``.
    """
    display = _display_name(cluster, name_key)
    descriptions = list({
        spoke['description']
        for spoke in cluster
        if spoke.get('description')
    })

    if len(descriptions) >= 2:
        text = await synthesizer.aforward(display, descriptions)
    elif len(descriptions) == 1:
        text = descriptions[0]
    else:
        text = display

    # Embed the definition text
    if embeddings.is_configured():
        embedder = embeddings.embedder()
        vector = (await embedder.embed([text]))[0]
    else:
        vector = None

    return {
        'hub_uuid': hub_uuid,
        'display_name': display,
        'definition_text': text,
        'definition_embedding': vector,
    }


async def _define_clusters(
    clusters: list[list[dict]],
    name_key: str,
    source: str,
    synthesizer: DefinitionSynthesizer,
    hub_uuid_fn,
) -> list[dict]:
    """Synthesise definitions for every cluster.

    Args:
        clusters: One list of spoke dicts per cluster.
        name_key: ``'name'`` or ``'predicate'``.
        source: The stable book identity.
        synthesizer: The definition-writing LLM module.
        hub_uuid_fn: A callable ``(source, spoke_uuids) -> str`` that
            returns the hub uuid for a cluster.

    Returns:
        One definition dict per cluster.
    """
    definitions: list[dict] = []
    for cluster in clusters:
        spoke_uuids = [spoke['uuid'] for spoke in cluster]
        hub_uuid = hub_uuid_fn(source, spoke_uuids)
        definition = await _define_cluster(
            cluster, name_key, source, synthesizer, hub_uuid
        )
        definitions.append(definition)
    return definitions


# ============================================================================
# Public entry points
# ============================================================================


async def canonicalize_entities(
    spokes: list[dict],
    threshold: float,
    source: str,
    language_model: dspy.LM,
) -> tuple[list[list[dict]], list[dict]]:
    """Cluster entity spokes and synthesise canonical definitions.

    Args:
        spokes: One dict per entity spoke, each with ``uuid``, ``name``,
            ``description`` (optional), and ``embedding``.
        threshold: Minimum cosine similarity for two spokes to be
            considered the same entity.
        source: The stable book identity.
        language_model: The LM for definition synthesis.

    Returns:
        A tuple of ``(clusters, definitions)``.

        ``clusters`` is one list of spoke dicts per cluster — ready for
        ``persist_entity_hubs``.
        ``definitions`` is one dict per cluster:
        ``{hub_uuid, display_name, definition_text,
          definition_embedding}``.
    """
    from kms.graph.entity_hubs import entity_hub_uuid

    clusters = _cluster(spokes, threshold)
    definitions = await _define_clusters(
        clusters,
        name_key='name',
        source=source,
        synthesizer=DefinitionSynthesizer(language_model),
        hub_uuid_fn=entity_hub_uuid,
    )

    singleton_count = sum(
        1 for cluster in clusters if len(cluster) == 1
    )
    multi_count = len(clusters) - singleton_count
    logger.info(
        'entity canonicalization: %d cluster(s) (%d multi-spoke, '
        '%d singleton) from %d spoke(s) at threshold %.2f',
        len(clusters),
        multi_count,
        singleton_count,
        len(spokes),
        threshold,
    )
    return clusters, definitions


async def canonicalize_predicates(
    spokes: list[dict],
    threshold: float,
    source: str,
    language_model: dspy.LM,
) -> tuple[list[list[dict]], list[dict]]:
    """Cluster predicate spokes and synthesise canonical definitions.

    Args:
        spokes: One dict per predicate spoke, each with ``uuid``,
            ``predicate``, ``description`` (optional), and ``embedding``.
        threshold: Minimum cosine similarity for two spokes to be
            considered the same predicate.
        source: The stable book identity.
        language_model: The LM for definition synthesis.

    Returns:
        A tuple of ``(clusters, definitions)``, same shape as
        ``canonicalize_entities``.
    """
    from kms.graph.predicate_hubs import predicate_hub_uuid

    clusters = _cluster(spokes, threshold)
    definitions = await _define_clusters(
        clusters,
        name_key='predicate',
        source=source,
        synthesizer=DefinitionSynthesizer(language_model),
        hub_uuid_fn=predicate_hub_uuid,
    )

    singleton_count = sum(
        1 for cluster in clusters if len(cluster) == 1
    )
    multi_count = len(clusters) - singleton_count
    logger.info(
        'predicate canonicalization: %d cluster(s) (%d multi-spoke, '
        '%d singleton) from %d spoke(s) at threshold %.2f',
        len(clusters),
        multi_count,
        singleton_count,
        len(spokes),
        threshold,
    )
    return clusters, definitions
