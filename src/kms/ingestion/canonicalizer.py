"""
Entity and predicate canonicalization — DIAL-KG–style incremental
clustering with cross-batch alignment.

Four phases, run as a standalone pass after one or more pipeline runs
have populated the graph with :Entity/:Predicate spokes:

  Phase 1 — READ.  Fetch every new spoke (has an embedding, no existing
  :CANONICAL edge) from Neo4j.

  Phase 2 — INTRA-BATCH CLUSTER.  All-pairs cosine similarity →
  connected components, same algorithm as before.  Each cluster gets a
  centroid embedding and a display name (most frequent surface form).

  Phase 3 — CROSS-BATCH ALIGN.  For each cluster, vector-search
  existing :EntityHub/:PredicateHub definitions.  An LLM adjudicates
  each candidate pair: {Merge, New, Hierarchy, Review}.

  Phase 4 — WRITE.  Merge → add :CANONICAL edges to an existing hub
  and regenerate its :Definition.  New → create :EntityHub /
  :PredicateHub + :Definition + :CANONICAL edges.  Review → flag, leave
  uncanonicalized so a human can resolve.

Hubs carry a display_name and (eventually) aliases, updated on merge.
The hub uuid is deterministic-from-create (uuid5(source,
sorted(spoke_uuids))) on first creation and is never recomputed; new
spokes join existing hubs via :CANONICAL edges.

Design commitments:

* CRASH ON MISSING EMBEDDINGS — every spoke must carry an embedding.
* ONE LLM CALL PER CANDIDATE PAIR — adjudication is per (cluster,
  candidate), not per spoke.
* DEFINITION REGENERATION ON MERGE — when a hub gains spokes the
  canonical definition is re-synthesised from the full description set.
"""

import asyncio
import logging
from collections import Counter

import dspy
from pydantic import BaseModel, Field

from kms.core import embeddings

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy modules — definition synthesis (unchanged) + adjudication (new)
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


# ============================================================================
# Adjudication — the LLM decides Merge / New / Hierarchy / Review
# ============================================================================


class AdjudicationSignature(dspy.Signature):
    """
    You are a curator of a knowledge graph. You are given one CLUSTER of
    newly extracted entity (or predicate) mentions from a document, and
    one CANDIDATE hub that already exists in the knowledge graph.  Decide
    what action to take.

    The cluster represents a concept — its display name is the most
    common surface form, and the descriptions are what different readers
    wrote about it.

    The candidate hub already has a canonical definition (and possibly
    aliases — other surface forms that have been merged into it).

    Decide ONE of:

    - Merge: the cluster is the SAME concept as the candidate.  The
      cluster's spokes should be merged into this existing hub.
    - New: the cluster is a DIFFERENT concept.  It should get its own
      new hub.  (The system will try the next candidate, or create a
      new hub if no candidate matches.)
    - Hierarchy: the cluster is RELATED but not identical — the new
      concept is a subtype, instance, or specialisation of the
      candidate (e.g. "$G_1$" is a specific graph, matched against a
      hub about "graph").  Treat as New for now — the hierarchical
      relationship can be added later.
    - Review: you cannot decide with confidence.  Flag for human
      review rather than merging incorrectly.

    Prefer Merge when the cluster and candidate clearly refer to the
    same thing even if the surface forms differ.  Prefer New when they
    refer to different things even if the surface forms happen to
    match.  In ambiguous cases, New is safer than a wrong Merge.
    """

    cluster_name: str = dspy.InputField(
        description='The most common surface form in the new cluster.'
    )
    cluster_descriptions: list[str] = dspy.InputField(
        description='Descriptions from different occurrences of this '
        'concept in the new document.'
    )
    hub_display_name: str = dspy.InputField(
        description='The canonical surface form of the existing hub.'
    )
    hub_definition: str = dspy.InputField(
        description='The canonical definition of the existing hub.'
    )
    hub_aliases: list[str] = dspy.InputField(
        description='Other surface forms that have been merged into '
        'this hub.'
    )
    similarity_score: float = dspy.InputField(
        description='The cosine similarity between the cluster '
        'centroid embedding and the hub definition embedding.'
    )
    decision: str = dspy.OutputField(
        description='Merge | New | Hierarchy | Review'
    )
    reasoning: str = dspy.OutputField(
        description='One sentence explaining your decision.'
    )


class Adjudicator(dspy.Module):
    """Decides whether a cluster belongs to an existing hub.

    Args:
        language_model: The LM to run on.
    """

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(AdjudicationSignature)
        self.set_lm(language_model)

    async def aforward(
        self,
        cluster_name: str,
        cluster_descriptions: list[str],
        hub_display_name: str,
        hub_definition: str,
        hub_aliases: list[str],
        similarity_score: float,
    ) -> tuple[str, str]:
        """Adjudicate one (cluster, candidate) pair.

        Returns:
            ``(decision, reasoning)`` where decision is one of
            Merge / New / Hierarchy / Review.
        """
        result = await self.judge.acall(
            cluster_name=cluster_name,
            cluster_descriptions=cluster_descriptions,
            hub_display_name=hub_display_name,
            hub_definition=hub_definition,
            hub_aliases=hub_aliases,
            similarity_score=similarity_score,
        )
        return result.decision, result.reasoning

    def forward(
        self,
        cluster_name: str,
        cluster_descriptions: list[str],
        hub_display_name: str,
        hub_definition: str,
        hub_aliases: list[str],
        similarity_score: float,
    ) -> tuple[str, str]:
        return asyncio.run(
            self.aforward(
                cluster_name,
                cluster_descriptions,
                hub_display_name,
                hub_definition,
                hub_aliases,
                similarity_score,
            )
        )


# ============================================================================
# Clustering (unchanged from original)
# ============================================================================


def _cluster(
    spokes: list[dict], threshold: float
) -> list[list[dict]]:
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
    candidates = [
        v for v, c in counts.items() if c == max_count
    ]
    return min(candidates, key=len)


def _display_name(cluster: list[dict], name_key: str) -> str:
    return _most_frequent([spoke[name_key] for spoke in cluster])


def _centroid_embedding(cluster: list[dict]) -> list[float]:
    dim = len(cluster[0]['embedding'])
    centroid = [0.0] * dim
    for spoke in cluster:
        for i, v in enumerate(spoke['embedding']):
            centroid[i] += v
    n = len(cluster)
    return [v / n for v in centroid]


def _collect_descriptions(
    cluster: list[dict],
) -> list[str]:
    return list({
        spoke['description']
        for spoke in cluster
        if spoke.get('description')
    })


# ============================================================================
# Phase 3: cross-batch alignment — one cluster against existing hubs
# ============================================================================


async def _align_cluster(
    cluster: list[dict],
    name_key: str,
    source: str,
    adjudicator: Adjudicator,
    session_factory,
    *,
    top_k: int = 5,
    min_score: float = 0.7,
    cross_source: bool = False,
) -> str | None:
    """Decide which existing hub (if any) a cluster belongs to.

    Args:
        cluster: One cluster's spoke dicts.
        name_key: ``'name'`` for entities, ``'predicate'`` for predicates.
        source: The stable book identity.
        adjudicator: The LLM adjudication module.
        session_factory: Neo4j session factory.
        top_k: Max candidate hubs to retrieve.
        min_score: Minimum vector similarity to consider.
        cross_source: If True, search across all sources; if False,
            scope to *source*.

    Returns:
        The existing hub uuid to merge into, or None if the cluster
        should become a new hub.
    """
    from kms.graph import queries

    centroid = _centroid_embedding(cluster)
    display = _display_name(cluster, name_key)
    descriptions = _collect_descriptions(cluster)

    scope = None if cross_source else source

    if name_key == 'name':
        candidates = await queries.candidate_entity_hubs(
            session_factory,
            query_embedding=centroid,
            source=scope,
            top_k=top_k,
            min_score=min_score,
        )
    else:
        candidates = await queries.candidate_predicate_hubs(
            session_factory,
            query_embedding=centroid,
            source=scope,
            top_k=top_k,
            min_score=min_score,
        )

    if not candidates:
        return None  # No existing hubs — definitely new

    for candidate in candidates:
        decision, reasoning = await adjudicator.aforward(
            cluster_name=display,
            cluster_descriptions=descriptions,
            hub_display_name=candidate['display_name'] or display,
            hub_definition=candidate['definition_text'],
            hub_aliases=candidate['aliases'],
            similarity_score=candidate['score'],
        )
        logger.info(
            'adjudication: cluster=%r candidate=%r score=%.3f '
            '-> %s (%s)',
            display,
            candidate['display_name'],
            candidate['score'],
            decision,
            reasoning,
        )
        if decision == 'Merge':
            return candidate['hub_uuid']
        elif decision == 'New':
            continue  # Try next candidate
        elif decision == 'Review':
            # Flag for later — treat as New for now but log it
            logger.warning(
                'Review flagged: cluster=%r candidate=%r',
                display,
                candidate['display_name'],
            )
            continue
        # Hierarchy: treat as New (separate hub), but log for later
        elif decision == 'Hierarchy':
            logger.info(
                'Hierarchy: cluster=%r is a subtype of %r — '
                'creating separate hub',
                display,
                candidate['display_name'],
            )
            continue

    return None  # No candidate matched — new hub


# ============================================================================
# Phase 4: definition synthesis (old) + definition update (new)
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


async def _update_definition(
    existing_definition: str,
    new_descriptions: list[str],
    display_name: str,
    synthesizer: DefinitionSynthesizer,
) -> str:
    """Regenerate the canonical definition when a hub gains spokes.

    Args:
        existing_definition: The current canonical definition.
        new_descriptions: The new spokes' descriptions.
        display_name: The hub's canonical surface form.
        synthesizer: The definition-writing LLM module.

    Returns:
        The updated definition text.
    """
    all_descriptions = [existing_definition] + list(new_descriptions)
    if len(all_descriptions) <= 1:
        return existing_definition
    return await synthesizer.aforward(display_name, all_descriptions)


async def _embed_text(text: str) -> list[float] | None:
    """Embed a single string, returning None when no embedder is
    configured."""
    if not embeddings.is_configured():
        return None
    embedder = embeddings.embedder()
    return (await embedder.embed([text]))[0]


# ============================================================================
# Public entry point — the full four-phase run
# ============================================================================


async def run_canonicalization(
    source: str,
    threshold: float,
    language_model: dspy.LM,
    session_factory,
    *,
    cross_source: bool = False,
    top_k: int = 5,
    min_score: float = 0.7,
    entity_kind: bool = True,
    predicate_kind: bool = True,
) -> dict:
    """Run the full four-phase canonicalization for *source*.

    Args:
        source: The stable book identity.
        threshold: Minimum cosine similarity for intra-batch clustering.
        language_model: The LM for adjudication and definition synthesis.
        session_factory: Neo4j session factory.
        cross_source: If True, search existing hubs across all sources.
        top_k: Max candidate hubs per cluster.
        min_score: Minimum vector similarity for a candidate to be
            considered.
        entity_kind: Whether to canonicalize entities.
        predicate_kind: Whether to canonicalize predicates.

    Returns:
        A dict with keys ``entity`` and ``predicate``, each a dict:
        ``{merged, new_hubs, review_flagged, total_spokes}``.
    """
    from kms.graph import queries, writer

    adjudicator = Adjudicator(language_model)
    synthesizer = DefinitionSynthesizer(language_model)

    result: dict = {}

    # ==================================================================
    # ENTITIES
    # ==================================================================
    if entity_kind:
        print('=' * 60)
        print('ENTITY CANONICALIZATION')
        print('=' * 60)

        # Phase 1 — Read
        spokes = await queries.uncanonicalized_entity_spokes(
            session_factory, source=source
        )
        print(f'\nPhase 1 — Read: {len(spokes)} uncanonicalized entity '
              f'spoke(s)')

        if spokes:
            # Phase 2 — Intra-batch cluster
            clusters = _cluster(spokes, threshold)
            print(f'Phase 2 — Cluster: {len(clusters)} cluster(s) '
                  f'from {len(spokes)} spoke(s) at threshold '
                  f'{threshold}')

            # Phase 3 — Cross-batch align + Phase 4 — Write
            entity_result = await _canonicalize_entities(
                clusters=clusters,
                source=source,
                adjudicator=adjudicator,
                synthesizer=synthesizer,
                session_factory=session_factory,
                cross_source=cross_source,
                top_k=top_k,
                min_score=min_score,
            )
            result['entity'] = entity_result
        else:
            result['entity'] = {
                'merged': 0, 'new_hubs': 0, 'review_flagged': 0,
                'total_spokes': 0,
            }

    # ==================================================================
    # PREDICATES
    # ==================================================================
    if predicate_kind:
        print(f'\n{"=" * 60}')
        print('PREDICATE CANONICALIZATION')
        print('=' * 60)

        spokes = await queries.uncanonicalized_predicate_spokes(
            session_factory, source=source
        )
        print(f'\nPhase 1 — Read: {len(spokes)} uncanonicalized '
              f'predicate spoke(s)')

        if spokes:
            clusters = _cluster(spokes, threshold)
            print(f'Phase 2 — Cluster: {len(clusters)} cluster(s) '
                  f'from {len(spokes)} spoke(s) at threshold '
                  f'{threshold}')

            pred_result = await _canonicalize_predicates(
                clusters=clusters,
                source=source,
                adjudicator=adjudicator,
                synthesizer=synthesizer,
                session_factory=session_factory,
                cross_source=cross_source,
                top_k=top_k,
                min_score=min_score,
            )
            result['predicate'] = pred_result
        else:
            result['predicate'] = {
                'merged': 0, 'new_hubs': 0, 'review_flagged': 0,
                'total_spokes': 0,
            }

    return result


# ============================================================================
# Internal: entity canonicalization loop
# ============================================================================


async def _canonicalize_entities(
    clusters: list[list[dict]],
    source: str,
    adjudicator: Adjudicator,
    synthesizer: DefinitionSynthesizer,
    session_factory,
    *,
    cross_source: bool = False,
    top_k: int = 5,
    min_score: float = 0.7,
) -> dict:
    """Run phases 3 + 4 for entity clusters."""
    from kms.graph import writer
    from kms.graph.entity_hubs import entity_hub_uuid

    merged_count = 0
    new_count = 0
    review_count = 0
    total_spokes = sum(len(c) for c in clusters)

    new_clusters: list[list[dict]] = []
    new_definitions: list[dict] = []

    for i, cluster in enumerate(clusters):
        display = _display_name(cluster, 'name')
        print(f'\n  Cluster {i}: "{display}" '
              f'({len(cluster)} spoke(s))')

        # Phase 3 — align against existing hubs
        existing_hub = await _align_cluster(
            cluster=cluster,
            name_key='name',
            source=source,
            adjudicator=adjudicator,
            session_factory=session_factory,
            top_k=top_k,
            min_score=min_score,
            cross_source=cross_source,
        )

        if existing_hub:
            # Phase 4a — Merge into existing hub
            print(f'    -> Merge into hub {existing_hub[:20]}...')
            new_descriptions = _collect_descriptions(cluster)
            # Fetch current definition text (we don't have it in
            # memory — read it from Neo4j)
            from kms.graph import queries
            candidates = await queries.candidate_entity_hubs(
                session_factory,
                query_embedding=_centroid_embedding(cluster),
                source=None if cross_source else source,
                top_k=1,
                min_score=0.0,
            )
            old_def = (
                candidates[0]['definition_text']
                if candidates else display
            )
            updated_text = await _update_definition(
                existing_definition=old_def,
                new_descriptions=new_descriptions,
                display_name=display,
                synthesizer=synthesizer,
            )
            updated_embedding = await _embed_text(updated_text)
            spoke_uuids = [s['uuid'] for s in cluster]
            await writer.persist_canonical_merge(
                spoke_uuids=spoke_uuids,
                hub_uuid=existing_hub,
                hub_type='entity',
                definition_text=updated_text,
                definition_embedding=updated_embedding,
                session_factory=session_factory,
            )
            merged_count += 1
        else:
            # Phase 4b — New hub
            print(f'    -> New hub')
            spoke_uuids = [s['uuid'] for s in cluster]
            hub_uuid = entity_hub_uuid(source, spoke_uuids)
            definition_text = await _synthesize_definition(
                cluster, 'name', synthesizer
            )
            definition_embedding = await _embed_text(definition_text)
            new_clusters.append(cluster)
            new_definitions.append({
                'hub_uuid': hub_uuid,
                'display_name': display,
                'definition_text': definition_text,
                'definition_embedding': definition_embedding,
            })
            new_count += 1

    # Write new hubs in one batch
    if new_clusters:
        await writer.persist_entity_hubs(
            new_clusters,
            new_definitions,
            source,
            session_factory=session_factory,
            definitions=new_definitions,
        )

    print(f'\n  Entity result: {merged_count} merged, {new_count} new '
          f'hub(s), {review_count} flagged, {total_spokes} total spoke(s)')
    return {
        'merged': merged_count,
        'new_hubs': new_count,
        'review_flagged': review_count,
        'total_spokes': total_spokes,
    }


# ============================================================================
# Internal: predicate canonicalization loop
# ============================================================================


async def _canonicalize_predicates(
    clusters: list[list[dict]],
    source: str,
    adjudicator: Adjudicator,
    synthesizer: DefinitionSynthesizer,
    session_factory,
    *,
    cross_source: bool = False,
    top_k: int = 5,
    min_score: float = 0.7,
) -> dict:
    """Run phases 3 + 4 for predicate clusters."""
    from kms.graph import writer
    from kms.graph.predicate_hubs import predicate_hub_uuid

    merged_count = 0
    new_count = 0
    review_count = 0
    total_spokes = sum(len(c) for c in clusters)

    new_clusters: list[list[dict]] = []
    new_definitions: list[dict] = []

    for i, cluster in enumerate(clusters):
        display = _display_name(cluster, 'predicate')
        print(f'\n  Cluster {i}: "{display}" '
              f'({len(cluster)} spoke(s))')

        existing_hub = await _align_cluster(
            cluster=cluster,
            name_key='predicate',
            source=source,
            adjudicator=adjudicator,
            session_factory=session_factory,
            top_k=top_k,
            min_score=min_score,
            cross_source=cross_source,
        )

        if existing_hub:
            print(f'    -> Merge into hub {existing_hub[:20]}...')
            new_descriptions = _collect_descriptions(cluster)
            from kms.graph import queries
            candidates = await queries.candidate_predicate_hubs(
                session_factory,
                query_embedding=_centroid_embedding(cluster),
                source=None if cross_source else source,
                top_k=1,
                min_score=0.0,
            )
            old_def = (
                candidates[0]['definition_text']
                if candidates else display
            )
            updated_text = await _update_definition(
                existing_definition=old_def,
                new_descriptions=new_descriptions,
                display_name=display,
                synthesizer=synthesizer,
            )
            updated_embedding = await _embed_text(updated_text)
            spoke_uuids = [s['uuid'] for s in cluster]
            await writer.persist_canonical_merge(
                spoke_uuids=spoke_uuids,
                hub_uuid=existing_hub,
                hub_type='predicate',
                definition_text=updated_text,
                definition_embedding=updated_embedding,
                session_factory=session_factory,
            )
            merged_count += 1
        else:
            print(f'    -> New hub')
            spoke_uuids = [s['uuid'] for s in cluster]
            hub_uuid = predicate_hub_uuid(source, spoke_uuids)
            definition_text = await _synthesize_definition(
                cluster, 'predicate', synthesizer
            )
            definition_embedding = await _embed_text(definition_text)
            new_clusters.append(cluster)
            new_definitions.append({
                'hub_uuid': hub_uuid,
                'display_name': display,
                'definition_text': definition_text,
                'definition_embedding': definition_embedding,
            })
            new_count += 1

    if new_clusters:
        await writer.persist_predicate_hubs(
            new_clusters,
            new_definitions,
            source,
            session_factory=session_factory,
            definitions=new_definitions,
        )

    print(f'\n  Predicate result: {merged_count} merged, {new_count} '
          f'new hub(s), {review_count} flagged, {total_spokes} '
          f'total spoke(s)')
    return {
        'merged': merged_count,
        'new_hubs': new_count,
        'review_flagged': review_count,
        'total_spokes': total_spokes,
    }
