"""
Community detection and summary synthesis over the canonical hub graph.

Runs after canonicalization.  Reads the ``:EntityHub`` / ``:PredicateHub``
graph (connected through ``:Triplet`` hubs), finds densely-connected
groups via connected components, and synthesises a summary paragraph for
each community via an LLM.

The communities are written as ``:Community`` nodes with ``:HAS_MEMBER``
edges to member hubs and ``:COMMUNITY_EVIDENCE`` edges to the underlying
``:Triplet`` hubs.

Design commitments:

* DETERMINISTIC COMPONENTS.  Connected components on the hub adjacency
  graph — no randomness, no seed.  Re-running on the same graph always
  produces the same communities.  Leiden can replace this later if
  needed.

* ONE LLM CALL PER COMMUNITY.  The summary synthesis sees the member hub
  definitions and the canonical triplets within the community.

* COMMUNITIES ARE DERIVED.  Regenerated when the graph changes.  The
  uuid is deterministic from member hub uuids, so re-building is
  idempotent.
"""

import asyncio
import logging
from collections import defaultdict

import dspy

from kms.core import embeddings

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy module — community summary synthesis
# ============================================================================


class CommunitySummarySignature(dspy.Signature):
    """
    You are given a group of related concepts (entities and predicates)
    from a knowledge graph.  Each concept has a canonical definition.
    You are also given the relationships (triplets) that connect these
    concepts within this group.

    Write a 3-5 sentence summary paragraph that describes what this
    group of concepts is collectively about.  The summary should:

    - Identify the main topic or theme that unites these concepts
    - Describe the key relationships among them in prose
    - Be readable standalone — someone reading only this summary should
      understand what this knowledge is about
    - Use LaTeX with $ delimiters for any mathematical notation
    - NEVER invent information that is not supported by the input

    Return ONLY the summary paragraph, no preamble.
    """

    member_definitions: list[str] = dspy.InputField(
        description='The canonical definitions of every concept in '
        'this community.'
    )
    triplets: list[str] = dspy.InputField(
        description='The canonical triplets within this community, '
        'each formatted as "(subject) --[predicate]--> (object)".'
    )
    summary: str = dspy.OutputField(
        description='A 3-5 sentence paragraph summarising this community.'
    )


class CommunitySummarizer(dspy.Module):
    """Synthesises a summary paragraph for one community.

    Args:
        language_model: The LM to run on.
    """

    def __init__(self, language_model: dspy.LM) -> None:
        super().__init__()
        self.summarizer = dspy.ChainOfThought(CommunitySummarySignature)
        self.set_lm(language_model)

    async def aforward(
        self,
        member_definitions: list[str],
        triplets: list[str],
    ) -> str:
        """Synthesize a summary for one community asynchronously.

        Args:
            member_definitions: Canonical definitions of member concepts.
            triplets: Canonical triplets as formatted strings.

        Returns:
            The synthesized summary paragraph.
        """
        result = await self.summarizer.acall(
            member_definitions=member_definitions,
            triplets=triplets,
        )
        return result.summary

    def forward(
        self, member_definitions: list[str], triplets: list[str]
    ) -> str:
        """Synthesize a summary for one community synchronously.

        Wraps :meth:`aforward` in an asyncio event loop.

        Args:
            member_definitions: Canonical definitions of member concepts.
            triplets: Canonical triplets as formatted strings.

        Returns:
            The synthesized summary paragraph.
        """
        return asyncio.run(self.aforward(member_definitions, triplets))


# ============================================================================
# Community detection
# ============================================================================


def _build_hub_adjacency(
    hub_triplets: list[dict],
) -> dict[str, set[str]]:
    """Build an undirected adjacency graph of EntityHubs.

    Two EntityHubs are adjacent if they co-occur as subject and object
    of the same canonical triplet.

    Args:
        hub_triplets: One dict per canonical triplet:
            ``{subj_hub, pred_hub, obj_hub, triplet_uuid}``.

    Returns:
        Adjacency map: hub_uuid → set of adjacent hub_uuids.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    for hub_triplet in hub_triplets:
        subject_hub = hub_triplet['subj_hub']
        object_hub = hub_triplet['obj_hub']
        adjacency[subject_hub].add(object_hub)
        adjacency[object_hub].add(subject_hub)
        # Predicate hubs are connectors — they join communities but
        # are not themselves EntityHubs.  Include them in the adjacency
        # so predicate hubs can bridge entity hubs.
        predicate_hub = hub_triplet['pred_hub']
        adjacency[subject_hub].add(predicate_hub)
        adjacency[predicate_hub].add(subject_hub)
        adjacency[object_hub].add(predicate_hub)
        adjacency[predicate_hub].add(object_hub)
    return dict(adjacency)


def _connected_components(
    adjacency: dict[str, set[str]],
) -> list[set[str]]:
    """Find connected components in an undirected adjacency graph.

    Args:
        adjacency: Adjacency map: node → set of neighbors.

    Returns:
        One set of node uuids per component.
    """
    visited: set[str] = set()
    components: list[set[str]] = []

    for node in adjacency:
        if node not in visited:
            component: set[str] = set()
            stack = [node]
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    component.add(vertex)
                    for neighbor in adjacency.get(vertex, set()):
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append(component)

    return components


# ============================================================================
# Public entry point
# ============================================================================


async def build_communities(
    source: str,
    language_model: dspy.LM,
    session_factory,
) -> list[dict]:
    """Run community detection and summary synthesis.

    Args:
        source: The stable book identity.
        language_model: The LM for summary synthesis.
        session_factory: Neo4j session factory.

    Returns:
        One dict per community:
        ``{community_uuid, member_hub_uuids, triplet_uuids,
          summary_text, summary_embedding}``.
    """
    from kms.graph import queries

    summarizer = CommunitySummarizer(language_model)

    # --- Phase 1: read the canonical hub graph --------------------------
    print('Phase 1 — Reading canonical hub graph...')
    hub_triplets = await queries.canonical_hub_triplets(
        session_factory, source=source
    )
    print(f'  {len(hub_triplets)} canonical triplet(s) at hub level')

    if not hub_triplets:
        print('  No triplets — nothing to build communities from.')
        return []

    # --- Phase 2: community detection -----------------------------------
    print('Phase 2 — Detecting communities...')
    adjacency = _build_hub_adjacency(hub_triplets)
    components = _connected_components(adjacency)
    print(f'  {len(components)} connected component(s)')

    # --- Phase 3: collect per-community data ----------------------------
    print('Phase 3 — Collecting per-community data...')

    communities: list[dict] = []

    for i, component in enumerate(components):
        # Which triplets have all their hubs in this component?
        component_triplets: list[dict] = []
        for hub_triplet in hub_triplets:
            if (
                hub_triplet['subj_hub'] in component
                and hub_triplet['obj_hub'] in component
                and hub_triplet['pred_hub'] in component
            ):
                component_triplets.append(hub_triplet)

        # Collect member hub definitions
        member_uuids = sorted(component)
        member_definitions: list[str] = []
        for hub_triplet in hub_triplets:
            for hub_uuid, definition_key in [
                (hub_triplet['subj_hub'], 'subj_def'),
                (hub_triplet['pred_hub'], 'pred_def'),
                (hub_triplet['obj_hub'], 'obj_def'),
            ]:
                if hub_uuid in component:
                    definition = hub_triplet.get(definition_key)
                    if definition and definition not in member_definitions:
                        member_definitions.append(definition)

        # Build canonical triplet strings
        triplet_strings: list[str] = []
        seen_triplets: set[str] = set()
        for hub_triplet in component_triplets:
            key = (
                f'{hub_triplet["subj_name"]}--'
                f'{hub_triplet["pred_name"]}--'
                f'{hub_triplet["obj_name"]}'
            )
            if key not in seen_triplets:
                seen_triplets.add(key)
                triplet_strings.append(
                    f'({hub_triplet["subj_name"]}) '
                    f'--[{hub_triplet["pred_name"]}]--> '
                    f'({hub_triplet["obj_name"]})'
                )

        triplet_uuids = [
            hub_triplet['triplet_uuid'] for hub_triplet in component_triplets
        ]

        # --- Phase 4: synthesize summary -------------------------------
        print(
            f'  Community {i}: {len(component)} hub(s), '
            f'{len(component_triplets)} triplet(s) — '
            f'synthesizing summary...'
        )

        summary_text = await summarizer.aforward(
            member_definitions=member_definitions,
            triplets=triplet_strings,
        )

        # Embed the summary
        summary_embedding = None
        if embeddings.is_configured():
            embedder = embeddings.embedder()
            summary_embedding = (await embedder.embed([summary_text]))[0]

        from kms.graph import community

        community_uuid = community.community_uuid(source, member_uuids)

        communities.append(
            {
                'community_uuid': community_uuid,
                'member_hub_uuids': member_uuids,
                'triplet_uuids': triplet_uuids,
                'summary_text': summary_text,
                'summary_embedding': summary_embedding,
            }
        )

        print(f'    Summary: {summary_text[:120]}...')

    return communities
