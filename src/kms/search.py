"""
Generalized two-tier search over the knowledge graph.

Three-stage pipeline shared by every target type:

    1.  embed   — query vector via ``core.embeddings``
    2.  retrieve — Neo4j vector index, per-target
    3.  rerank  — cross-encoder scores candidates via ``core.reranker``

An optional fourth stage (``judge=True``) runs an LLM adjudicator over
the reranked candidates, filtering out false positives with a
RELEVANT / NOT_RELEVANT decision per candidate.

Specialized entry points (``search_communities``, ``search_entity_hubs``,
``search_predicate_hubs``) call the shared ``search()`` and return
graph-expanded results (connected hubs, triplets, definitions).

Text-only queries are plain strings.  Multimodal queries pass
``{"text": "...", "image": "<url or base64>"}`` — the same shape the
embedder and reranker already accept.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import dspy
from pydantic import BaseModel, Field


async def search(
    query: str | dict[str, Any],
    source: str,
    session_factory: Callable,
    *,
    target: str,
    top_k: int = 20,
    rerank_top_n: int = 5,
    judge: bool = False,
    language_model: dspy.LM | None = None,
) -> list[dict[str, Any]]:
    """Two-tier (optionally three-tier) search over *source*'s graph.

    Args:
        query: Natural-language query (string or ``{"text": ..., "image":
            ...}`` dict for multimodal).
        source: The stable book identity.
        session_factory: Neo4j session factory.
        target: Which index to search — ``'community'``,
            ``'entity_hub'``, or ``'predicate_hub'``.
        top_k: Vector candidates to retrieve.
        rerank_top_n: Final result count after reranking (and
            judging, when enabled).
        judge: If True, run an LLM relevance filter over the reranked
            candidates before returning.
        language_model: The LLM for the judge.  Defaults to the
            shared text LM when None.

    Returns:
        One dict per result:
        ``{uuid, text, score, ...}`` where extra keys depend on
        *target*.  When ``judge=True``, each dict also carries
        ``judge_relevant`` (bool).
    """
    from kms.core import embeddings as _emb
    from kms.core import reranker as _reranker
    from kms.graph import queries

    # --- 1. embed ----------------------------------------------------------
    if not _emb.is_configured():
        raise RuntimeError('Embedding API key not configured.')

    query_input: Any = (
        query if isinstance(query, dict) else query
    )
    embedder = _emb.embedder()
    query_vector = (await embedder.embed([query_input]))[0]

    # --- 2. retrieve -------------------------------------------------------
    if target == 'community':
        candidates = await queries.vector_search_communities(
            session_factory,
            query_embedding=query_vector,
            source=source,
            top_k=top_k,
        )
    elif target in ('entity_hub', 'predicate_hub'):
        kind = 'entity' if target == 'entity_hub' else 'predicate'
        candidates = await queries.vector_search_hub_definitions(
            session_factory,
            query_embedding=query_vector,
            source=source,
            kind=kind,
            top_k=top_k,
        )
    else:
        raise ValueError(f'Unknown search target: {target!r}')

    if not candidates:
        return []

    # --- 3. rerank ---------------------------------------------------------
    if _reranker.is_configured():
        r = _reranker.reranker()
        query_str = (
            query if isinstance(query, str) else query.get('text', '')
        )
        docs = [
            c.get('summary_text') or c.get('definition_text', '')
            for c in candidates
        ]
        reranked = await r.rerank(
            query=query_str,
            documents=docs,
            top_n=rerank_top_n,
        )
        results = [candidates[item['index']] for item in reranked]
    else:
        results = candidates[:rerank_top_n]

    # --- 4. judge (optional) ----------------------------------------------
    if judge and results:
        query_str = (
            query if isinstance(query, str) else query.get('text', '')
        )
        cand_texts = [
            r.get('summary_text') or r.get('definition_text', '')
            for r in results
        ]
        judge_module = SearchJudge(language_model)
        decisions = await judge_module.aforward(query_str, cand_texts)
        # Attach judge verdicts to each candidate
        for decision in decisions:
            idx = decision.index
            if idx < len(results):
                results[idx]['judge_relevant'] = decision.relevant

    return results


# ============================================================================
# DSPy judge — LLM filters reranked candidates
# ============================================================================

default_language_model: dspy.LM | None = None


def _get_lm() -> dspy.LM:
    """The shared language model for the judge, cached lazily."""
    global default_language_model
    if default_language_model is None:
        from kms.core import llm

        default_language_model = llm.text_lm()
    return default_language_model


class SearchJudgeDecision(BaseModel):
    """One candidate's relevance verdict."""

    index: int = Field(description='The candidate position (0-based).')
    relevant: bool = Field(
        description='True if this candidate helps answer the query.'
    )


class SearchJudgeSignature(dspy.Signature):
    """
    You are a search quality judge.  A user asked a query, and a
    retrieval system returned several candidates.  For EACH candidate,
    decide whether it is RELEVANT to answering the query.

    RELEVANT means the candidate contains information that directly
    addresses the query — it defines a concept the query asks about,
    describes a relationship the query mentions, or provides context
    needed to understand the query's topic.

    NOT_RELEVANT means the candidate is about an unrelated topic,
    is too vague to help, or mentions the query's terms only
    incidentally without addressing the actual question.

    Return one decision per candidate.  Be conservative — mark
    RELEVANT only when the candidate genuinely helps.
    """

    query: str = dspy.InputField(description='The user query.')
    candidates: list[str] = dspy.InputField(
        description='The retrieved candidate texts, one per position.'
    )
    decisions: list[SearchJudgeDecision] = dspy.OutputField(
        description='One relevance decision per candidate.'
    )


class SearchJudge(dspy.Module):
    """Filters search results by relevance to the query.

    Args:
        language_model: The LLM to run on.  Defaults to the shared
            text LM when None.
    """

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(SearchJudgeSignature)
        self.set_lm(language_model or _get_lm())

    async def aforward(
        self, query: str, candidates: list[str]
    ) -> list[SearchJudgeDecision]:
        result = await self.judge.acall(
            query=query, candidates=candidates
        )
        return list(result.decisions or [])

    def forward(
        self, query: str, candidates: list[str]
    ) -> list[SearchJudgeDecision]:
        return asyncio.run(self.aforward(query, candidates))


# ============================================================================
# Search pipeline
# ============================================================================


async def search_communities(
    query: str | dict[str, Any],
    source: str,
    session_factory: Callable,
    *,
    top_k: int = 20,
    rerank_top_n: int = 5,
    expand: bool = True,
    judge: bool = False,
    language_model: dspy.LM | None = None,
) -> list[dict[str, Any]]:
    """Search community summaries and optionally expand to member hubs.

    Args:
        query: Natural-language query.
        source: The stable book identity.
        session_factory: Neo4j session factory.
        top_k: Vector candidates.
        rerank_top_n: Results after reranking.
        expand: If True, attach member hubs and evidence triplets.

    Returns:
        One dict per community:
        ``{uuid, summary_text, score, hubs?, triplets?}``.
    """
    results = await search(
        query=query,
        source=source,
        session_factory=session_factory,
        target='community',
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        judge=judge,
        language_model=language_model,
    )

    if not expand:
        return results

    from kms.graph import queries
    from kms.graph.nodes import source_uuid as _su

    # Expand each community: query member hubs + triplets per community
    for result in results:
        community_uuid = result['uuid']

        # Member hubs
        result['hubs'] = await _hubs_in_community(
            community_uuid, session_factory
        )

        # Canonical triplets whose endpoints are all in this community
        hub_triplets = await queries.canonical_hub_triplets(
            session_factory, source=source
        )
        member_uuids = {h['uuid'] for h in result['hubs']}
        result['triplets'] = [
            ht for ht in hub_triplets
            if (ht['subj_hub'] in member_uuids
                and ht['pred_hub'] in member_uuids
                and ht['obj_hub'] in member_uuids)
        ]

    return results


async def search_entity_hubs(
    query: str | dict[str, Any],
    source: str,
    session_factory: Callable,
    *,
    top_k: int = 20,
    rerank_top_n: int = 5,
    expand: bool = True,
    judge: bool = False,
    language_model: dspy.LM | None = None,
) -> list[dict[str, Any]]:
    """Search canonical entity definitions.

    Args:
        query: Natural-language query.
        source: The stable book identity.
        session_factory: Neo4j session factory.
        top_k: Vector candidates.
        rerank_top_n: Results after reranking.
        expand: If True, attach connected triplets.
        judge: If True, run LLM relevance filter over results.

    Returns:
        One dict per hub:
        ``{hub_uuid, display_name, definition_text, score,
          triplets?, judge_relevant?}``.
    """
    results = await search(
        query=query,
        source=source,
        session_factory=session_factory,
        target='entity_hub',
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        judge=judge,
        language_model=language_model,
    )

    if not expand or not results:
        return results

    from kms.graph import queries

    all_triplets = await queries.canonical_hub_triplets(
        session_factory, source=source
    )
    for result in results:
        hub_uuid = result['hub_uuid']
        result['triplets'] = [
            ht for ht in all_triplets
            if ht['subj_hub'] == hub_uuid or ht['obj_hub'] == hub_uuid
        ]

    return results


async def search_predicate_hubs(
    query: str | dict[str, Any],
    source: str,
    session_factory: Callable,
    *,
    top_k: int = 20,
    rerank_top_n: int = 5,
    expand: bool = True,
    judge: bool = False,
    language_model: dspy.LM | None = None,
) -> list[dict[str, Any]]:
    """Search canonical predicate definitions.

    Args:
        query: Natural-language query.
        source: The stable book identity.
        session_factory: Neo4j session factory.
        top_k: Vector candidates.
        rerank_top_n: Results after reranking.
        expand: If True, attach connected triplets.
        judge: If True, run LLM relevance filter over results.

    Returns:
        One dict per hub:
        ``{hub_uuid, display_name, definition_text, score,
          triplets?, judge_relevant?}``.
    """
    results = await search(
        query=query,
        source=source,
        session_factory=session_factory,
        target='predicate_hub',
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        judge=judge,
        language_model=language_model,
    )

    if not expand or not results:
        return results

    from kms.graph import queries

    all_triplets = await queries.canonical_hub_triplets(
        session_factory, source=source
    )
    for result in results:
        hub_uuid = result['hub_uuid']
        result['triplets'] = [
            ht for ht in all_triplets
            if ht['pred_hub'] == hub_uuid
        ]

    return results


# ============================================================================
# Helpers
# ============================================================================


async def _hubs_in_community(
    community_uuid: str,
    session_factory: Callable,
) -> list[dict]:
    """Member hubs (EntityHub + PredicateHub) of one community."""
    from kms.graph import queries as _q
    from kms.graph.nodes import source_uuid as _su

    cypher = (
        'MATCH (c:Community {uuid: $uuid})'
        '-[:HAS_MEMBER]->(h) '
        'OPTIONAL MATCH (h)-[:HAS_DEFINITION]->(d:Definition) '
        'RETURN h.uuid AS uuid, '
        'labels(h) AS labels, '
        'h.display_name AS display_name, '
        'd.text AS definition_text'
    )
    async with session_factory() as session:
        result = await session.run(cypher, uuid=community_uuid)
        return [
            {
                'uuid': record['uuid'],
                'labels': record['labels'],
                'display_name': record['display_name'],
                'definition_text': record.get('definition_text'),
            }
            async for record in result
        ]
