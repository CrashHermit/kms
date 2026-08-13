import asyncio
from collections.abc import Callable
from typing import Any

import dspy
from pydantic import BaseModel, Field

from kms.core import content, embeddings, llm, reranker
from kms.graph import queries


class SearchResult(BaseModel):
    text: str = Field(default='', description='The node text used for judging.')
    image_path: str | None = Field(
        default=None, description='Path to the node image, if any.'
    )
    score: float = Field(
        default=0.0, description='The vector similarity score.'
    )
    judge_relevant: bool | None = Field(
        default=None, description='LLM relevance decision.'
    )
    properties: dict[str, Any] = Field(
        default_factory=dict, description='The remaining node properties.'
    )

    @classmethod
    def from_node(cls, node: dict[str, Any], text_field: str) -> 'SearchResult':
        return cls(
            text=node.get(text_field, '') or '',
            image_path=node.get('image_path'),
            score=node.get('score', 0.0),
            properties={
                key: value for key, value in node.items() if key != 'score'
            },
        )


class SearchGroup(BaseModel):
    sub_query: str = Field(description='The text that was searched.')
    results: list[SearchResult] = Field(
        description='Matching nodes with their text, image, and score.'
    )


def _candidate_content(results: list[SearchResult]) -> content.Content:
    parts: list[content.TextPart | content.ImagePart] = []
    for index, result in enumerate(results):
        parts.append(content.TextPart(text=f'Candidate {index}:'))
        if result.text:
            parts.append(content.TextPart(text=result.text))
        image = content.load_image(result.image_path)
        if image:
            parts.append(content.ImagePart(image=image))
    return content.Content(parts=parts)


class SubQueryPlan(BaseModel):
    label: str = Field(description='A concise label for this sub-query.')
    part_indices: list[int] = Field(
        description='Indices of the original query parts that make up '
        'this sub-query.'
    )


async def search(
    query: str | list[str | dspy.Image],
    *,
    index_name: str,
    text_field: str,
    session_factory: Callable,
    source: str | None = None,
    top_k: int = 20,
    rerank_top_n: int = 5,
    language_model: dspy.LM | None = None,
) -> list[SearchGroup]:
    """Searches a vector index, decomposing the query when warranted.

    A judge always decides whether the query should be decomposed into
    independent sub-queries.  Each part — the raw query or one
    sub-query — is embedded and searched separately with the full
    top_k budget, then optionally reranked and filtered through an
    LLM relevance judge.

    Args:
        query: Natural-language query string, or an ordered list of
            strings and dspy.Image objects for a multimodal query.
            A query may be text-only, image-only, or a mix.
        index_name: Neo4j vector index name (e.g. 'node_content',
            'entity_hub_embedding').
        text_field: Property name on each returned node to use as
            the text source for reranking and judging.
        session_factory: Async callable returning a Neo4j session.
        source: Optional source filter. Only used when the underlying
            vector search supports source filtering.
        top_k: Number of candidates to retrieve per search.
        rerank_top_n: Number of candidates to keep after reranking.
        language_model: DSPy LM for the judge, decomposer, and
            relevance judge.  Created fresh if None.

    Returns:
        A list of SearchGroup, one per search part.  Each group has a
        'sub_query' with the text that was searched and a 'results'
        list of SearchResult (text, image_path, score, judge_relevant,
        and the remaining node properties).

    Raises:
        ValueError: If query is empty.
        RuntimeError: If the embedding API key is not configured.
    """
    if not query:
        raise ValueError('query must be a non-empty string or list.')

    if not embeddings.is_configured():
        raise RuntimeError('Embedding API key not configured.')

    query_parts: list[str | dspy.Image] = (
        [query] if isinstance(query, str) else query
    )
    query_content = content.Content.from_parts(query_parts)
    rendered = query_content.render()
    language_model = language_model or _get_language_model()

    decompose_judge = DecomposeJudge(language_model)
    units: list[tuple[str, content.Content]] = []
    if await decompose_judge.aforward(query_content):
        decomposer = QueryDecomposer(language_model)
        plans = await decomposer.aforward(query_content)
        for plan in plans:
            part_set = [
                query_content.parts[index]
                for index in plan.part_indices
                if 0 <= index < len(query_content.parts)
            ]
            if part_set:
                units.append((plan.label, content.Content(parts=part_set)))
    if not units:
        units = [(rendered, query_content)]

    embedder = embeddings.embedder()
    relevance_judge = SearchJudge(language_model)
    groups: list[SearchGroup] = []
    for sub_query, part_content in units:
        query_vector = (await embedder.embed([part_content]))[0]
        candidates = await queries.vector_search(
            session_factory,
            index_name=index_name,
            query_embedding=query_vector,
            top_k=top_k,
            source=source,
        )
        if not candidates:
            groups.append(SearchGroup(sub_query=sub_query, results=[]))
            continue

        results = [
            SearchResult.from_node(candidate, text_field)
            for candidate in candidates
        ]
        if reranker.is_configured():
            reranker_instance = reranker.reranker()
            docs = [result.text for result in results]
            query_images = [
                part
                for part in part_content.parts
                if isinstance(part, content.ImagePart)
            ]
            if query_images:
                image = content.image_url(query_images[0].image)
                docs = [{'text': doc, 'image': image} for doc in docs]
            reranked = await reranker_instance.rerank(
                query=sub_query,
                documents=docs,
                top_n=rerank_top_n,
            )
            results = [results[item['index']] for item in reranked]
        else:
            results = results[:rerank_top_n]

        if results:
            candidates_content = content.ContentParts(
                content=_candidate_content(results)
            )
            decisions = await relevance_judge.aforward(
                part_content, candidates_content
            )
            for decision in decisions:
                if decision.index < len(results):
                    results[decision.index].judge_relevant = decision.relevant

        groups.append(SearchGroup(sub_query=sub_query, results=results))

    return groups


default_language_model: dspy.LM | None = None


def _get_language_model() -> dspy.LM:
    """Returns the default language model, creating it lazily.

    All pipeline models are multimodal, so the same model handles
    text-only and image queries.

    Returns:
        The language model for the search.
    """
    global default_language_model
    if default_language_model is None:
        default_language_model = llm.pipeline_lm()
    return default_language_model


class SearchJudgeDecision(BaseModel):
    index: int = Field(description='The candidate position (0-based).')
    relevant: bool = Field(
        description='True if this candidate helps answer the query.'
    )


class SearchJudgeSignature(dspy.Signature):
    """
    You are a search quality judge.  A user asked a query — a list of
    text parts and images — and a retrieval system returned several
    candidates.  For EACH candidate, decide whether it is RELEVANT to
    answering the query.

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

    parts: content.ContentParts = dspy.InputField(
        description='The searched query parts, in order.'
    )
    candidates: content.ContentParts = dspy.InputField(
        description='The retrieved candidates in order, each labelled '
        '"Candidate N:", followed by its text and any images.'
    )
    decisions: list[SearchJudgeDecision] = dspy.OutputField(
        description='One relevance decision per candidate.'
    )


class SearchJudge(dspy.Module):
    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.judge = dspy.Predict(SearchJudgeSignature)
        self.set_lm(language_model or _get_language_model())

    async def aforward(
        self,
        parts: content.Content,
        candidates: content.ContentParts,
    ) -> list[SearchJudgeDecision]:
        result = await self.judge.acall(
            parts=content.ContentParts(content=parts), candidates=candidates
        )
        return list(result.decisions or [])

    def forward(
        self,
        parts: content.Content,
        candidates: content.ContentParts,
    ) -> list[SearchJudgeDecision]:
        return asyncio.run(self.aforward(parts, candidates))


class DecomposeJudgeSignature(dspy.Signature):
    """
    You are given a QUERY for a knowledge-base search as a list of
    parts in order.  Each part is either text or an image.  Decide
    whether the query should be DECOMPOSED into several independent
    sub-queries or sent to search as-is (RAW).

    DECOMPOSE (True) when the query asks about multiple distinct
    concepts, names several entities or relationships, combines
    several ideas that would each match different stored knowledge
    units, or spans several images that should be searched
    separately.  Splitting improves recall because each part gets its
    own focused embedding.

    RAW (False) when the query is atomic — a single concept, entity,
    or relationship — or is short and self-contained.

    Answer only True or False.
    """

    parts: content.ContentParts = dspy.InputField(
        description='The search query parts, in order.'
    )
    should_decompose: bool = dspy.OutputField(
        description='True if the query should be decomposed.'
    )


class DecomposeJudge(dspy.Module):
    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.judge = dspy.Predict(DecomposeJudgeSignature)
        self.set_lm(language_model or _get_language_model())

    async def aforward(self, parts: content.Content) -> bool:
        result = await self.judge.acall(
            parts=content.ContentParts(content=parts)
        )
        return result.should_decompose

    def forward(self, parts: content.Content) -> bool:
        return asyncio.run(self.aforward(parts))


class QueryDecomposerSignature(dspy.Signature):
    """
    You are given a QUERY for a knowledge-base search as a list of
    parts in order.  Each part is either text or an image.  Break the
    query into independent sub-queries, each covering a CONTIGUOUS
    group of one or more parts.

    Rules:
    - Each sub-query must be self-contained and independently
      searchable: resolve pronouns and abbreviations, keep any
      conditions it needs, and include every image part that belongs
      to it by listing its index.
    - Part indices are 0-based positions in the input list.
    - Every part belongs to exactly one sub-query.
    - Do not invent content the query did not contain.
    - Return at most 4 sub-queries.
    """

    parts: content.ContentParts = dspy.InputField(
        description='The query parts, in order.'
    )
    sub_queries: list[SubQueryPlan] = dspy.OutputField(
        description='One plan per sub-query: a label and the part '
        'indices it covers.'
    )


class QueryDecomposer(dspy.Module):
    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.decomposer = dspy.Predict(QueryDecomposerSignature)
        self.set_lm(language_model or _get_language_model())

    async def aforward(self, parts: content.Content) -> list[SubQueryPlan]:
        result = await self.decomposer.acall(
            parts=content.ContentParts(content=parts)
        )
        return list(result.sub_queries or [])

    def forward(self, parts: content.Content) -> list[SubQueryPlan]:
        return asyncio.run(self.aforward(parts))
