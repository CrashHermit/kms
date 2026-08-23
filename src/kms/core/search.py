"""Vector search with query decomposition, reranking, and judging."""

from collections.abc import Callable
from typing import Any

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import content, embeddings, llm, module, reranker
from kms.graph import queries


def _get_language_model() -> dspy.LM:
    """Returns the search entry module's configured LM.

    This is a testable seam for callers that do not inject a model. It is
    module configuration, not a shared pipeline role.
    """
    return llm.module_lm('decompose_judge')


class SearchResult(BaseModel):
    """One retrieved candidate node, with score and relevance decision."""

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
        """Builds a SearchResult from a vector-search node record."""
        return cls(
            text=node.get(text_field, '') or '',
            image_path=node.get('image_path'),
            score=node.get('score', 0.0),
            properties={
                key: value for key, value in node.items() if key != 'score'
            },
        )


class SearchGroup(BaseModel):
    """The results of one (sub-)query search."""

    sub_query: str = Field(description='The text that was searched.')
    results: list[SearchResult] = Field(
        description='Matching nodes with their text, image, and score.'
    )


def _result_content(result: SearchResult) -> content.Content:
    """Builds one candidate's ordered text-and-image content."""
    parts: list[content.TextPart | content.ImagePart] = []
    if result.text:
        parts.append(content.TextPart(text=result.text))
    image = content.load_image(
        result.image_path,
        max_dim=config.get_settings().image.max_dim,
    )
    if image:
        parts.append(content.ImagePart(image=image))
    return content.Content(parts=parts)


def _candidate_content(results: list[SearchResult]) -> content.Content:
    """Builds labelled multimodal content for the relevance judge."""
    parts: list[content.TextPart | content.ImagePart] = []
    for index, result in enumerate(results):
        parts.append(content.TextPart(text=f'Candidate {index}:'))
        parts.extend(_result_content(result).parts)
    return content.Content(parts=parts)


class SubQueryPlan(BaseModel):
    """One planned sub-query: a label and the query parts it covers."""

    label: str = Field(description='A concise label for this sub-query.')
    part_indices: list[int] = Field(
        description='Indices of the original query parts that make up '
        'this sub-query.'
    )


async def search(
    query: str | content.Content | list[str | dspy.Image],
    *,
    index_name: str,
    text_field: str,
    session_factory: Callable,
    source: str | None = None,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    language_model: dspy.LM | None = None,
) -> list[SearchGroup]:
    """Searches a vector index, decomposing the query when warranted.

    A multimodal judge decides whether the query should be decomposed into
    independent sub-queries. Each Content sub-query is embedded and searched
    separately with the full top_k budget, then candidate Content values are
    reranked and filtered through a multimodal LLM relevance judge. OpenRouter's
    rerank endpoint accepts a text query and multimodal documents, so image
    query intent is first described by QueryTextifier for that boundary.

    Args:
        query: A text string, canonical ``Content`` value, or ordered list
            of strings and dspy.Image objects. A query may be text-only,
            image-only, or a mix.
        index_name: Neo4j vector index name (e.g. 'node_content',
            'entity_hub_embedding').
        text_field: Property name on each returned node to use as
            the text source for reranking and judging.
        session_factory: Async callable returning a Neo4j session.
        source: Optional source filter. Only used when the underlying
            vector search supports source filtering.
        top_k: Number of candidates to retrieve per search; defaults to
            the configured ``stages.search.top_k``.
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

    search_settings = config.get_settings().stages.search
    if top_k is None:
        top_k = search_settings.top_k
    if rerank_top_n is None:
        rerank_top_n = search_settings.rerank_top_n

    if not embeddings.is_configured():
        raise RuntimeError('Embedding API key not configured.')

    if isinstance(query, content.Content):
        query_content = query
    else:
        query_parts: list[str | dspy.Image] = (
            [query] if isinstance(query, str) else query
        )
        query_content = content.Content.from_parts(query_parts)
    if not query_content.parts:
        raise ValueError('query must contain at least one content part.')
    rendered = query_content.render()
    language_model = language_model or _get_language_model()

    decompose_judge = DecomposeJudge(language_model)
    units: list[tuple[str, content.Content]] = []
    if await decompose_judge.aforward(parts=query_content):
        decomposer = QueryDecomposer(language_model)
        plans = await decomposer.aforward(parts=query_content)
        for plan in plans:
            parts = [query_content.parts[index] for index in plan.part_indices]
            if parts:
                units.append((plan.label, content.Content(parts=parts)))
    if not units:
        units = [(rendered, query_content)]

    embedder = embeddings.embedder()
    relevance_judge = SearchJudge(language_model)
    query_textifier = QueryTextifier(language_model)
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
            rerank_query = sub_query
            if any(
                isinstance(part, content.ImagePart)
                for part in part_content.parts
            ):
                rerank_query = await query_textifier.aforward(
                    parts=part_content
                )
            reranked = await reranker_instance.rerank(
                query=rerank_query,
                documents=[_result_content(result) for result in results],
                top_n=rerank_top_n,
            )
            results = [results[item['index']] for item in reranked]
        else:
            results = results[:rerank_top_n]

        if results:
            decisions = await relevance_judge.aforward(
                parts=part_content, candidates=results
            )
            for decision in decisions:
                results[decision.index].judge_relevant = decision.relevant

        groups.append(SearchGroup(sub_query=sub_query, results=results))

    return groups


class QueryTextifierSignature(dspy.Signature):
    """Describe multimodal query content for a text-query reranker."""

    parts: content.ContentParts = dspy.InputField(
        description='The complete text and image query content.'
    )
    query: str = dspy.OutputField(
        description='A concise text query preserving the visual intent.'
    )


class QueryTextifier(module.Module):
    """Converts visual query intent to text for APIs with text-only queries."""

    signature = QueryTextifierSignature
    record_name = 'query_textifier'

    def encode(self, parts: content.Content) -> dict:
        return {'parts': content.ContentParts(content=parts)}

    def decode(self, prediction, **inputs) -> str:
        return module.require_text(prediction.query, 'query')


class SearchJudgeDecision(BaseModel):
    """The judge's relevance decision for one candidate."""

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


class SearchJudge(module.Module):
    """Decides per-candidate relevance against the searched query."""

    signature = SearchJudgeSignature
    record_name = 'search_judge'

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__(language_model or llm.module_lm('search_judge'))

    def encode(
        self,
        parts: content.Content,
        candidates: list[SearchResult],
    ) -> dict:
        """Builds the search-judge signature kwargs."""
        return {
            'parts': content.ContentParts(content=parts),
            'candidates': content.ContentParts(
                content=_candidate_content(candidates)
            ),
        }

    def decode(self, prediction, **inputs) -> list[SearchJudgeDecision]:
        """Returns validated relevance decisions for retrieved candidates."""
        decisions = module.as_list(prediction.decisions)
        if any(
            not isinstance(decision, SearchJudgeDecision)
            for decision in decisions
        ):
            raise TypeError(
                'decisions must contain SearchJudgeDecision values'
            )
        candidate_count = len(inputs['candidates'])
        if len(decisions) != candidate_count:
            raise ValueError(
                'decisions must contain exactly one item per candidate'
            )
        module.require_positions(
            [decision.index for decision in decisions],
            field_name='decisions.index',
            upper_bound=candidate_count,
            ordered=True,
        )
        for decision in decisions:
            module.require_bool(decision.relevant, 'decisions.relevant')
        return decisions


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


class DecomposeJudge(module.Module):
    """Decides whether a query should be split into sub-queries."""

    signature = DecomposeJudgeSignature
    record_name = 'decompose_judge'

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__(language_model or llm.module_lm('decompose_judge'))

    def encode(self, parts: content.Content) -> dict:
        """Builds the decompose-judge signature kwargs."""
        return {'parts': content.ContentParts(content=parts)}

    def decode(self, prediction, **inputs) -> bool:
        """Returns whether the query should be decomposed."""
        return module.require_bool(
            prediction.should_decompose, 'should_decompose'
        )


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


class QueryDecomposer(module.Module):
    """Splits a query into independent, self-contained sub-queries."""

    signature = QueryDecomposerSignature
    record_name = 'query_decomposer'

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__(language_model or llm.module_lm('query_decomposer'))

    def encode(self, parts: content.Content) -> dict:
        """Builds the query-decomposer signature kwargs."""
        return {'parts': content.ContentParts(content=parts)}

    def decode(self, prediction, **inputs) -> list[SubQueryPlan]:
        """Returns validated sub-query plans for the query parts."""
        plans = module.as_list(prediction.sub_queries)
        if any(not isinstance(plan, SubQueryPlan) for plan in plans):
            raise TypeError('sub_queries must contain SubQueryPlan values')
        if len(plans) > 4:
            raise ValueError('sub_queries must contain at most four plans')
        part_count = len(inputs['parts'].parts)
        covered: list[int] = []
        previous_end = -1
        for plan_index, plan in enumerate(plans):
            if not plan.label.strip():
                raise ValueError(f'sub_queries[{plan_index}] has an empty label')
            indices = module.require_positions(
                plan.part_indices,
                field_name=f'sub_queries[{plan_index}].part_indices',
                upper_bound=part_count,
                ordered=True,
            )
            if not indices or indices[0] != previous_end + 1:
                raise ValueError(
                    'sub-query plans must cover contiguous, non-overlapping '
                    'query parts in order'
                )
            covered.extend(indices)
            previous_end = indices[-1]
        if covered != list(range(part_count)):
            raise ValueError(
                'sub-query plans must cover every query part exactly once'
            )
        return plans
