"""Vector search with text query decomposition, reranking, and judging."""

from collections.abc import Callable
from typing import Any

import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import embeddings, llm, models, module, reranker
from kms.graph import queries


def _get_language_model() -> dspy.LM:
    """Return the configured search entry module LM."""
    return llm.module_lm('decompose_judge')


class SearchResult(BaseModel):
    """One retrieved candidate node, with score and relevance decision."""

    text: str = Field(default='', description='Canonical node text.')
    image_paths: list[str] = Field(
        default_factory=list,
        description='Source visual asset paths, in source order.',
    )
    score: float = Field(default=0.0, description='Vector similarity score.')
    judge_relevant: bool | None = Field(
        default=None, description='LLM relevance decision.'
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description='The remaining node properties.',
    )

    @classmethod
    def from_node(cls, node: dict[str, Any], text_field: str) -> 'SearchResult':
        """Build a SearchResult from a vector-search node record."""
        return cls(
            text=node.get(text_field, '') or '',
            image_paths=node.get('image_paths', []) or [],
            score=node.get('score', 0.0),
            properties={
                key: value for key, value in node.items() if key != 'score'
            },
        )


class SearchGroup(BaseModel):
    """The results of one sub-query search."""

    sub_query: str = Field(description='The text that was searched.')
    results: list[SearchResult] = Field(
        description='Matching nodes with text, image evidence, and score.'
    )


class SubQueryPlan(BaseModel):
    """One planned sub-query and the query parts it covers."""

    label: str = Field(description='A concise label for this sub-query.')
    part_indices: list[int] = Field(
        description='Indices of original query parts covered by this sub-query.'
    )


def _render(parts: list[models.TextNodeInput]) -> str:
    return ' '.join(part.node_text.strip() for part in parts).strip()


def _candidate_inputs(
    results: list[SearchResult],
) -> list[models.TextNodeInput]:
    return [
        models.TextNodeInput(
            local_index=index,
            node_type=str(
                result.properties.get('node_type')
                or result.properties.get('type')
                or ''
            ),
            node_text=result.text,
        )
        for index, result in enumerate(results)
    ]


async def search(
    query: models.SearchQuery,
    *,
    index_name: str,
    text_field: str,
    session_factory: Callable,
    source: str | None = None,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    language_model: dspy.LM | None = None,
) -> list[SearchGroup]:
    """Search a vector index using one ordered, text-only SearchQuery."""
    if not isinstance(query, models.SearchQuery):
        raise TypeError('search() requires a models.SearchQuery value')

    search_settings = config.get_settings().stages.search
    top_k = search_settings.top_k if top_k is None else top_k
    rerank_top_n = (
        search_settings.rerank_top_n if rerank_top_n is None else rerank_top_n
    )
    if not embeddings.is_configured():
        raise RuntimeError('Local embedding model is not configured.')

    language_model = language_model or _get_language_model()
    query_parts = list(query.parts)
    decompose_judge = DecomposeJudge(language_model)
    units: list[tuple[str, list[models.TextNodeInput]]] = []
    if await decompose_judge.aforward(parts=query_parts):
        decomposer = QueryDecomposer(language_model)
        plans = await decomposer.aforward(parts=query_parts)
        for plan in plans:
            parts = [query_parts[index] for index in plan.part_indices]
            if parts:
                units.append((plan.label, parts))
    if not units:
        units = [(_render(query_parts), query_parts)]

    embedder = embeddings.embedder()
    relevance_judge = SearchJudge(language_model)
    groups: list[SearchGroup] = []
    for sub_query, parts in units:
        query_vector = await embedder.embed_query(_render(parts))
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
            reranked = await reranker.reranker().rerank(
                query=sub_query,
                documents=[result.text for result in results],
                top_n=rerank_top_n,
            )
            results = [results[item['index']] for item in reranked]
        else:
            results = results[:rerank_top_n]

        if results:
            decisions = await relevance_judge.aforward(
                query=parts,
                candidates=_candidate_inputs(results),
            )
            for decision in decisions:
                results[decision.index].judge_relevant = decision.relevant
        groups.append(SearchGroup(sub_query=sub_query, results=results))
    return groups


class SearchJudgeDecision(BaseModel):
    """The judge's relevance decision for one candidate."""

    index: int = Field(description='The candidate position (0-based).')
    relevant: bool = Field(
        description='True if this candidate helps answer the query.'
    )


class SearchJudgeSignature(dspy.Signature):
    """Judge whether each ordered text candidate addresses the query."""

    query: list[models.TextNodeInput] = dspy.InputField(
        description='The searched query parts, in order.'
    )
    candidates: list[models.TextNodeInput] = dspy.InputField(
        description='The retrieved text candidates, in order.'
    )
    decisions: list[SearchJudgeDecision] = dspy.OutputField(
        description='One relevance decision per candidate.'
    )


class SearchJudge(module.Module):
    """Decide per-candidate relevance against the searched text."""

    signature = SearchJudgeSignature
    record_name = 'search_judge'

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__(language_model or llm.module_lm('search_judge'))

    def encode(
        self,
        query: list[models.TextNodeInput],
        candidates: list[models.TextNodeInput],
    ) -> dict:
        return {'query': query, 'candidates': candidates}

    def decode(self, prediction, **inputs) -> list[SearchJudgeDecision]:
        decisions = module.as_list(prediction.decisions)
        if any(
            not isinstance(decision, SearchJudgeDecision)
            for decision in decisions
        ):
            raise TypeError('decisions must contain SearchJudgeDecision values')
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
    """Decide whether ordered text query parts need decomposition."""

    parts: list[models.TextNodeInput] = dspy.InputField(
        description='The text query parts, in order.'
    )
    should_decompose: bool = dspy.OutputField(
        description='True if the query should be decomposed.'
    )


class DecomposeJudge(module.Module):
    """Decide whether a query should be split into sub-queries."""

    signature = DecomposeJudgeSignature
    record_name = 'decompose_judge'

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__(language_model or llm.module_lm('decompose_judge'))

    def encode(self, parts: list[models.TextNodeInput]) -> dict:
        return {'parts': parts}

    def decode(self, prediction, **inputs) -> bool:
        return module.require_bool(
            prediction.should_decompose, 'should_decompose'
        )


class QueryDecomposerSignature(dspy.Signature):
    """Break ordered text query parts into contiguous sub-query plans."""

    parts: list[models.TextNodeInput] = dspy.InputField(
        description='The text query parts, in order.'
    )
    sub_queries: list[SubQueryPlan] = dspy.OutputField(
        description='Plans covering contiguous query part indices.'
    )


class QueryDecomposer(module.Module):
    """Split a query into independent, self-contained sub-queries."""

    signature = QueryDecomposerSignature
    record_name = 'query_decomposer'

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__(language_model or llm.module_lm('query_decomposer'))

    def encode(self, parts: list[models.TextNodeInput]) -> dict:
        return {'parts': parts}

    def decode(self, prediction, **inputs) -> list[SubQueryPlan]:
        plans = module.as_list(prediction.sub_queries)
        if any(not isinstance(plan, SubQueryPlan) for plan in plans):
            raise TypeError('sub_queries must contain SubQueryPlan values')
        if len(plans) > 4:
            raise ValueError('sub_queries must contain at most four plans')
        part_count = len(inputs['parts'])
        covered: list[int] = []
        previous_end = -1
        for plan_index, plan in enumerate(plans):
            if not plan.label.strip():
                raise ValueError(
                    f'sub_queries[{plan_index}] has an empty label'
                )
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
