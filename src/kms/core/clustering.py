import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence

from kms.core import batching, context_window, embeddings, logs, reranker

logger = logging.getLogger(__name__)


async def select_reranked_candidates[T](
    query: str,
    candidates: Sequence[T],
    render_document: Callable[[T], str],
    *,
    top_n: int,
) -> list[T]:
    """Returns only the reranker-selected candidates, in relevance order.

    Renders the query, renders each candidate document, and invokes the
    local reranker to score relevance, returning only the reranker-selected
    subset (at most ``top_n``) in relevance-score order. When the reranker
    is not configured, the input candidate order is retained in full so
    downstream threshold or adjudication logic sees the complete set.

    Reranker-omitted candidates are intentionally dropped here; callers that
    need to preserve them (e.g. multi-pivot clustering loops) must re-queue
    them separately.

    An empty candidate input returns an empty list without contacting the
    reranker.
    """
    if not candidates:
        return []
    if not reranker.is_configured():
        logger.info(
            'reranker not configured: %d candidate(s) retained',
            len(candidates),
        )
        return list(candidates)
    documents = [render_document(candidate) for candidate in candidates]
    results = await reranker.reranker().rerank(query, documents, top_n=top_n)
    ranked_indices = [result['index'] for result in results]
    logger.info(
        'reranker: %d candidate(s) -> %d selected (top_n=%d)',
        len(candidates),
        len(ranked_indices),
        top_n,
    )
    return [candidates[index] for index in ranked_indices]


def coarse_groups(
    records: list[dict],
    recall_threshold: float,
) -> list[list[dict]]:
    adjacency = {index: [] for index in range(len(records))}
    for left_index, left_record in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            if (
                embeddings.cosine_similarity(
                    left_record.embedding, records[right_index].embedding
                )
                >= recall_threshold
            ):
                adjacency[left_index].append(right_index)
                adjacency[right_index].append(left_index)

    groups: list[list[dict]] = []
    visited: set[int] = set()
    for start_index in range(len(records)):
        if start_index in visited:
            continue
        group: list[dict] = []
        pending = [start_index]
        while pending:
            record_index = pending.pop()
            if record_index in visited:
                continue
            visited.add(record_index)
            group.append(records[record_index])
            pending.extend(adjacency[record_index])
        groups.append(group)
    return groups


def central_record(group: list[dict]) -> dict:
    if len(group) == 1:
        return group[0]
    best_record = group[0]
    best_score = -1.0
    for candidate in group:
        others = [record for record in group if record is not candidate]
        score = sum(
            embeddings.cosine_similarity(candidate.embedding, other.embedding)
            for other in others
        ) / len(others)
        if score > best_score:
            best_record = candidate
            best_score = score
    return best_record


async def adjudicated_groups(
    records: list[dict],
    *,
    recall_threshold: float,
    merge_above: float,
    separate_below: float,
    adjudicate: Callable[[dict, dict], Awaitable[bool]],
    max_concurrency: int,
    comparison_token_budget: int = 4096,
    rerank_top_n: int | None = None,
) -> list[list[dict]]:
    """Clusters by embeddings and adjudicates boundary pairs in token waves."""
    gate = asyncio.Semaphore(max_concurrency)
    groups: list[list[dict]] = []
    automatic_merges = 0
    automatic_separations = 0
    boundary_pairs = 0
    wave_count = 0
    largest_wave_tokens = 0
    for coarse_group in coarse_groups(records, recall_threshold):
        unassigned = list(coarse_group)
        while unassigned:
            pivot = central_record(unassigned)
            candidates = [
                record for record in unassigned if record is not pivot
            ]
            if rerank_top_n:
                selected = await select_reranked_candidates(
                    getattr(pivot, 'description', '') or '',
                    candidates,
                    lambda record: getattr(record, 'description', '') or '',
                    top_n=rerank_top_n,
                )
                selected_ids = {id(record) for record in selected}
                omitted = [
                    record
                    for record in candidates
                    if id(record) not in selected_ids
                ]
                candidates = selected
            else:
                omitted = []
            unassigned = []
            current_group = [pivot]
            boundary_candidates: list[dict] = []
            for record in candidates:
                score = embeddings.cosine_similarity(
                    pivot.embedding, record.embedding
                )
                if score >= merge_above:
                    classification = 'merge'
                    current_group.append(record)
                    automatic_merges += 1
                elif score <= separate_below:
                    classification = 'separate'
                    unassigned.append(record)
                    automatic_separations += 1
                else:
                    classification = 'boundary'
                    boundary_candidates.append(record)
                logger.debug(
                    'clustering embedding: pivot=%s; candidate=%s; '
                    'cosine=%.4f; classification=%s',
                    logs.elide(getattr(pivot, 'description', '') or ''),
                    logs.elide(getattr(record, 'description', '') or ''),
                    score,
                    classification,
                )

            # Reranker-omitted candidates are returned to unassigned so they
            # are considered against a later pivot rather than discarded.
            unassigned.extend(omitted)

            def token_cost(record: dict, pivot_record: dict = pivot) -> int:
                return context_window.estimate_text_tokens(
                    getattr(pivot_record, 'description', '')
                ) + context_window.estimate_text_tokens(
                    getattr(record, 'description', '')
                )

            boundary_pairs += len(boundary_candidates)
            for wave in batching.token_batches(
                boundary_candidates,
                token_cost=token_cost,
                token_budget=comparison_token_budget,
            ):
                wave_count += 1
                wave_tokens = sum(token_cost(record) for record in wave)
                largest_wave_tokens = max(largest_wave_tokens, wave_tokens)

                async def decide(
                    record: dict, pivot_record: dict = pivot
                ) -> bool:
                    async with gate:
                        outcome = await adjudicate(pivot_record, record)
                        logger.debug(
                            'clustering adjudication: pivot=%s; '
                            'candidate=%s; decision=%s',
                            logs.elide(
                                getattr(pivot_record, 'description', '') or ''
                            ),
                            logs.elide(
                                getattr(record, 'description', '') or ''
                            ),
                            'merge' if outcome else 'separate',
                        )
                        return outcome

                decisions = await asyncio.gather(
                    *(decide(record) for record in wave)
                )
                for record, should_merge in zip(wave, decisions, strict=True):
                    if should_merge:
                        current_group.append(record)
                    else:
                        unassigned.append(record)
            groups.append(current_group)
    logger.info(
        'adjudicated clustering: %d records -> %d groups; '
        '%d automatic merges, %d automatic separations, %d boundary pairs, '
        '%d waves, largest wave %d estimated tokens',
        len(records),
        len(groups),
        automatic_merges,
        automatic_separations,
        boundary_pairs,
        wave_count,
        largest_wave_tokens,
    )
    return groups
