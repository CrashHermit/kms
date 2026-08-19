import asyncio
from collections.abc import Awaitable, Callable

from kms.core import embeddings


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
            embeddings.cosine_similarity(
                candidate.embedding, other.embedding
            )
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
) -> list[list[dict]]:
    gate = asyncio.Semaphore(max_concurrency)
    groups: list[list[dict]] = []
    for coarse_group in coarse_groups(records, recall_threshold):
        unassigned = list(coarse_group)
        while unassigned:
            pivot = central_record(unassigned)
            unassigned = [
                record for record in unassigned if record is not pivot
            ]
            current_group = [pivot]
            remaining: list[dict] = []
            for record in unassigned:
                score = embeddings.cosine_similarity(
                    pivot.embedding, record.embedding
                )
                if score >= merge_above:
                    current_group.append(record)
                elif score <= separate_below:
                    remaining.append(record)
                else:
                    async with gate:
                        should_merge = await adjudicate(pivot, record)
                    if should_merge:
                        current_group.append(record)
                    else:
                        remaining.append(record)
            groups.append(current_group)
            unassigned = remaining
    return groups
