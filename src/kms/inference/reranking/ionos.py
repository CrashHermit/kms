"""Direct IONOS multimodal Qwen reranking adapter."""

import asyncio
import json
import math
from typing import cast

import httpx

from kms.config.reranking import RerankingSettings
from kms.core.content import ContentImage
from kms.core.model.content import MultimodalContent
from kms.core.reranking import Reranker, RerankResult
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.inference.reranking.budget import IonosRerankingBudget
from kms.service.content import encode_content_images

IONOS_RERANK_URL = 'https://openai.inference.de-txl.ionos.com/v1/rerank'
IONOS_MAX_DOCUMENTS = 1_000
IONOS_MAX_BODY_BYTES = 16 * 1024 * 1024
IONOS_MAX_IMAGE_BYTES = 12 * 1024 * 1024
IONOS_MAX_CONTENT_PARTS = 2_048


class IonosReranker(Reranker):
    """Rerank all candidates in bounded, globally remapped HTTP batches."""

    def __init__(
        self,
        settings: RerankingSettings,
        budget: IonosRerankingBudget,
        client: httpx.AsyncClient,
    ) -> None:
        if settings.api_key is None or not settings.api_key.get_secret_value():
            raise ValueError('Reranking API key is required')
        self._settings = settings
        self._budget = budget
        self._client = client
        self._api_key = settings.api_key.get_secret_value()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def rerank(
        self,
        query: MultimodalContent,
        documents: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []
        if top_n is not None and top_n < 1:
            raise ValueError('top_n must be positive when provided')
        costs = self._budget.count_pairs(
            query, documents, content_images=content_images
        )
        for material in [query, *documents]:
            if (
                len(material.image_positions) + bool(material.text)
                > IONOS_MAX_CONTENT_PARTS
            ):
                raise InputBudgetExceeded(
                    'IONOS query or candidate exceeds 2,048 content parts'
                )
        used_positions = sorted(
            {
                position
                for material in [query, *documents]
                for position in material.image_positions
            }
        )
        for position in used_positions:
            if (
                self._budget.image_metadata.measure(
                    content_images[position]
                ).byte_count
                > IONOS_MAX_IMAGE_BYTES
            ):
                raise InputBudgetExceeded(
                    'IONOS inline image exceeds 12 MiB encoded bytes'
                )
        encoded = await encode_content_images(
            [content_images[position] for position in used_positions]
        )
        uri_by_position = dict(zip(used_positions, encoded, strict=True))

        def parts(material: MultimodalContent) -> list[dict[str, object]]:
            content: list[dict[str, object]] = []
            if material.text:
                content.append({'type': 'text', 'text': material.text})
            content.extend(
                {
                    'type': 'image_url',
                    'image_url': {'url': uri_by_position[position]},
                }
                for position in material.image_positions
            )
            if len(content) > IONOS_MAX_CONTENT_PARTS:
                raise InputBudgetExceeded(
                    'IONOS query or candidate exceeds 2,048 content parts'
                )
            return content

        query_parts = parts(query)
        batches: list[tuple[list[tuple[int, MultimodalContent]], bytes]] = []
        current: list[tuple[int, MultimodalContent]] = []
        current_cost = 0
        current_body: bytes | None = None
        for index, (document, cost) in enumerate(
            zip(documents, costs, strict=True)
        ):
            candidate = current + [(index, document)]
            candidate_cost = current_cost + cost
            exceeds_soft_limit = current and (
                candidate_cost > self._settings.batch_token_budget
                or len(candidate) > IONOS_MAX_DOCUMENTS
            )
            if exceeds_soft_limit:
                assert current_body is not None
                batches.append((current, current_body))
                current = [(index, document)]
                current_cost = cost
                body = self._body(query_parts, [parts(document)])
            else:
                candidate_body = self._body(
                    query_parts, [parts(item) for _, item in candidate]
                )
                if current and len(candidate_body) > IONOS_MAX_BODY_BYTES:
                    assert current_body is not None
                    batches.append((current, current_body))
                    current = [(index, document)]
                    current_cost = cost
                    body = self._body(query_parts, [parts(document)])
                else:
                    current = candidate
                    current_cost = candidate_cost
                    body = candidate_body
            if len(body) > IONOS_MAX_BODY_BYTES:
                raise InputBudgetExceeded(
                    'IONOS reranking request exceeds 16 MiB JSON body'
                )
            current_body = body
        if current:
            assert current_body is not None
            batches.append((current, current_body))
        next_batch = iter(enumerate(batches))
        result_slots: list[list[tuple[int, float]] | None] = [None] * len(
            batches
        )

        async def worker() -> None:
            for batch_index, (batch, body_bytes) in next_batch:
                async with self._semaphore:
                    response = await self._client.post(
                        IONOS_RERANK_URL,
                        content=body_bytes,
                        headers={
                            'Authorization': f'Bearer {self._api_key}',
                            'Content-Type': 'application/json',
                        },
                    )
                response.raise_for_status()
                data = response.json()['results']
                if len(data) != len(batch):
                    raise ValueError(
                        'IONOS returned an unexpected reranking count'
                    )
                seen_indexes: set[int] = set()
                mapped_results: list[tuple[int, float]] = []
                for item in data:
                    local_index = item['index']
                    score = item['relevance_score']
                    if (
                        type(local_index) is not int
                        or not 0 <= local_index < len(batch)
                        or local_index in seen_indexes
                    ):
                        raise ValueError(
                            'IONOS returned duplicate or invalid indexes'
                        )
                    if type(score) not in (int, float) or not math.isfinite(
                        score
                    ):
                        raise ValueError(
                            'IONOS returned an invalid relevance score'
                        )
                    seen_indexes.add(local_index)
                    mapped_results.append((batch[local_index][0], float(score)))
                result_slots[batch_index] = mapped_results

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self._settings.max_concurrency, len(batches)))
        ]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        batch_results = cast(list[list[tuple[int, float]]], result_slots)
        if len(batches) == 1:
            ordered = batch_results[0]
        else:
            # Scores from separate requests are not a calibrated global rank.
            ordered = sorted(
                (item for batch in batch_results for item in batch),
                key=lambda item: (-item[1], item[0]),
            )
        results: list[RerankResult] = [
            {'index': index, 'relevance_score': score}
            for index, score in ordered
        ]
        return results[:top_n] if top_n is not None else results

    def _body(
        self,
        query_parts: list[dict[str, object]],
        documents: list[list[dict[str, object]]],
    ) -> bytes:
        payload = {
            'model': self._settings.model,
            'query': query_parts,
            'documents': [{'content': item} for item in documents],
            'return_documents': False,
        }
        return json.dumps(
            payload, ensure_ascii=False, separators=(',', ':')
        ).encode('utf-8')
