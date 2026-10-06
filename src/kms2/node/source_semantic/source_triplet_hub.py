"""Dispatch and collect source-local triplet hub phases."""

import json
from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.embedding import EmbeddingClient
from kms2.core.model.source_semantic.source_triplet_hub import (
    SourceTripletHub,
    SourceTripletHubGroup,
    SourceTripletHubSummary,
    SourceTripletHubSummaryInput,
    SourceTripletHubSummaryMergeInput,
    SourceTripletHubSummarySynthesisInput,
    SourceTripletHubSynthesisInput,
    SourceTripletHubSynthesisResult,
)
from kms2.core.windowing import TokenBudget, fits_token_budget, pack_items
from kms2.database.source_semantic.source_triplet_repository import (
    SourceTripletRepository,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.module.source_semantic.source_triplet_hub import (
    SourceTripletHubModule,
    SourceTripletHubSummaryMergeModule,
    SourceTripletHubSummaryModule,
)


class SourceTripletHubSynthesisWorkerState(TypedDict):
    """State supplied to one triplet synthesis worker."""

    source_triplet_hub_synthesis_ordinal: int
    source_triplet_hub_group: SourceTripletHubGroup


class SourceTripletHubNode:
    """Run exact triplet grouping, synthesis, and embedding phases."""

    def __init__(
        self,
        repository: SourceTripletRepository,
        module: SourceTripletHubModule,
        embedding_client: EmbeddingClient,
        *,
        summary_module: SourceTripletHubSummaryModule,
        merge_module: SourceTripletHubSummaryMergeModule,
        final_budget: TokenBudget,
        summary_budget: TokenBudget,
        merge_budget: TokenBudget,
    ) -> None:
        self._repository = repository
        self._module = module
        self._embedding_client = embedding_client
        self._summary_module = summary_module
        self._merge_module = merge_module
        self._final_budget = final_budget
        self._summary_budget = summary_budget
        self._merge_budget = merge_budget

    async def load_groups(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Read deterministic exact triplet groups after typed hub persistence."""
        groups = await self._repository.read_source_triplet_hub_groups(
            state.source_uuid
        )
        return {'source_triplet_hub_groups': groups}

    def dispatch_synthesis(
        self, state: SourceSemanticState
    ) -> list[Send] | Literal['source_triplet_hub_synthesis_collect']:
        """Dispatch one synthesis worker per ordered triplet group."""
        if not state.source_triplet_hub_groups:
            return 'source_triplet_hub_synthesis_collect'
        sends = [
            Send(
                'source_triplet_hub_synthesis_worker',
                {
                    'source_triplet_hub_synthesis_ordinal': ordinal,
                    'source_triplet_hub_group': group,
                },
            )
            for ordinal, group in enumerate(state.source_triplet_hub_groups)
        ]
        return sends

    async def synthesis_worker(
        self, state: SourceTripletHubSynthesisWorkerState
    ) -> dict[str, list[SourceTripletHubSynthesisResult]]:
        """Synthesize one exact triplet group."""
        group = state['source_triplet_hub_group']
        request = SourceTripletHubSynthesisInput(
            subject_hub=group.subject_hub,
            predicate_hub=group.predicate_hub,
            object_hub=group.object_hub,
            source_facts=sorted(
                {evidence.fact_text for evidence in group.evidence}
            ),
            triplets=sorted(
                {
                    f'{evidence.subject} | {evidence.predicate} | '
                    f'{evidence.object}'
                    for evidence in group.evidence
                }
            ),
        )
        if fits_token_budget(
            token_count=self._final_budget.counter.count_texts(
                [request.model_dump_json()]
            )[0],
            threshold=self._final_budget.token_limit,
        ):
            definition = await self._module.acall(request=request)
        else:
            evidence = [
                json.dumps(
                    {
                        'role': role,
                        'name': hub.name,
                        'description': hub.description,
                    },
                    ensure_ascii=False,
                )
                for role, hub in (
                    ('subject_hub', request.subject_hub),
                    ('predicate_hub', request.predicate_hub),
                    ('object_hub', request.object_hub),
                )
            ]
            evidence.extend(
                json.dumps({'source_fact': text}, ensure_ascii=False)
                for text in request.source_facts
            )
            evidence.extend(
                json.dumps({'triplet': text}, ensure_ascii=False)
                for text in request.triplets
            )
            summaries = await self._summarize_evidence(evidence)
            while not fits_token_budget(
                token_count=self._final_budget.counter.count_texts(
                    [
                        SourceTripletHubSummarySynthesisInput(
                            summaries=summaries
                        ).model_dump_json()
                    ]
                )[0],
                threshold=self._final_budget.token_limit,
            ):
                summaries = await self._merge_summary_level(summaries)
            definition = await self._module.acall(
                request=SourceTripletHubSummarySynthesisInput(
                    summaries=summaries
                )
            )
        return {
            'source_triplet_hub_synthesis_results': [
                SourceTripletHubSynthesisResult(
                    ordinal=state['source_triplet_hub_synthesis_ordinal'],
                    definition=definition,
                )
            ]
        }

    async def _summarize_evidence(
        self, evidence: list[str]
    ) -> list[SourceTripletHubSummary]:
        """Summarize ordered whole triplet evidence in fitting batches."""

        batches = pack_items(
            evidence,
            token_counts=self._summary_budget.counter.count_texts(evidence),
            token_budget=self._summary_budget.token_limit,
        )
        return [
            await self._summary_module.acall(
                request=SourceTripletHubSummaryInput(evidence=batch)
            )
            for batch in batches
        ]

    async def _merge_summary_level(
        self, summaries: list[SourceTripletHubSummary]
    ) -> list[SourceTripletHubSummary]:
        """Merge one input-fitting level, carrying any trailing singleton."""

        batches = pack_items(
            summaries,
            token_counts=self._merge_budget.counter.count_texts(
                [summary.model_dump_json() for summary in summaries]
            ),
            token_budget=self._merge_budget.token_limit,
        )
        merged: list[SourceTripletHubSummary] = []
        for batch in batches:
            if len(batch) == 1:
                merged.extend(batch)
            else:
                merged.append(
                    await self._merge_module.acall(
                        request=SourceTripletHubSummaryMergeInput(
                            summaries=batch
                        )
                    )
                )
        return merged

    def collect_synthesis(
        self, state: SourceSemanticState
    ) -> dict[str, object]:
        """Collect triplet definitions in group order."""
        return {
            'source_triplet_hub_synthesis_results_ordered': sorted(
                state.source_triplet_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: SourceSemanticState) -> dict[str, object]:
        """Embed ordered triplet definitions and construct typed hubs."""
        results = state.source_triplet_hub_synthesis_results_ordered
        groups = state.source_triplet_hub_groups
        definitions = [result.definition for result in results]
        if not definitions:
            return {
                'source_triplet_hubs': [],
                'source_triplet_hub_memberships': [],
            }
        vectors = await self._embedding_client.embed(
            [
                f'{definition.canonical_name}: {definition.description}'
                for definition in definitions
            ]
        )
        hubs = [
            SourceTripletHub(
                source_uuid=state.source_uuid,
                canonical_name=definition.canonical_name,
                description=definition.description,
                embedding=vector,
                subject_hub_uuid=group.subject_hub_uuid,
                predicate_hub_uuid=group.predicate_hub_uuid,
                object_hub_uuid=group.object_hub_uuid,
            )
            for group, result, definition, vector in zip(
                groups, results, definitions, vectors, strict=True
            )
        ]
        return {
            'source_triplet_hubs': hubs,
            'source_triplet_hub_memberships': [
                group.triplet_uuids for group in groups
            ],
        }
