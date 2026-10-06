"""Dispatch and collect global triplet hub phases."""

import json
from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.embedding import EmbeddingClient
from kms2.core.model.global_semantic.global_triplet_hub import (
    GlobalTripletHub,
    GlobalTripletHubGroup,
    GlobalTripletHubSummary,
    GlobalTripletHubSummaryInput,
    GlobalTripletHubSummaryMergeInput,
    GlobalTripletHubSummarySynthesisInput,
    GlobalTripletHubSynthesisInput,
    GlobalTripletHubSynthesisResult,
)
from kms2.core.windowing import TokenBudget, fits_token_budget, pack_items
from kms2.database.global_semantic.global_triplet_repository import (
    GlobalTripletRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.module.global_semantic.global_triplet_hub import (
    GlobalTripletHubModule,
    GlobalTripletHubSummaryMergeModule,
    GlobalTripletHubSummaryModule,
)


class GlobalTripletHubSynthesisWorkerState(TypedDict):
    """State supplied to one global triplet synthesis worker."""

    global_triplet_hub_synthesis_ordinal: int
    global_triplet_hub_group: GlobalTripletHubGroup


class GlobalTripletHubNode:
    """Run exact global triplet grouping, synthesis, and embedding phases."""

    def __init__(
        self,
        repository: GlobalTripletRepository,
        module: GlobalTripletHubModule,
        embedding_client: EmbeddingClient,
        *,
        summary_module: GlobalTripletHubSummaryModule,
        merge_module: GlobalTripletHubSummaryMergeModule,
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
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Read deterministic exact global triplet groups."""
        groups = await self._repository.read_global_triplet_hub_groups()
        return {'global_triplet_hub_groups': groups}

    def dispatch_synthesis(
        self, state: GlobalSemanticState
    ) -> list[Send] | Literal['global_triplet_hub_synthesis_collect']:
        """Dispatch one synthesis worker per ordered global triplet group."""
        if not state.global_triplet_hub_groups:
            return 'global_triplet_hub_synthesis_collect'
        return [
            Send(
                'global_triplet_hub_synthesis_worker',
                {
                    'global_triplet_hub_synthesis_ordinal': ordinal,
                    'global_triplet_hub_group': group,
                },
            )
            for ordinal, group in enumerate(state.global_triplet_hub_groups)
        ]

    async def synthesis_worker(
        self, state: GlobalTripletHubSynthesisWorkerState
    ) -> dict[str, list[GlobalTripletHubSynthesisResult]]:
        """Synthesize one exact global triplet group with evidence reduction."""
        group = state['global_triplet_hub_group']
        original = _triplet_request(group)
        if fits_token_budget(
            token_count=self._final_budget.counter.count_texts(
                [original.model_dump_json()]
            )[0],
            threshold=self._final_budget.token_limit,
        ):
            definition = await self._module.acall(request=original)
        else:
            records = [
                json.dumps(
                    {
                        'role': f'{role_name}_hub',
                        'name': role.name,
                        'description': role.description,
                    },
                    ensure_ascii=False,
                )
                for role_name, role in (
                    ('subject', group.subject_hub),
                    ('predicate', group.predicate_hub),
                    ('object', group.object_hub),
                )
            ]
            records.extend(
                json.dumps({'source_triplet_hub': text}, ensure_ascii=False)
                for text in original.source_triplet_hubs
            )
            records.extend(
                json.dumps({'global_triplet': text}, ensure_ascii=False)
                for text in original.global_triplets
            )

            current: list[GlobalTripletHubSummary] = []
            for batch in pack_items(
                records,
                token_counts=self._summary_budget.counter.count_texts(records),
                token_budget=self._summary_budget.token_limit,
            ):
                current.append(
                    await self._summary_module.acall(
                        request=GlobalTripletHubSummaryInput(evidence=batch)
                    )
                )
            while True:
                final_request = GlobalTripletHubSummarySynthesisInput(
                    summaries=current
                )
                if fits_token_budget(
                    token_count=self._final_budget.counter.count_texts(
                        [final_request.model_dump_json()]
                    )[0],
                    threshold=self._final_budget.token_limit,
                ):
                    definition = await self._module.acall(request=final_request)
                    break
                current = await self._merge_summary_level(current)
        return {
            'global_triplet_hub_synthesis_results': [
                GlobalTripletHubSynthesisResult(
                    ordinal=state['global_triplet_hub_synthesis_ordinal'],
                    definition=definition,
                )
            ]
        }

    async def _merge_summary_level(
        self, summaries: list[GlobalTripletHubSummary]
    ) -> list[GlobalTripletHubSummary]:
        """Merge global triplet summaries in ordered fitting batches."""

        merged: list[GlobalTripletHubSummary] = []
        for batch in pack_items(
            summaries,
            token_counts=self._merge_budget.counter.count_texts(
                [summary.model_dump_json() for summary in summaries]
            ),
            token_budget=self._merge_budget.token_limit,
        ):
            if len(batch) == 1:
                merged.append(batch[0])
            else:
                merged.append(
                    await self._merge_module.acall(
                        request=GlobalTripletHubSummaryMergeInput(
                            summaries=batch
                        )
                    )
                )
        return merged

    def collect_synthesis(
        self, state: GlobalSemanticState
    ) -> dict[str, object]:
        """Collect global triplet definitions in group order."""
        return {
            'global_triplet_hub_synthesis_results_ordered': sorted(
                state.global_triplet_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: GlobalSemanticState) -> dict[str, object]:
        """Embed ordered definitions and construct typed global hubs."""
        results = state.global_triplet_hub_synthesis_results_ordered
        groups = state.global_triplet_hub_groups
        definitions = [result.definition for result in results]
        if not definitions:
            return {
                'global_triplet_hubs': [],
                'global_triplet_hub_memberships': [],
            }
        vectors = await self._embedding_client.embed(
            [
                f'{definition.canonical_name}: {definition.description}'
                for definition in definitions
            ]
        )
        hubs = [
            GlobalTripletHub(
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
            'global_triplet_hubs': hubs,
            'global_triplet_hub_memberships': [
                group.global_triplet_uuids for group in groups
            ],
        }


def _triplet_request(
    group: GlobalTripletHubGroup,
) -> GlobalTripletHubSynthesisInput:
    """Build the original UUID-free global triplet synthesis request."""
    return GlobalTripletHubSynthesisInput(
        subject_hub=group.subject_hub,
        predicate_hub=group.predicate_hub,
        object_hub=group.object_hub,
        source_triplet_hubs=sorted(
            {
                f'{evidence.canonical_name}: {evidence.description}'
                for evidence in group.evidence
            }
        ),
        global_triplets=sorted(
            {
                f'{group.subject_hub.name} | '
                f'{group.predicate_hub.name} | {group.object_hub.name}'
                for _ in group.evidence
            }
        ),
    )
