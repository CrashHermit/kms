"""Dispatch and collect global triplet hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.embedding import EmbeddingClient
from kms2.core.model.global_semantic.global_triplet_hub import (
    GlobalTripletHub,
    GlobalTripletHubGroup,
    GlobalTripletHubSynthesisInput,
    GlobalTripletHubSynthesisResult,
)
from kms2.database.global_semantic.global_triplet_repository import (
    GlobalTripletRepository,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.module.global_semantic.global_triplet_hub import (
    GlobalTripletHubModule,
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
    ) -> None:
        self._repository = repository
        self._module = module
        self._embedding_client = embedding_client

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
        """Synthesize one exact global triplet group."""
        group = state['global_triplet_hub_group']
        definition = await self._module.aforward(
            request=GlobalTripletHubSynthesisInput(
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
                        f'{group.predicate_hub.name} | '
                        f'{group.object_hub.name}'
                        for _ in group.evidence
                    }
                ),
            )
        )
        return {
            'global_triplet_hub_synthesis_results': [
                GlobalTripletHubSynthesisResult(
                    ordinal=state['global_triplet_hub_synthesis_ordinal'],
                    definition=definition,
                )
            ]
        }

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
