"""Dispatch and collect source-local triplet hub phases."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.embedding import EmbeddingClient
from kms2.core.model import (
    SourceTripletHub,
    SourceTripletHubGroup,
    SourceTripletHubSynthesisInput,
    SourceTripletHubSynthesisResult,
)
from kms2.database.semantic.source_triplet_repository import (
    SourceTripletRepository,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.module.semantic.source_triplet_hub import SourceTripletHubModule


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
    ) -> None:
        self._repository = repository
        self._module = module
        self._embedding_client = embedding_client

    async def load_groups(self, state: SemanticState) -> dict[str, object]:
        """Read deterministic exact triplet groups after typed hub persistence."""
        groups = await self._repository.read_source_triplet_hub_groups(
            state.source_uuid
        )
        return {'source_triplet_hub_groups': groups}

    def dispatch_synthesis(
        self, state: SemanticState
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
        definition = await self._module.aforward(
            request=SourceTripletHubSynthesisInput(
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
        )
        return {
            'source_triplet_hub_synthesis_results': [
                SourceTripletHubSynthesisResult(
                    ordinal=state['source_triplet_hub_synthesis_ordinal'],
                    definition=definition,
                )
            ]
        }

    def collect_synthesis(self, state: SemanticState) -> dict[str, object]:
        """Collect triplet definitions in group order."""
        return {
            'source_triplet_hub_synthesis_results_ordered': sorted(
                state.source_triplet_hub_synthesis_results,
                key=lambda item: item.ordinal,
            )
        }

    async def embed(self, state: SemanticState) -> dict[str, object]:
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


__all__ = ['SourceTripletHubNode']
