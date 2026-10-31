"""Persistence access for source propositions and proposition hubs."""

from collections.abc import Callable

from neo4j import AsyncSession

from kms2.core.model.block import SourceBlock
from kms2.core.model.similarity import SourcePropositionHubSearchMatch
from kms2.core.model.source_extraction import SourceExtractionTarget
from kms2.core.model.source_semantic.source_proposition_extraction import (
    SourceProposition,
    SourcePropositionEmbeddingResult,
)
from kms2.core.model.source_semantic.source_proposition_hub import (
    SourcePropositionHub,
    SourcePropositionHubCandidate,
    SourcePropositionHubMember,
)
from kms2.database.source_semantic.queries.source_proposition import (
    READ_SOURCE_PROPOSITION_HUB_CANDIDATES,
    READ_SOURCE_PROPOSITION_HUB_MEMBERS,
    READ_SOURCE_PROPOSITIONS,
    REPLACE_SOURCE_PROPOSITION_HUBS,
    REPLACE_SOURCE_PROPOSITIONS,
    SEARCH_SOURCE_PROPOSITION_HUBS,
    UPDATE_SOURCE_PROPOSITION_EMBEDDINGS,
)


class SourcePropositionRepository:
    """Access source-owned proposition occurrences, evidence, and hubs."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def replace_source_propositions(
        self, source_uuid: str, propositions: list[SourceProposition]
    ) -> None:
        """Replace all proposition occurrences and hubs owned by a source."""
        rows = [
            {
                'uuid': proposition.uuid,
                'text': proposition.text,
                'target_block_uuid': proposition.target.source_blocks[0].uuid,
            }
            for proposition in propositions
        ]
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_PROPOSITIONS,
                source_uuid=source_uuid,
                propositions=rows,
            )
            await result.single(strict=True)

    async def load_source_propositions(
        self, source_uuid: str
    ) -> list[SourceProposition]:
        """Load source propositions with ordered block evidence."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_PROPOSITIONS,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [
            SourceProposition(
                uuid=row['uuid'],
                source_uuid=row['source_uuid'],
                text=row['text'],
                target=SourceExtractionTarget(
                    source_blocks=[
                        _source_block(block) for block in row['target_blocks']
                    ],
                ),
            )
            for row in rows
        ]

    async def update_source_proposition_embeddings(
        self,
        source_uuid: str,
        results: list[SourcePropositionEmbeddingResult],
    ) -> None:
        """Persist complete-text embeddings on owned proposition occurrences."""
        if not results:
            return
        rows = [
            {'uuid': result.uuid, 'embedding': result.embedding}
            for result in results
        ]
        async with self._session_factory() as session:
            response = await session.run(
                UPDATE_SOURCE_PROPOSITION_EMBEDDINGS,
                source_uuid=source_uuid,
                rows=rows,
            )
            await response.consume()

    async def read_source_proposition_hub_candidates(
        self,
        source_uuid: str,
        *,
        candidate_limit: int,
        minimum_similarity: float,
    ) -> list[SourcePropositionHubCandidate]:
        """Read bounded same-source vector candidates in canonical pair order."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_PROPOSITION_HUB_CANDIDATES,
                source_uuid=source_uuid,
                candidate_limit=candidate_limit,
                minimum_similarity=minimum_similarity,
            )
            rows = await result.data()
        return [
            SourcePropositionHubCandidate.model_validate(row) for row in rows
        ]

    async def read_source_proposition_hub_members(
        self, source_uuid: str
    ) -> list[SourcePropositionHubMember]:
        """Read every occurrence as an ordered grouping member."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_SOURCE_PROPOSITION_HUB_MEMBERS,
                source_uuid=source_uuid,
            )
            rows = await result.data()
        return [SourcePropositionHubMember.model_validate(row) for row in rows]

    async def replace_source_proposition_hubs(
        self,
        source_uuid: str,
        hubs: list[SourcePropositionHub],
        memberships: list[list[str]],
    ) -> None:
        """Replace identity-only source hubs and occurrence memberships."""
        rows = [
            {**hub.model_dump(), 'member_uuids': member_uuids}
            for hub, member_uuids in zip(hubs, memberships, strict=True)
        ]
        async with self._session_factory() as session:
            result = await session.run(
                REPLACE_SOURCE_PROPOSITION_HUBS,
                source_uuid=source_uuid,
                hubs=rows,
            )
            await result.single(strict=True)

    async def search_source_proposition_hubs(
        self,
        source_uuid: str,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[SourcePropositionHubSearchMatch]:
        """Search source hubs by best member cosine score and return that evidence."""
        _validate_search(query_embedding, top_k)
        async with self._session_factory() as session:
            result = await session.run(
                SEARCH_SOURCE_PROPOSITION_HUBS,
                source_uuid=source_uuid,
                query_embedding=query_embedding,
                top_k=top_k,
            )
            rows = await result.data()
        return [
            SourcePropositionHubSearchMatch.model_validate(row) for row in rows
        ]


def _source_block(row: dict[str, object]) -> SourceBlock:
    """Build canonical source-block evidence from a query projection."""
    return SourceBlock(
        uuid=row['uuid'],
        block_type=row['block_type'],
        content=row['content'],
    )


def _validate_search(embedding: list[float], top_k: int) -> None:
    if not embedding:
        raise ValueError('query_embedding must not be empty')
    if top_k <= 0:
        raise ValueError('top_k must be positive')
