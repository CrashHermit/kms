"""Serialize candidate auditing, canonicalization, embedding, and persistence."""

from kms.core.embedding import Embedder, EmbeddingRequest
from kms.knowledge.candidates import (
    ExistingPropositionReference,
    NewPropositionReference,
)
from kms.knowledge.extraction.models import UnitExtractionResult
from kms.knowledge.identity.audit import IdentityAuditor
from kms.knowledge.models import (
    Argument,
    KnowledgeContext,
    StoredKnowledge,
)
from kms.knowledge.reasoning.models import ReasoningCandidates
from kms.knowledge.state import KnowledgeState
from kms.knowledge.storage import KnowledgeRepository
from kms.media.models import Material


class KnowledgeCommitter:
    def __init__(
        self,
        repository: KnowledgeRepository,
        embedder: Embedder,
        embedding_model: str,
        *,
        auditor: IdentityAuditor,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._embedding_model = embedding_model
        self._auditor = auditor

    async def commit(
        self,
        user_uuid: str,
        result: UnitExtractionResult | ReasoningCandidates,
    ) -> StoredKnowledge:
        if isinstance(result, UnitExtractionResult):
            new_propositions = result.candidates.new_propositions
            new_arguments = result.candidates.new_arguments
            source_uuid = result.source_uuid
            target_unit_uuid = result.target_unit_uuid
            existing_proposition_uuids = (
                result.candidates.existing_proposition_uuids
            )
            existing_argument_uuids = result.candidates.existing_argument_uuids
        else:
            new_propositions = result.new_propositions
            new_arguments = result.new_arguments
            source_uuid = None
            target_unit_uuid = None
            existing_proposition_uuids = []
            existing_argument_uuids = []

        if not (
            new_propositions
            or new_arguments
            or existing_proposition_uuids
            or existing_argument_uuids
        ):
            return StoredKnowledge(propositions=[], arguments=[])

        async with self._repository.serialized(user_uuid) as transaction:
            audited = await self._auditor.audit(user_uuid, new_propositions)
            by_id = audited.canonical

            def resolve(
                reference: ExistingPropositionReference
                | NewPropositionReference,
            ) -> str:
                if isinstance(reference, ExistingPropositionReference):
                    return reference.uuid
                return by_id[reference.id].uuid

            arguments = [
                Argument(
                    premise_uuids=[resolve(item) for item in argument.premises],
                    conclusion_uuid=resolve(argument.conclusion),
                )
                for argument in new_arguments
            ]
            propositions = audited.new_propositions
            embeddings = (
                await self._embedder.embed(
                    EmbeddingRequest(
                        materials=[
                            Material.from_text(item.statement)
                            for item in propositions
                        ],
                        input_type='document',
                    )
                )
                if propositions
                else []
            )
            new_uuids = {item.uuid for item in propositions}
            reused_uuids = [
                item.uuid
                for item in by_id.values()
                if item.uuid not in new_uuids
            ]
            arguments = await self._repository.save(
                user_uuid,
                propositions,
                embeddings,
                arguments,
                source_uuid=source_uuid,
                target_unit_uuid=target_unit_uuid,
                existing_proposition_uuids=[
                    *existing_proposition_uuids,
                    *reused_uuids,
                ],
                existing_argument_uuids=existing_argument_uuids,
                embedding_model=self._embedding_model,
                transaction=transaction,
            )
        canonical = {item.uuid: item for item in by_id.values()}
        return StoredKnowledge(
            propositions=list(canonical.values()), arguments=arguments
        )


class KnowledgeCommitNode:
    def __init__(self, committer: KnowledgeCommitter) -> None:
        self._committer = committer

    async def run(self, state: KnowledgeState) -> dict:
        pending = state['pending']
        if pending is None:
            raise PermissionError('Source unit is no longer accessible')
        stored = await self._committer.commit(state['user_uuid'], pending)
        return {
            'context': _include(state['context'], stored),
            'stored': stored,
            'pending': None,
        }


def _include(
    context: KnowledgeContext, stored: StoredKnowledge
) -> KnowledgeContext:
    matched = {item.uuid: item for item in context.matched}
    matched.update({item.uuid: item for item in stored.propositions})
    propositions = {item.uuid: item for item in context.propositions}
    propositions.update({item.uuid: item for item in stored.propositions})
    arguments = {item.uuid: item for item in context.arguments}
    arguments.update({item.uuid: item for item in stored.arguments})
    return KnowledgeContext(
        matched=list(matched.values()),
        propositions=list(propositions.values()),
        arguments=list(arguments.values()),
    )
