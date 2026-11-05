"""Serialize reference-only session workspaces and restore from owned graphs."""

from collections import defaultdict

from kms.core.model.source import Source
from kms.core.model.source_evidence import (
    SourceImage,
    SourceProcedurePassage,
    SourceStatementPassage,
)
from kms.core.model.source_workspace import (
    SourceWorkspaceBlockEntry,
    SourceWorkspaceImageEntry,
)
from kms.core.model.teaching.workspace import (
    BlockInspection,
    ImageInspection,
    SavedBlockInspection,
    SavedImageInspection,
    SavedSourceBlockEntry,
    SavedSourceImageEntry,
    SavedTeachingEvidenceWorkspace,
    SavedTeachingLearnerObservationRegistryEntry,
    SavedTeachingProcedureRegistryEntry,
    SavedTeachingStatementRegistryEntry,
    TeachingEvidenceWorkspace,
    TeachingLearnerObservationRegistryEntry,
    TeachingProcedureRegistryEntry,
    TeachingStatementRegistryEntry,
)
from kms.core.model.visual_asset import VisualAsset
from kms.service.learner_observations import LearnerObservationService
from kms.service.source_artifacts import read_asset_bytes
from kms.service.source_evidence import SourceEvidenceService


def serialize_workspace(
    workspace: TeachingEvidenceWorkspace,
) -> SavedTeachingEvidenceWorkspace:
    """Project runtime registries into reference-only saved values."""
    saved = SavedTeachingEvidenceWorkspace(
        block_registry=[
            SavedSourceBlockEntry(
                source=entry.source,
                page_indexes=entry.page_indexes,
                block_uuid=entry.block.uuid,
            )
            for entry in workspace.block_registry
        ],
        image_registry=[
            SavedSourceImageEntry(
                source=entry.source,
                asset_uuid=entry.asset.uuid,
                block_positions=entry.block_positions,
            )
            for entry in workspace.image_registry
        ],
        group_registry=workspace.group_registry,
        proposition_registry=workspace.proposition_registry,
        learner_observation_registry=[
            SavedTeachingLearnerObservationRegistryEntry(
                uuid=entry.observation.uuid,
                was_read=entry.record is not None,
            )
            for entry in workspace.learner_observation_registry
        ],
        argument_registry=workspace.argument_registry,
        statement_registry=[
            SavedTeachingStatementRegistryEntry(
                source=entry.source,
                page_indexes=entry.page_indexes,
                statement=entry.statement,
                was_read=entry.passage is not None,
            )
            for entry in workspace.statement_registry
        ],
        procedure_registry=[
            SavedTeachingProcedureRegistryEntry(
                source=entry.source,
                page_indexes=entry.page_indexes,
                procedure=entry.procedure,
                was_read=entry.passage is not None,
            )
            for entry in workspace.procedure_registry
        ],
        read_block_positions=workspace.read_block_positions,
        read_image_positions=workspace.read_image_positions,
        inspected_block_positions=workspace.inspected_block_positions,
        inspected_image_positions=workspace.inspected_image_positions,
        inspected_proposition_positions=(
            workspace.inspected_proposition_positions
        ),
        inspected_argument_positions=workspace.inspected_argument_positions,
        inspected_statement_positions=workspace.inspected_statement_positions,
        inspected_procedure_positions=workspace.inspected_procedure_positions,
        local_outcomes=workspace.local_outcomes,
        receipts=[
            SavedBlockInspection(
                source=receipt.source,
                block_uuid=receipt.block.uuid,
                page_indexes=receipt.page_indexes,
            )
            if isinstance(receipt, BlockInspection)
            else SavedImageInspection(
                source=receipt.source,
                asset_uuid=receipt.asset.uuid,
                block_uuids=receipt.block_uuids,
            )
            if isinstance(receipt, ImageInspection)
            else receipt
            for receipt in workspace.receipts
        ],
        observations=workspace.observations,
        active_block_positions=workspace.active_block_positions,
        active_image_positions=workspace.active_image_positions,
        active_proposition_positions=workspace.active_proposition_positions,
        active_argument_positions=workspace.active_argument_positions,
        active_statement_positions=workspace.active_statement_positions,
        active_procedure_positions=workspace.active_procedure_positions,
        discovered_block_positions=workspace.discovered_block_positions,
        discovered_image_positions=workspace.discovered_image_positions,
        discovered_proposition_positions=(
            workspace.discovered_proposition_positions
        ),
        discovered_argument_positions=workspace.discovered_argument_positions,
        discovered_statement_positions=workspace.discovered_statement_positions,
        discovered_procedure_positions=workspace.discovered_procedure_positions,
        active_learner_observation_positions=(
            workspace.active_learner_observation_positions
        ),
        discovered_learner_observation_positions=(
            workspace.discovered_learner_observation_positions
        ),
    )
    return saved


async def restore_workspace(
    saved: SavedTeachingEvidenceWorkspace | None,
    *,
    user_uuid: str,
    source_evidence: SourceEvidenceService,
    learner_observations: LearnerObservationService,
) -> TeachingEvidenceWorkspace:
    """Restore current evidence and prune references to replaced revisions."""
    if saved is None:
        return TeachingEvidenceWorkspace()
    learner_records = (
        await learner_observations.read(
            user_uuid,
            [entry.uuid for entry in saved.learner_observation_registry],
        )
        if saved.learner_observation_registry
        else []
    )
    learner_observation_registry = [
        TeachingLearnerObservationRegistryEntry(
            observation=record.observation,
            record=record if saved_entry.was_read else None,
        )
        for saved_entry, record in zip(
            saved.learner_observation_registry,
            learner_records,
            strict=True,
        )
    ]

    references: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(dict)

    def add_block(
        source: Source, block_uuid: str, page_indexes: list[int]
    ) -> None:
        references[(source.uuid, source.revision)].setdefault(
            block_uuid, page_indexes
        )

    for entry in saved.block_registry:
        add_block(entry.source, entry.block_uuid, entry.page_indexes)
    for receipt in saved.receipts:
        if isinstance(receipt, SavedBlockInspection):
            add_block(receipt.source, receipt.block_uuid, receipt.page_indexes)
    for entry in saved.group_registry:
        for block_uuid in entry.group.member_block_uuids:
            add_block(entry.source, block_uuid, [])
    for entry in saved.statement_registry:
        for block_uuid in entry.statement.member_block_uuids:
            add_block(entry.source, block_uuid, entry.page_indexes)
    for entry in saved.procedure_registry:
        for block_uuid in entry.procedure.member_block_uuids:
            add_block(entry.source, block_uuid, entry.page_indexes)

    blocks: dict[tuple[str, int, str], SourceWorkspaceBlockEntry] = {}
    stale_sources: set[tuple[str, int]] = set()
    for (source_uuid, revision), requested in references.items():
        passages = await source_evidence.read_passages(
            user_uuid,
            list(requested),
            expand_memberships=False,
            before=0,
            after=0,
        )
        found: dict[str, SourceWorkspaceBlockEntry] = {}
        for passage in passages:
            if (
                passage.source.uuid != source_uuid
                or passage.source.revision != revision
            ):
                stale_sources.add((source_uuid, revision))
                break
            for block in passage.blocks:
                if block.block.uuid in requested:
                    found[block.block.uuid] = SourceWorkspaceBlockEntry(
                        source=passage.source,
                        page_indexes=requested[block.block.uuid],
                        block=block.block,
                    )
        if found.keys() != requested.keys():
            stale_sources.add((source_uuid, revision))
        if (source_uuid, revision) not in stale_sources:
            for block_uuid, entry in found.items():
                blocks[(source_uuid, revision, block_uuid)] = entry
    statement_passages: dict[str, SourceStatementPassage] = {}
    read_statements = [
        entry for entry in saved.statement_registry if entry.was_read
    ]
    if read_statements:
        passages = await source_evidence.read_statements(
            user_uuid,
            list(
                dict.fromkeys(entry.statement.uuid for entry in read_statements)
            ),
        )
        statement_passages = {
            passage.statement.uuid: passage for passage in passages
        }
        for entry in read_statements:
            passage = statement_passages.get(entry.statement.uuid)
            if (
                passage is None
                or not passage.passages
                or any(
                    source_passage.source.uuid != entry.source.uuid
                    or source_passage.source.revision != entry.source.revision
                    for source_passage in passage.passages
                )
            ):
                stale_sources.add((entry.source.uuid, entry.source.revision))

    procedure_passages: dict[str, SourceProcedurePassage] = {}
    read_procedures = [
        entry for entry in saved.procedure_registry if entry.was_read
    ]
    if read_procedures:
        passages = await source_evidence.read_procedures(
            user_uuid,
            list(
                dict.fromkeys(entry.procedure.uuid for entry in read_procedures)
            ),
        )
        procedure_passages = {
            passage.procedure.uuid: passage for passage in passages
        }
        for entry in read_procedures:
            passage = procedure_passages.get(entry.procedure.uuid)
            if (
                passage is None
                or not passage.passages
                or any(
                    source_passage.source.uuid != entry.source.uuid
                    or source_passage.source.revision != entry.source.revision
                    for source_passage in passage.passages
                )
            ):
                stale_sources.add((entry.source.uuid, entry.source.revision))
    valid_block_entries = [
        (position, entry)
        for position, entry in enumerate(saved.block_registry)
        if (entry.source.uuid, entry.source.revision) not in stale_sources
    ]
    block_position_remap = {
        old: new for new, (old, _) in enumerate(valid_block_entries)
    }
    block_registry = [
        blocks[(entry.source.uuid, entry.source.revision, entry.block_uuid)]
        for _, entry in valid_block_entries
    ]
    valid_image_entries = [
        (position, entry)
        for position, entry in enumerate(saved.image_registry)
        if (entry.source.uuid, entry.source.revision) not in stale_sources
    ]
    image_position_remap = {
        old: new for new, (old, _) in enumerate(valid_image_entries)
    }

    def remap_positions(
        positions: list[int] | set[int], remap: dict[int, int]
    ) -> list[int]:
        return [remap[position] for position in positions if position in remap]

    loaded_assets: dict[tuple[str, int, str], VisualAsset] = {}
    for (source_uuid, revision, _), entry in blocks.items():
        for asset in entry.block.assets:
            loaded_assets[(source_uuid, revision, asset.uuid)] = asset

    image_refs = [
        (entry.source.uuid, entry.source.revision, entry.asset_uuid)
        for _, entry in valid_image_entries
    ]
    for receipt in saved.receipts:
        if (
            isinstance(receipt, SavedImageInspection)
            and (receipt.source.uuid, receipt.source.revision)
            not in stale_sources
        ):
            reference = (
                receipt.source.uuid,
                receipt.source.revision,
                receipt.asset_uuid,
            )
            if reference not in image_refs:
                image_refs.append(reference)

    missing_refs = [
        reference for reference in image_refs if reference not in loaded_assets
    ]
    loaded_images: dict[tuple[str, int, str], SourceImage] = {}
    if missing_refs:
        images = await source_evidence.read_images_at_revisions(
            user_uuid, missing_refs
        )
        loaded_images = dict(zip(missing_refs, images, strict=True))
        loaded_assets.update(
            {
                reference: image.asset
                for reference, image in loaded_images.items()
            }
        )

    image_registry: list[SourceWorkspaceImageEntry] = []
    image_data: dict[str, bytes] = {}

    def load_data(asset: VisualAsset, reference: tuple[str, int, str]) -> bytes:
        if asset.content_sha256 not in image_data:
            image = loaded_images.get(reference)
            image_data[asset.content_sha256] = (
                image.data if image is not None else read_asset_bytes(asset)
            )
        return image_data[asset.content_sha256]

    data_positions = (
        saved.read_image_positions | saved.inspected_image_positions
    )
    for position, entry in valid_image_entries:
        reference = (entry.source.uuid, entry.source.revision, entry.asset_uuid)
        asset = loaded_assets[reference]
        image_registry.append(
            SourceWorkspaceImageEntry(
                asset=asset,
                source=entry.source,
                block_positions=remap_positions(
                    entry.block_positions, block_position_remap
                ),
                data=(
                    load_data(asset, reference)
                    if position in data_positions
                    else None
                ),
            )
        )

    receipts = []
    for receipt in saved.receipts:
        if (
            isinstance(receipt, (SavedBlockInspection, SavedImageInspection))
            and (receipt.source.uuid, receipt.source.revision) in stale_sources
        ):
            continue
        if isinstance(receipt, SavedBlockInspection):
            entry = blocks[
                (
                    receipt.source.uuid,
                    receipt.source.revision,
                    receipt.block_uuid,
                )
            ]
            receipts.append(
                BlockInspection(
                    source=entry.source,
                    block=entry.block,
                    page_indexes=receipt.page_indexes,
                )
            )
        elif isinstance(receipt, SavedImageInspection):
            reference = (
                receipt.source.uuid,
                receipt.source.revision,
                receipt.asset_uuid,
            )
            asset = loaded_assets[reference]
            receipts.append(
                ImageInspection(
                    source=receipt.source,
                    asset=asset,
                    block_uuids=receipt.block_uuids,
                    data=load_data(asset, reference),
                )
            )
        else:
            receipts.append(receipt)

    valid_statement_entries = [
        (position, entry)
        for position, entry in enumerate(saved.statement_registry)
        if (entry.source.uuid, entry.source.revision) not in stale_sources
    ]
    statement_position_remap = {
        old: new for new, (old, _) in enumerate(valid_statement_entries)
    }
    valid_procedure_entries = [
        (position, entry)
        for position, entry in enumerate(saved.procedure_registry)
        if (entry.source.uuid, entry.source.revision) not in stale_sources
    ]
    procedure_position_remap = {
        old: new for new, (old, _) in enumerate(valid_procedure_entries)
    }

    valid_group_entries = [
        (position, entry)
        for position, entry in enumerate(saved.group_registry)
        if (entry.source.uuid, entry.source.revision) not in stale_sources
    ]
    group_position_remap = {
        old: new for new, (old, _) in enumerate(valid_group_entries)
    }

    return TeachingEvidenceWorkspace(
        block_registry=block_registry,
        image_registry=image_registry,
        group_registry=[entry for _, entry in valid_group_entries],
        proposition_registry=saved.proposition_registry,
        argument_registry=saved.argument_registry,
        learner_observation_registry=learner_observation_registry,
        statement_registry=[
            TeachingStatementRegistryEntry(
                source=entry.source,
                page_indexes=entry.page_indexes,
                statement=entry.statement,
                passage=statement_passages.get(entry.statement.uuid),
            )
            for _, entry in valid_statement_entries
        ],
        procedure_registry=[
            TeachingProcedureRegistryEntry(
                source=entry.source,
                page_indexes=entry.page_indexes,
                procedure=entry.procedure,
                passage=procedure_passages.get(entry.procedure.uuid),
            )
            for _, entry in valid_procedure_entries
        ],
        read_block_positions=set(
            remap_positions(saved.read_block_positions, block_position_remap)
        ),
        read_image_positions=set(
            remap_positions(saved.read_image_positions, image_position_remap)
        ),
        inspected_block_positions=set(
            remap_positions(
                saved.inspected_block_positions, block_position_remap
            )
        ),
        inspected_image_positions=set(
            remap_positions(
                saved.inspected_image_positions, image_position_remap
            )
        ),
        inspected_proposition_positions=saved.inspected_proposition_positions,
        inspected_argument_positions=saved.inspected_argument_positions,
        inspected_statement_positions=set(
            remap_positions(
                saved.inspected_statement_positions, statement_position_remap
            )
        ),
        inspected_procedure_positions=set(
            remap_positions(
                saved.inspected_procedure_positions, procedure_position_remap
            )
        ),
        local_outcomes=[
            outcome.model_copy(
                update={
                    'block_positions': remap_positions(
                        outcome.block_positions, block_position_remap
                    )
                }
            )
            for outcome in saved.local_outcomes
        ],
        receipts=receipts,
        observations=[
            observation.model_copy(
                update={
                    'block_positions': remap_positions(
                        observation.block_positions, block_position_remap
                    ),
                    'image_positions': remap_positions(
                        observation.image_positions, image_position_remap
                    ),
                    'statement_positions': remap_positions(
                        observation.statement_positions,
                        statement_position_remap,
                    ),
                    'procedure_positions': remap_positions(
                        observation.procedure_positions,
                        procedure_position_remap,
                    ),
                    'groups': [
                        group.model_copy(
                            update={
                                'position': group_position_remap[
                                    group.position
                                ],
                                'member_block_positions': remap_positions(
                                    group.member_block_positions,
                                    block_position_remap,
                                ),
                                'governed_statement_positions': remap_positions(
                                    group.governed_statement_positions,
                                    statement_position_remap,
                                ),
                            }
                        )
                        for group in observation.groups
                        if group.position in group_position_remap
                    ],
                }
            )
            for observation in saved.observations
        ],
        active_block_positions=remap_positions(
            saved.active_block_positions, block_position_remap
        ),
        active_image_positions=remap_positions(
            saved.active_image_positions, image_position_remap
        ),
        active_proposition_positions=saved.active_proposition_positions,
        active_argument_positions=saved.active_argument_positions,
        active_statement_positions=remap_positions(
            saved.active_statement_positions, statement_position_remap
        ),
        active_procedure_positions=remap_positions(
            saved.active_procedure_positions, procedure_position_remap
        ),
        discovered_block_positions=remap_positions(
            saved.discovered_block_positions, block_position_remap
        ),
        discovered_image_positions=remap_positions(
            saved.discovered_image_positions, image_position_remap
        ),
        discovered_proposition_positions=saved.discovered_proposition_positions,
        discovered_argument_positions=saved.discovered_argument_positions,
        discovered_statement_positions=remap_positions(
            saved.discovered_statement_positions, statement_position_remap
        ),
        discovered_procedure_positions=remap_positions(
            saved.discovered_procedure_positions, procedure_position_remap
        ),
        active_learner_observation_positions=(
            saved.active_learner_observation_positions
        ),
        discovered_learner_observation_positions=(
            saved.discovered_learner_observation_positions
        ),
    )
