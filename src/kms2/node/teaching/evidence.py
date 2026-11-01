"""Evidence registry and passage helpers for teaching workflows."""

from kms2.core.model.block import SourceBlock
from kms2.core.model.knowledge.proposition import Proposition
from kms2.core.model.source import Source
from kms2.core.model.source_evidence import (
    SourceEvidenceBlock,
    SourceEvidenceGroup,
    SourcePassage,
)
from kms2.core.model.source_processing.admission import StagedSourceSnapshot
from kms2.core.model.teaching.inference import TeachingContextGroup
from kms2.core.model.teaching.workspace import (
    TeachingBlockRegistryEntry,
    TeachingEvidenceWorkspace,
    TeachingGroupRegistryEntry,
    TeachingImageRegistryEntry,
    TeachingPropositionRegistryEntry,
    TeachingReasoningRegistryEntry,
)
from kms2.core.model.visual_asset import VisualAsset


def register_block(
    state: TeachingEvidenceWorkspace,
    source: Source,
    block: SourceBlock,
    page_indexes: list[int],
) -> int:
    key = (source.uuid, source.revision, block.uuid)
    for position, entry in enumerate(state.block_registry):
        if (entry.source.uuid, entry.source.revision, entry.block.uuid) == key:
            entry.page_indexes = sorted(set(entry.page_indexes + page_indexes))
            return position
    state.block_registry.append(
        TeachingBlockRegistryEntry(
            source=source, page_indexes=sorted(set(page_indexes)), block=block
        )
    )
    return len(state.block_registry) - 1


def register_image(
    state: TeachingEvidenceWorkspace,
    asset: VisualAsset,
    source: Source,
    block_position: int,
) -> int:
    key = (source.uuid, source.revision, asset.uuid)
    for position, entry in enumerate(state.image_registry):
        if (entry.source.uuid, entry.source.revision, entry.asset.uuid) == key:
            append_unique(entry.block_positions, block_position)
            return position
    state.image_registry.append(
        TeachingImageRegistryEntry(
            asset=asset, source=source, block_positions=[block_position]
        )
    )
    return len(state.image_registry) - 1


def register_group(
    state: TeachingEvidenceWorkspace, source: Source, group: SourceEvidenceGroup
) -> int:
    for position, entry in enumerate(state.group_registry):
        if (
            entry.source.uuid == source.uuid
            and entry.source.revision == source.revision
            and entry.group.uuid == group.uuid
        ):
            entry.group = group
            return position
    state.group_registry.append(
        TeachingGroupRegistryEntry(source=source, group=group)
    )
    return len(state.group_registry) - 1


def register_proposition(
    state: TeachingEvidenceWorkspace, proposition: Proposition | str
) -> int:
    uuid = (
        proposition.uuid
        if isinstance(proposition, Proposition)
        else proposition
    )
    for position, entry in enumerate(state.proposition_registry):
        if entry.uuid == uuid:
            if isinstance(proposition, Proposition):
                entry.proposition = proposition
            return position
    state.proposition_registry.append(
        TeachingPropositionRegistryEntry(
            uuid=uuid,
            proposition=proposition
            if isinstance(proposition, Proposition)
            else None,
        )
    )
    return len(state.proposition_registry) - 1


def register_reasoning_refs(
    state: TeachingEvidenceWorkspace, uuids: list[str]
) -> None:
    """Replace reasoning discovery while preserving stable registry positions."""
    registry_positions = {
        entry.uuid: position
        for position, entry in enumerate(state.reasoning_registry)
    }
    positions: list[int] = []
    for uuid in uuids:
        if uuid not in registry_positions:
            registry_positions[uuid] = len(state.reasoning_registry)
            state.reasoning_registry.append(
                TeachingReasoningRegistryEntry(uuid=uuid)
            )
        append_unique(positions, registry_positions[uuid])
    state.discovered_reasoning_positions = positions


def register_occurrence_blocks(
    state: TeachingEvidenceWorkspace,
    passages: list[SourcePassage],
    *,
    inspected: bool = False,
) -> None:
    for passage in passages:
        for evidence in passage.blocks:
            position = register_block(
                state, passage.source, evidence.block, evidence.page_indexes
            )
            if inspected:
                state.inspected_block_positions.add(position)
            for asset in evidence.block.assets:
                register_image(state, asset, passage.source, position)


def context_group(
    group: SourceEvidenceGroup,
    position: int,
    block_positions: dict[str, int],
    group_positions: dict[str, int],
) -> TeachingContextGroup:
    return TeachingContextGroup(
        position=position,
        kind=group.kind,
        member_block_positions=[
            block_positions[uuid] for uuid in group.member_block_uuids
        ],
        governed_statement_positions=[
            group_positions[uuid] for uuid in group.governed_statement_uuids
        ],
    )


def append_unique[T](values: list[T], value: T) -> None:
    if value not in values:
        values.append(value)


def staged_source_passages(
    snapshot: StagedSourceSnapshot,
    block_uuids: list[str],
    *,
    before: int,
    after: int,
) -> SourcePassage:
    """Build an ordered staged passage around requested blocks."""
    ordered: list[SourceEvidenceBlock] = []
    positions: dict[str, int] = {}
    for page in snapshot.pages:
        for block in page.blocks:
            position = positions.get(block.uuid)
            if position is None:
                positions[block.uuid] = len(ordered)
                ordered.append(
                    SourceEvidenceBlock(block=block, page_indexes=[page.index])
                )
            else:
                append_unique(ordered[position].page_indexes, page.index)
    targets = list(dict.fromkeys(block_uuids))
    target_positions = [positions[uuid] for uuid in targets]
    included: set[int] = set()
    for position in target_positions:
        included.update(
            range(
                max(0, position - before),
                min(len(ordered), position + after + 1),
            )
        )
    return SourcePassage(
        source=snapshot.source,
        selected_block_uuids=targets,
        blocks=[ordered[index] for index in sorted(included)],
        groups=[],
    )


def active_block_positions(state: TeachingEvidenceWorkspace) -> list[int]:
    """Return the inspected active blocks, including active group members."""
    positions = set(state.active_block_positions)
    for registry_position in state.active_statement_positions:
        entry = state.statement_registry[registry_position]
        positions.update(
            position
            for position, block in enumerate(state.block_registry)
            if block.source.uuid == entry.source.uuid
            and block.source.revision == entry.source.revision
            and block.block.uuid in entry.statement.member_block_uuids
        )
    for registry_position in state.active_procedure_positions:
        entry = state.procedure_registry[registry_position]
        positions.update(
            position
            for position, block in enumerate(state.block_registry)
            if block.source.uuid == entry.source.uuid
            and block.source.revision == entry.source.revision
            and block.block.uuid in entry.procedure.member_block_uuids
        )
    return sorted(positions & state.inspected_block_positions)


def active_image_positions(state: TeachingEvidenceWorkspace) -> list[int]:
    """Return the stable active image positions that have read payloads."""
    return sorted(
        set(state.active_image_positions) & state.inspected_image_positions
    )


def active_proposition_positions(
    state: TeachingEvidenceWorkspace,
) -> list[int]:
    """Return active inspected propositions and active reasoning endpoints."""
    positions = set(state.active_proposition_positions)
    for registry_position in state.active_reasoning_positions:
        record = state.reasoning_registry[registry_position].record
        if record is None:
            continue
        endpoint_uuids = {*record.premise_uuids, record.conclusion_uuid}
        positions.update(
            position
            for position, entry in enumerate(state.proposition_registry)
            if entry.uuid in endpoint_uuids
        )
    return sorted(positions & state.inspected_proposition_positions)


def active_reasoning_positions(state: TeachingEvidenceWorkspace) -> list[int]:
    """Return the inspected active reasoning records in registry order."""
    return sorted(
        set(state.active_reasoning_positions)
        & state.inspected_reasoning_positions
    )


def active_statement_positions(state: TeachingEvidenceWorkspace) -> list[int]:
    """Return the inspected active statement records in registry order."""
    return sorted(
        set(state.active_statement_positions)
        & state.inspected_statement_positions
    )


def active_procedure_positions(state: TeachingEvidenceWorkspace) -> list[int]:
    """Return the inspected active procedure records in registry order."""
    return sorted(
        set(state.active_procedure_positions)
        & state.inspected_procedure_positions
    )
