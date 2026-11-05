"""Evidence registry and passage helpers for teaching workflows."""

from kms.core.model.knowledge.proposition import Proposition
from kms.core.model.material import SourceGroupView
from kms.core.model.source import Source
from kms.core.model.source_evidence import SourceEvidenceGroup, SourcePassage
from kms.core.model.teaching.observation import LearnerObservation
from kms.core.model.teaching.workspace import (
    TeachingArgumentRegistryEntry,
    TeachingEvidenceWorkspace,
    TeachingGroupRegistryEntry,
    TeachingLearnerObservationRegistryEntry,
    TeachingPropositionRegistryEntry,
)
from kms.core.source_workspace import (
    append_unique,
    register_block,
    register_image,
)


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


def register_argument_refs(
    state: TeachingEvidenceWorkspace, uuids: list[str]
) -> None:
    """Replace argument discovery while preserving stable registry positions."""
    registry_positions = {
        entry.uuid: position
        for position, entry in enumerate(state.argument_registry)
    }
    positions: list[int] = []
    for uuid in uuids:
        if uuid not in registry_positions:
            registry_positions[uuid] = len(state.argument_registry)
            state.argument_registry.append(
                TeachingArgumentRegistryEntry(uuid=uuid)
            )
        append_unique(positions, registry_positions[uuid])
    state.discovered_argument_positions = positions


def register_learner_observation(
    state: TeachingEvidenceWorkspace, observation: LearnerObservation
) -> int:
    """Keep one stable registry slot per learner observation UUID."""
    for position, entry in enumerate(state.learner_observation_registry):
        if entry.observation.uuid == observation.uuid:
            entry.observation = observation
            return position
    state.learner_observation_registry.append(
        TeachingLearnerObservationRegistryEntry(observation=observation)
    )
    return len(state.learner_observation_registry) - 1


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
            else:
                state.read_block_positions.add(position)
            for asset in evidence.block.assets:
                image_position = register_image(
                    state, asset, passage.source, position
                )
                if inspected:
                    state.inspected_image_positions.add(image_position)


def context_group(
    group: SourceEvidenceGroup,
    position: int,
    block_positions: dict[str, int],
    group_positions: dict[str, int],
) -> SourceGroupView:
    return SourceGroupView(
        position=position,
        kind=group.kind,
        member_block_positions=[
            block_positions[uuid] for uuid in group.member_block_uuids
        ],
        governed_statement_positions=[
            group_positions[uuid] for uuid in group.governed_statement_uuids
        ],
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
    readable_or_inspected = (
        state.read_block_positions | state.inspected_block_positions
    )
    return sorted(positions & readable_or_inspected)


def active_image_positions(state: TeachingEvidenceWorkspace) -> list[int]:
    """Return the stable active image positions that have read payloads."""
    return sorted(
        set(state.active_image_positions)
        & (state.read_image_positions | state.inspected_image_positions)
    )


def active_proposition_positions(
    state: TeachingEvidenceWorkspace,
) -> list[int]:
    """Return active inspected propositions and active argument endpoints."""
    positions = set(state.active_proposition_positions)
    for registry_position in state.active_argument_positions:
        record = state.argument_registry[registry_position].record
        if record is None:
            continue
        endpoint_uuids = {*record.premise_uuids, record.conclusion_uuid}
        positions.update(
            position
            for position, entry in enumerate(state.proposition_registry)
            if entry.uuid in endpoint_uuids
        )
    return sorted(positions & state.inspected_proposition_positions)


def active_argument_positions(state: TeachingEvidenceWorkspace) -> list[int]:
    """Return the inspected active arguments in registry order."""
    return sorted(
        set(state.active_argument_positions)
        & state.inspected_argument_positions
    )


def active_statement_positions(state: TeachingEvidenceWorkspace) -> list[int]:
    """Return active source groups whose source material was made readable."""
    readable = {
        position
        for position, entry in enumerate(state.statement_registry)
        if entry.passage is not None
    } | state.inspected_statement_positions
    return sorted(set(state.active_statement_positions) & readable)


def active_procedure_positions(state: TeachingEvidenceWorkspace) -> list[int]:
    """Return active source groups whose source material was made readable."""
    readable = {
        position
        for position, entry in enumerate(state.procedure_registry)
        if entry.passage is not None
    } | state.inspected_procedure_positions
    return sorted(set(state.active_procedure_positions) & readable)
