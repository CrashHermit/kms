"""Assemble classified source blocks into ordered nonempty units."""

from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.processing.state import SourceProcessingState


class SourceUnitAssemblyNode:
    """Combine classified units and retain each unclaimed block as content."""

    async def run(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourceUnit]]:
        blocks = state.procedure_split_blocks
        positions = {
            block.uuid: position for position, block in enumerate(blocks)
        }
        units = [*state.exercise_units, *state.procedure_units]
        claimed = {block.uuid for unit in units for block in unit.blocks}
        units.extend(
            SourceUnit(kind=SourceUnitKind.CONTENT, blocks=[block])
            for block in blocks
            if block.uuid not in claimed
        )
        units.sort(
            key=lambda unit: min(positions[block.uuid] for block in unit.blocks)
        )
        return {'units': units}
