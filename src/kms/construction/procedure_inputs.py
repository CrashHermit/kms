"""Build procedure materialization inputs from construction state."""

from kms.construction import composition, knowledge
from kms.core import content, models, state


def _content(composed: composition.ComposedContent) -> content.Content:
    """Converts composed source parts to multimodal content."""
    parts: list[content.TextPart | content.ImagePart] = []
    for part in composed.parts:
        if part.type == models.NodeType.IMAGE:
            if part.image_path:
                try:
                    image = content.load_image(part.image_path)
                except OSError:
                    image = None
                if image:
                    parts.append(content.ImagePart(image=image))
            continue
        if part.content:
            parts.append(content.TextPart(text=part.content))
    return content.Content(parts=parts)


def _procedure_content(
    composed: composition.ComposedProcedure,
) -> content.Content:
    """Adds ordered procedure steps to composed source content."""
    result = list(_content(composed.content).parts)
    if composed.steps:
        result.append(content.TextPart(text='ORDERED STEPS:'))
        result.extend(
            content.TextPart(text=f'{step.index + 1}. {step.text}')
            for step in composed.steps
        )
    return content.Content(parts=result)


def _entity_definitions(
    bundle: models.ConstructionBundle, statement: models.Statement
) -> str:
    """Renders canonical knowledge relevant to one statement."""
    return knowledge.knowledge_for_statement(bundle, statement).render()


def build_procedure_materialization_inputs(
    bundle: models.ConstructionBundle,
) -> list[models.ProcedureMaterializationInput]:
    """Builds statement-centered materialization inputs from a bundle."""
    source = bundle.source.key or ''
    inputs: list[models.ProcedureMaterializationInput] = []

    def statement_id(statement: models.Statement) -> str:
        if statement.uuid is None:
            raise ValueError('procedure inputs require assigned statement uuids')
        return statement.uuid

    def procedure_id(procedure: models.Procedure) -> str:
        if procedure.uuid is None:
            raise ValueError('procedure inputs require assigned procedure uuids')
        return procedure.uuid

    for statement in bundle.statements:
        statement_uuid = statement_id(statement)
        statement_content = _content(
            composition.compose_statement(bundle, statement)
        )
        entity_definitions = _entity_definitions(bundle, statement)
        inputs.append(
            models.ProcedureMaterializationInput(
                source=source,
                statement_uuid=statement_uuid,
                statement=statement_content,
                entity_definitions=entity_definitions,
            )
        )
        attached = [
            procedure
            for procedure in bundle.procedures
            if (
                procedure.statement_uuid == statement_uuid
                or (
                    procedure.statement_uuid is None
                    and procedure.block == statement.block
                )
            )
        ]
        for procedure in attached:
            composed = composition.compose_procedure(bundle, procedure)
            inputs.append(
                models.ProcedureMaterializationInput(
                    source=source,
                    statement_uuid=statement_uuid,
                    statement=statement_content,
                    entity_definitions=entity_definitions,
                    procedure_uuid=procedure_id(procedure),
                    procedure=_procedure_content(composed),
                    has_steps=bool(procedure.steps),
                    member_count=len(procedure.members),
                )
            )
    return inputs


class ProcedureMaterializationInputNode:
    """Builds procedure materialization inputs without graph access."""

    async def run(self, current_state: state.State) -> dict:
        """Builds source-local materialization inputs from workflow state."""
        bundle = state.to_construction_bundle(current_state)
        inputs = build_procedure_materialization_inputs(bundle)
        bundle.procedure_materialization_inputs = inputs
        return {
            'procedure_materialization_inputs': inputs,
            'construction_bundle': bundle,
        }
