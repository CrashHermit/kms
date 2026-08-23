"""Compile canonical procedure content for the graph."""

import asyncio

import dspy

from kms import config
from kms.construction import composition, knowledge
from kms.core import (
    content,
    embeddings,
    identity,
    llm,
    models,
    module,
    recording,
)


def _content_from_composed(
    composed: composition.ComposedContent,
) -> content.Content:
    """Converts ordered source parts into multimodal LLM content."""
    parts: list[content.TextPart | content.ImagePart] = []
    max_dim = config.get_settings().image.max_dim
    for part in composed.parts:
        if part.type == models.NodeType.IMAGE:
            if part.image_path:
                try:
                    image = content.load_image(part.image_path, max_dim=max_dim)
                except OSError:
                    image = None
                if image is not None:
                    parts.append(content.ImagePart(image=image))
            continue
        if part.content:
            parts.append(content.TextPart(text=part.content))
    return content.Content(parts=parts)


def _statement_for_procedure(
    bundle: models.ConstructionBundle, procedure: models.Procedure
) -> models.Statement:
    """Finds the explicit owner of a procedure."""
    if procedure.statement_uuid:
        statement = next(
            (
                item
                for item in bundle.statements
                if item.uuid == procedure.statement_uuid
            ),
            None,
        )
        if statement is None:
            raise ValueError(
                f'procedure references unknown statement UUID '
                f'{procedure.statement_uuid!r}'
            )
        return statement
    matches = [
        item for item in bundle.statements if item.block == procedure.block
    ]
    if len(matches) != 1:
        raise ValueError('procedure must have exactly one statement owner')
    return matches[0]


def procedure_enrichment_input(
    bundle: models.ConstructionBundle,
    statement: models.Statement,
    procedure: models.Procedure | None,
    index: models.KnowledgeIndex | None = None,
) -> models.ProcedureEnrichmentInput:
    """Builds one statement-centered procedure-compilation input."""
    if not statement.uuid:
        raise ValueError(
            'procedure enrichment requires an assigned statement uuid'
        )
    statement_content = (
        content.Content.from_text(statement.statement)
        if statement.statement
        else _content_from_composed(
            composition.compose_statement(bundle, statement)
        )
    )
    source_procedure = None
    procedure_uuid = None
    if procedure is not None:
        if not procedure.uuid:
            raise ValueError(
                'procedure enrichment requires assigned procedure uuids'
            )
        procedure_uuid = procedure.uuid
        if procedure.member_positions:
            source_procedure = _content_from_composed(
                composition.compose_procedure(bundle, procedure).content
            )
        elif procedure.procedure:
            source_procedure = content.Content.from_text(procedure.procedure)
    selected = knowledge.knowledge_for_statement(bundle, statement, index)
    if procedure is not None:
        selected = selected.union(
            knowledge.knowledge_for_procedure(bundle, procedure, index)
        )
    return models.ProcedureEnrichmentInput(
        source=bundle.source.key or '',
        statement_uuid=statement.uuid,
        statement=statement_content,
        procedure_uuid=procedure_uuid,
        procedure=source_procedure,
        canonical_knowledge=selected.render(),
    )


class ProcedureNeedRouterSignature(dspy.Signature):
    r"""
    Decide whether the supplied statement source requires a procedure.

    The statement may contain text, equations, diagrams, or other images.
    Return True when fulfilling it requires a proof, derivation, calculation,
    construction, verification, or other worked solution. Return False for
    definitions, notation, explanatory prose, headings, and assertions that
    can be understood without producing work. Return only True or False.
    """

    statement: content.ContentParts = dspy.InputField(
        description='The complete canonical statement as ordered text and images.'
    )
    needs_procedure: bool = dspy.OutputField(
        description='Whether the statement requires a worked procedure.'
    )


class ProcedureNeedRouter(module.Module):
    """Routes only statements that need a worked procedure to the writer."""

    signature = ProcedureNeedRouterSignature
    record_name = 'procedure_need_router'
    use_chain_of_thought = True

    def encode(self, statement: content.Content) -> dict:
        return {'statement': content.ContentParts(content=statement)}

    def decode(self, prediction, **inputs) -> bool:
        """Returns the validated procedure-routing decision."""
        return module.require_bool(
            prediction.needs_procedure, 'needs_procedure'
        )


class ProcedureWriterSignature(dspy.Signature):
    r"""
    Compile one complete canonical procedure for the supplied statement.

    The statement is authoritative about the goal and requested parts. If
    SOURCE PROCEDURE CONTENT is supplied, preserve its supported reasoning and
    use it as evidence while rewriting it into a complete standalone
    procedure. If it is empty, write the correct proof, solution, derivation,
    calculation, or construction needed by the statement. Make meaningful
    reasoning explicit, answer every requested part, and do not invent facts.

    Return continuous prose with Markdown LaTeX. Return only the procedure.
    """

    statement: content.ContentParts = dspy.InputField(
        description='The complete canonical statement and learner goal as ordered text and images.'
    )
    source_procedure: content.ContentParts = dspy.InputField(
        description='Existing source procedure content, if any.'
    )
    canonical_knowledge: str = dspy.InputField(
        description='Supported canonical concepts, relations, and facts.'
    )
    procedure: str = dspy.OutputField(
        description='The complete standalone worked procedure.'
    )


class ProcedureWriter(module.Module):
    """Writes the canonical ``Procedure.procedure`` field."""

    signature = ProcedureWriterSignature
    record_name = 'procedure_writer'

    def encode(
        self,
        statement: content.Content,
        source_procedure: content.Content | None,
        canonical_knowledge: str,
    ) -> dict:
        return {
            'statement': content.ContentParts(content=statement),
            'source_procedure': content.ContentParts(
                content=source_procedure or content.Content.from_text('')
            ),
            'canonical_knowledge': canonical_knowledge,
        }

    def decode(self, prediction, **inputs) -> str:
        return module.require_text(prediction.procedure, 'procedure')


class ProcedureEnricher:
    """Compiles canonical procedure content after statement enrichment."""

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        self.router = ProcedureNeedRouter(language_model, recorder=recorder)
        self.writer = ProcedureWriter(language_model, recorder=recorder)

    async def compile(
        self, item: models.ProcedureEnrichmentInput
    ) -> tuple[bool, str | None]:
        """Returns whether a procedure was needed and its compiled text."""
        statement = item.statement
        if not statement.parts or not statement.render().strip():
            return False, None
        needs_procedure = True
        if item.procedure_uuid is None:
            needs_procedure = await self.router.aforward(statement=statement)
            if not needs_procedure:
                return False, None
        result = await self.writer.aforward(
            statement=statement,
            source_procedure=item.procedure,
            canonical_knowledge=item.canonical_knowledge,
        )
        result = result.strip()
        return needs_procedure, result or None


def build_procedure_inputs(
    bundle: models.ConstructionBundle,
) -> list[models.ProcedureEnrichmentInput]:
    """Builds statement-centered inputs from the current bundle."""
    inputs: list[models.ProcedureEnrichmentInput] = []
    by_statement: dict[str, list[models.Procedure]] = {}
    for procedure in bundle.procedures:
        owner = _statement_for_procedure(bundle, procedure)
        if not owner.uuid:
            raise ValueError('procedure owner is missing a uuid')
        by_statement.setdefault(owner.uuid, []).append(procedure)
    for statement in bundle.statements:
        if not statement.uuid:
            raise ValueError(
                'procedure inputs require assigned statement uuids'
            )
        attached = by_statement.get(statement.uuid, [])
        if attached:
            inputs.extend(
                procedure_enrichment_input(
                    bundle, statement, procedure, bundle.knowledge_index
                )
                for procedure in attached
            )
        else:
            inputs.append(
                procedure_enrichment_input(
                    bundle, statement, None, bundle.knowledge_index
                )
            )
    return inputs


class ProcedureEnrichmentNode:
    """Compiles and stores canonical procedure fields."""

    def __init__(self, enricher: ProcedureEnricher) -> None:
        self._enricher = enricher

    async def run(self, current_state: dict) -> dict:
        from kms.core import state

        bundle = state.to_construction_bundle(current_state)
        if not bundle.statements:
            return {
                'procedures_enriched': 0,
                'procedure_enrichments': [],
                'procedure_hub_records': [],
                'construction_bundle': bundle,
            }
        gate = llm.gate(
            config.get_settings().stages.procedure_enrichment.max_concurrent_calls
        )

        # Build procedure inputs inline (statement-centered)
        by_statement: dict[str, list[models.Procedure]] = {}
        for procedure in bundle.procedures:
            owner = _statement_for_procedure(bundle, procedure)
            if not owner.uuid:
                raise ValueError('procedure owner is missing a uuid')
            by_statement.setdefault(owner.uuid, []).append(procedure)

        procedure_inputs = []
        for statement in bundle.statements:
            if not statement.uuid:
                raise ValueError(
                    'procedure inputs require assigned statement uuids'
                )
            attached = by_statement.get(statement.uuid, [])
            if attached:
                for procedure in attached:
                    if not statement.uuid:
                        raise ValueError(
                            'procedure enrichment requires an assigned statement uuid'
                        )
                    statement_content = (
                        content.Content.from_text(statement.statement)
                        if statement.statement
                        else _content_from_composed(
                            composition.compose_statement(bundle, statement)
                        )
                    )
                    source_procedure = None
                    procedure_uuid = None
                    if procedure is not None:
                        if not procedure.uuid:
                            raise ValueError(
                                'procedure enrichment requires assigned procedure uuids'
                            )
                        procedure_uuid = procedure.uuid
                        if procedure.member_positions:
                            source_procedure = _content_from_composed(
                                composition.compose_procedure(
                                    bundle, procedure
                                ).content
                            )
                        elif procedure.procedure:
                            source_procedure = content.Content.from_text(
                                procedure.procedure
                            )
                    selected = knowledge.knowledge_for_statement(
                        bundle, statement, bundle.knowledge_index
                    )
                    selected = selected.union(
                        knowledge.knowledge_for_procedure(
                            bundle, procedure, bundle.knowledge_index
                        )
                    )
                    procedure_inputs.append(
                        models.ProcedureEnrichmentInput(
                            source=bundle.source.key or '',
                            statement_uuid=statement.uuid,
                            statement=statement_content,
                            procedure_uuid=procedure_uuid,
                            procedure=source_procedure,
                            canonical_knowledge=selected.render(),
                        )
                    )
            else:
                if not statement.uuid:
                    raise ValueError(
                        'procedure enrichment requires an assigned statement uuid'
                    )
                statement_content = (
                    content.Content.from_text(statement.statement)
                    if statement.statement
                    else _content_from_composed(
                        composition.compose_statement(bundle, statement)
                    )
                )
                selected = knowledge.knowledge_for_statement(
                    bundle, statement, bundle.knowledge_index
                )
                procedure_inputs.append(
                    models.ProcedureEnrichmentInput(
                        source=bundle.source.key or '',
                        statement_uuid=statement.uuid,
                        statement=statement_content,
                        procedure_uuid=None,
                        procedure=None,
                        canonical_knowledge=selected.render(),
                    )
                )

        if not procedure_inputs:
            return {
                'procedures_enriched': 0,
                'procedure_enrichments': [],
                'procedure_hub_records': [],
                'construction_bundle': bundle,
            }
        gate = llm.gate(
            config.get_settings().stages.procedure_enrichment.max_concurrent_calls
        )

        async def compile_one(item):
            async with gate:
                return item, await self._enricher.compile(item)

        results = await asyncio.gather(
            *(compile_one(item) for item in procedure_inputs)
        )
        statements = {
            statement.uuid: statement
            for statement in bundle.statements
            if statement.uuid
        }
        procedures = {
            procedure.uuid: procedure
            for procedure in bundle.procedures
            if procedure.uuid
        }
        generated: list[models.Procedure] = []
        links: list[models.ProcedureLink] = []
        completed: list[tuple[models.Procedure, str]] = []
        for item, (needed, text) in results:
            if not needed or not text:
                continue
            if item.procedure_uuid is None:
                procedure_uuid = identity.procedure_uuid(
                    item.source,
                    [],
                    0,
                    statement_uuid_value=item.statement_uuid,
                )
                procedure = models.Procedure(
                    block=[],
                    index=0,
                    uuid=procedure_uuid,
                    statement_uuid=item.statement_uuid,
                    procedure=text,
                )
                generated.append(procedure)
                links.append(
                    models.ProcedureLink(
                        source=item.source,
                        statement_uuid=item.statement_uuid,
                        procedure_uuid=procedure_uuid,
                    )
                )
            else:
                procedure = procedures[item.procedure_uuid]
                procedure.procedure = text
            completed.append((procedure, text))
        bundle.procedures.extend(generated)
        bundle.generated_procedures = generated
        bundle.procedure_links = links
        vectors = await embeddings.embedder().embed(
            [
                content.Content.from_text(
                    f'{statements[procedure.statement_uuid].statement}\n\n{text}'
                )
                for procedure, text in completed
                if procedure.statement_uuid in statements
            ]
        )
        source = bundle.source.key or ''
        enrichments = [
            {
                'uuid': procedure.uuid,
                'description': text,
                'procedure': text,
                'embedding': vector,
            }
            for (procedure, text), vector in zip(
                completed, vectors, strict=True
            )
            if procedure.uuid
        ]
        records = [
            models.ProcedureHubRecord(
                uuid=procedure.uuid,
                source=source,
                description=text,
                embedding=vector,
            )
            for (procedure, text), vector in zip(
                completed, vectors, strict=True
            )
            if procedure.uuid
        ]
        bundle.procedure_enrichments = enrichments
        bundle.procedure_hub_records = records
        return {
            'procedures_enriched': len(completed),
            'procedure_enrichments': enrichments,
            'procedure_hub_records': records,
            'procedures': bundle.procedures,
            'generated_procedures': generated,
            'procedure_links': links,
            'construction_bundle': bundle,
        }
