"""Compile canonical procedure content for the graph."""

import asyncio

import dspy

from kms import config
from kms.construction import composition, knowledge
from kms.core import (
    embeddings,
    llm,
    models,
    module,
    recording,
)


def _text_nodes(
    composed: composition.ComposedContent,
) -> tuple[models.TextNodeInput, ...]:
    return tuple(
        models.TextNodeInput(
            local_index=index,
            node_type=part.type.value if part.type else '',
            node_text=part.content or '',
        )
        for index, part in enumerate(composed.parts)
    )


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
    statement_content = _text_nodes(
        composition.compose_statement(bundle, statement)
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
            source_procedure = _text_nodes(
                composition.compose_procedure(bundle, procedure).content
            )
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


class SourceProcedureWriterSignature(dspy.Signature):
    r"""
    Compile the supplied source procedure members into one canonical procedure.

    The source procedure content is authoritative. Preserve its supported
    reasoning, ordered steps, conditions, qualifiers, and mathematical
    notation while resolving local references and removing source-navigation
    language. Do not add omitted steps, solve the statement, or invent facts.

    Return continuous prose with Markdown LaTeX. Return only the
    source-supported procedure.
    """

    statement: list[models.TextNodeInput] = dspy.InputField(
        description='The complete canonical statement that the source procedure addresses.'
    )
    source_procedure: list[models.TextNodeInput] = dspy.InputField(
        description='The source procedure members to compile.'
    )
    canonical_knowledge: str = dspy.InputField(
        description='Supported canonical concepts, relations, and facts.'
    )
    procedure: str = dspy.OutputField(
        description='The complete source-supported canonical procedure.'
    )


class SourceProcedureWriter(module.Module):
    """Writes canonical content for a source-backed procedure."""

    signature = SourceProcedureWriterSignature
    record_name = 'source_procedure_writer'

    def encode(
        self,
        statement: list[models.TextNodeInput],
        source_procedure: list[models.TextNodeInput],
        canonical_knowledge: str,
    ) -> dict:
        return {
            'statement': statement,
            'source_procedure': source_procedure,
            'canonical_knowledge': canonical_knowledge,
        }

    def decode(self, prediction, **inputs) -> str:
        return module.require_text(prediction.procedure, 'procedure')


class ProcedureEnricher:
    """Canonicalizes procedures supported by source evidence."""

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        self.writer = SourceProcedureWriter(language_model, recorder=recorder)

    async def compile(
        self, item: models.ProcedureEnrichmentInput
    ) -> tuple[bool, str | None]:
        """Returns whether source procedure content was compiled."""
        source_procedure = list(item.procedure or ())
        if item.procedure_uuid is None or not source_procedure:
            return False, None
        statement = list(item.statement)
        if not any(node.node_text.strip() for node in statement):
            return False, None
        result = await self.writer.aforward(
            statement=statement,
            source_procedure=source_procedure,
            canonical_knowledge=item.canonical_knowledge,
        )
        result = result.strip()
        return bool(result), result or None


def build_procedure_inputs(
    bundle: models.ConstructionBundle,
) -> list[models.ProcedureEnrichmentInput]:
    """Builds inputs only for procedures with source members."""
    inputs: list[models.ProcedureEnrichmentInput] = []
    for procedure in bundle.procedures:
        if procedure.kind is not models.ProcedureKind.SOURCE:
            continue
        owner = _statement_for_procedure(bundle, procedure)
        if not owner.uuid:
            raise ValueError('procedure owner is missing a uuid')
        inputs.append(
            procedure_enrichment_input(
                bundle, owner, procedure, bundle.knowledge_index
            )
        )
    return inputs


class ProcedureEnrichmentNode:
    """Compiles canonical content for source-backed procedures."""

    def __init__(self, enricher: ProcedureEnricher) -> None:
        self._enricher = enricher

    async def run(self, current_state: dict) -> dict:
        from kms.core import state

        bundle = state.to_construction_bundle(current_state)
        procedure_inputs = build_procedure_inputs(bundle)
        if not procedure_inputs:
            return {
                'procedures_enriched': 0,
                'procedure_enrichments': [],
                'procedure_hub_records': [],
                'generated_procedures': [],
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
        completed: list[tuple[models.Procedure, str]] = []
        for item, (compiled, text) in results:
            if not compiled or not text or item.procedure_uuid is None:
                continue
            procedure = procedures[item.procedure_uuid]
            procedure.procedure = text
            completed.append((procedure, text))
        if not completed:
            return {
                'procedures_enriched': 0,
                'procedure_enrichments': [],
                'procedure_hub_records': [],
                'generated_procedures': [],
                'construction_bundle': bundle,
            }
        vectors = await embeddings.embedder().embed(
            [
                f'{statements[procedure.statement_uuid].statement}\n\n{text}'
                for procedure, text in completed
                if procedure.statement_uuid in statements
            ]
        )
        source = bundle.source.key or ''
        enrichments = [
            {
                'uuid': procedure.uuid,
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
        bundle.generated_procedures = []
        return {
            'procedures_enriched': len(completed),
            'procedure_enrichments': enrichments,
            'procedure_hub_records': records,
            'procedures': bundle.procedures,
            'generated_procedures': [],
            'construction_bundle': bundle,
        }
