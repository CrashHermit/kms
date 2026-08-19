"""Enrich and embed general Statement and Procedure learning units."""

import asyncio
from collections.abc import Callable

import dspy

from kms import config
from kms.construction import composition, knowledge
from kms.core import content, embeddings, llm, models, module


def statement_enrichment_input(
    bundle: models.ConstructionBundle,
    statement: models.Statement,
    index: models.KnowledgeIndex | None = None,
) -> models.StatementEnrichmentInput:
    """Builds statement enrichment inputs without external reads."""
    composed = composition.compose_statement(bundle, statement)
    if not statement.uuid:
        raise ValueError('statement enrichment requires an assigned uuid')
    return models.StatementEnrichmentInput(
        statement_uuid=statement.uuid,
        statement=content.Content.from_text(composed.text),
        canonical_knowledge=knowledge.knowledge_for_statement(
            bundle, statement, index
        ).render(),
    )


def procedure_enrichment_input(
    bundle: models.ConstructionBundle,
    procedure: models.Procedure,
    index: models.KnowledgeIndex | None = None,
) -> models.ProcedureEnrichmentInput:
    """Builds procedure enrichment inputs without external reads."""
    composed = composition.compose_procedure(bundle, procedure)
    selected = knowledge.knowledge_for_procedure(bundle, procedure, index)
    if composed.statement is not None:
        statement = next(
            statement
            for statement in bundle.statements
            if (
                procedure.statement_uuid is not None
                and statement.uuid == procedure.statement_uuid
            )
            or (
                procedure.statement_uuid is None
                and statement.block == procedure.block
            )
        )
        selected = selected.union(
            knowledge.knowledge_for_statement(bundle, statement, index)
        )
    statement_content = content.Content.from_text('')
    if composed.statement is not None:
        statement_content = content.Content.from_text(composed.statement.text)
    procedure_parts = list(
        content.Content.from_text(composed.content.text).parts
    )
    if composed.steps:
        procedure_parts.append(content.TextPart(text='ORDERED STEPS:'))
        procedure_parts.extend(
            content.TextPart(text=f'{step.index + 1}. {step.text}')
            for step in composed.steps
        )
    if not procedure.uuid:
        raise ValueError('procedure enrichment requires an assigned uuid')
    return models.ProcedureEnrichmentInput(
        procedure_uuid=procedure.uuid,
        statement=statement_content,
        procedure=content.Content(parts=procedure_parts),
        canonical_knowledge=selected.render(),
    )


class StatementEnrichmentSignature(dspy.Signature):
    r"""
    Describe one source statement for semantic retrieval and clustering.

    State what the statement communicates, asserts, explains, or asks. Preserve
    important conditions and qualifiers. If it asks for work, describe the
    goal without solving it. Write a standalone description that does not
    refer to "the passage", "above", or source-specific navigation.

    The statement is source content. The supplied canonical knowledge is
    context for understanding it, not permission to add unsupported facts.
    Do not include an attached procedure or its solution: statements with and
    without procedures must have comparable statement representations.

    Return one concise paragraph and nothing else.
    """

    statement: content.ContentParts = dspy.InputField(
        description='The statement text and any figures it contains.'
    )
    canonical_knowledge: str = dspy.InputField(
        description=(
            'Canonical concepts, relations, and facts that clarify the '
            'statement; use only supported knowledge.'
        )
    )
    description: str = dspy.OutputField(
        description=(
            'A concise standalone description of what the statement '
            'communicates or asks, without solving it.'
        )
    )


class ProcedureEnrichmentSignature(dspy.Signature):
    r"""
    Describe one procedure for semantic retrieval and clustering.

    Explain the goal supplied by the statement and the general reasoning or
    action pattern used by the procedure. Mention the important concepts,
    canonical facts, intermediate reasoning, and kind of result or check when
    they are supported. Generalize away source-specific names and values when
    that preserves the method, but do not invent a method absent from the
    procedure.

    The statement tells you what the procedure accomplishes. The procedure
    and ordered steps tell you how it accomplishes it. Do not merely copy the
    solution or state only its final answer.

    Return one concise paragraph and nothing else.
    """

    statement: content.ContentParts = dspy.InputField(
        description='The goal statement and any figures it contains.'
    )
    procedure: content.ContentParts = dspy.InputField(
        description='The procedure text and its ordered steps.'
    )
    canonical_knowledge: str = dspy.InputField(
        description=(
            'Canonical concepts, relations, and facts used by the '
            'statement or procedure.'
        )
    )
    description: str = dspy.OutputField(
        description=(
            'A concise standalone description of the procedure goal and '
            'general method.'
        )
    )


class StatementEnricher(module.Module):
    """Writes semantic descriptions for source statements."""

    signature = StatementEnrichmentSignature
    record_name = 'statement_enrichment'

    def encode(
        self,
        statement: content.Content,
        canonical_knowledge: str,
    ) -> dict:
        """Builds the statement-enrichment signature inputs."""
        return {
            'statement': content.ContentParts(content=statement),
            'canonical_knowledge': canonical_knowledge,
        }

    def decode(self, prediction, **inputs) -> str:
        """Returns the generated statement description."""
        return prediction.description


class ProcedureEnricher(module.Module):
    """Writes semantic descriptions for source procedures."""

    signature = ProcedureEnrichmentSignature
    record_name = 'procedure_enrichment'

    def encode(
        self,
        statement: content.Content,
        procedure: content.Content,
        canonical_knowledge: str,
    ) -> dict:
        """Builds the procedure-enrichment signature inputs."""
        return {
            'statement': content.ContentParts(content=statement),
            'procedure': content.ContentParts(content=procedure),
            'canonical_knowledge': canonical_knowledge,
        }

    def decode(self, prediction, **inputs) -> str:
        """Returns the generated procedure description."""
        return prediction.description


def _with_description(
    source: content.Content, description: str
) -> content.Content:
    """Adds a generated semantic description to content for embedding."""
    return content.Content(
        parts=[
            *source.parts,
            content.TextPart(text=f'SEMANTIC DESCRIPTION: {description}'),
        ]
    )


class StatementEnrichmentNode:
    """Enriches typed statement inputs and persists derived values."""

    def __init__(self, enricher: StatementEnricher) -> None:
        self._enricher = enricher

    async def run(self, current_state: dict) -> dict:
        """Consumes prepared inputs without performing graph reads."""
        from kms.core import state

        bundle = state.to_construction_bundle(current_state)
        inputs = bundle.statement_enrichment_inputs
        if not inputs:
            return {
                'statements_enriched': 0,
                'statement_enrichments': [],
                'statement_hub_records': [],
                'construction_bundle': bundle,
            }
        gate = llm.gate(
            config.get_settings().stages.statement_enrichment.max_concurrent_calls
        )

        async def enrich(item: models.StatementEnrichmentInput) -> str:
            async with gate:
                return await self._enricher.aforward(
                    statement=item.statement,
                    canonical_knowledge=item.canonical_knowledge,
                )

        descriptions = await _ordered_results(inputs, enrich)
        vectors = await embeddings.embedder().embed(
            [
                _with_description(item.statement, description)
                for item, description in zip(inputs, descriptions, strict=True)
            ]
        )
        enrichments = [
            {
                'uuid': item.statement_uuid,
                'description': description,
                'embedding': vector,
            }
            for item, description, vector in zip(
                inputs, descriptions, vectors, strict=True
            )
        ]
        records = [
            models.StatementHubRecord(
                uuid=item.statement_uuid,
                source=bundle.source.key or '',
                description=description,
                embedding=vector,
            )
            for item, description, vector in zip(
                inputs, descriptions, vectors, strict=True
            )
        ]
        bundle.statement_enrichments = enrichments
        bundle.statement_hub_records = records
        return {
            'statements_enriched': len(inputs),
            'statement_enrichments': enrichments,
            'statement_hub_records': records,
            'construction_bundle': bundle,
        }


class ProcedureEnrichmentNode:
    """Enriches typed procedure inputs and persists derived values."""

    def __init__(self, enricher: ProcedureEnricher) -> None:
        self._enricher = enricher

    async def run(self, current_state: dict) -> dict:
        """Consumes prepared inputs without performing graph reads."""
        from kms.core import state

        bundle = state.to_construction_bundle(current_state)
        inputs = bundle.procedure_enrichment_inputs
        if not inputs:
            return {
                'procedures_enriched': 0,
                'procedure_enrichments': [],
                'procedure_hub_records': [],
                'construction_bundle': bundle,
            }
        gate = llm.gate(
            config.get_settings().stages.procedure_enrichment.max_concurrent_calls
        )

        async def enrich(item: models.ProcedureEnrichmentInput) -> str:
            async with gate:
                return await self._enricher.aforward(
                    statement=item.statement,
                    procedure=item.procedure,
                    canonical_knowledge=item.canonical_knowledge,
                )

        descriptions = await asyncio.gather(*(enrich(item) for item in inputs))
        vectors = await embeddings.embedder().embed(
            [
                _with_description(
                    content.Content(
                        parts=[*item.statement.parts, *item.procedure.parts]
                    ),
                    description,
                )
                for item, description in zip(inputs, descriptions, strict=True)
            ]
        )
        enrichments = [
            {
                'uuid': item.procedure_uuid,
                'description': description,
                'embedding': vector,
            }
            for item, description, vector in zip(
                inputs, descriptions, vectors, strict=True
            )
        ]
        records = [
            models.ProcedureHubRecord(
                uuid=item.procedure_uuid,
                source=bundle.source.key or '',
                description=description,
                embedding=vector,
            )
            for item, description, vector in zip(
                inputs, descriptions, vectors, strict=True
            )
        ]
        bundle.procedure_enrichments = enrichments
        bundle.procedure_hub_records = records
        return {
            'procedures_enriched': len(inputs),
            'procedure_enrichments': enrichments,
            'procedure_hub_records': records,
            'construction_bundle': bundle,
        }


async def _ordered_results(rows: list[dict], callback: Callable) -> list:
    """Runs callbacks concurrently while returning results in row order."""
    return list(await asyncio.gather(*(callback(row) for row in rows)))
