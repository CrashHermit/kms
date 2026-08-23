"""Compile canonical statement content for the graph."""

import asyncio

import dspy

from kms import config
from kms.construction import composition, knowledge
from kms.core import content, embeddings, llm, models, module


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


def statement_enrichment_input(
    bundle: models.ConstructionBundle,
    statement: models.Statement,
    index: models.KnowledgeIndex | None = None,
) -> models.StatementEnrichmentInput:
    """Builds one canonical statement-compilation input."""
    if not statement.uuid:
        raise ValueError('statement enrichment requires an assigned uuid')
    return models.StatementEnrichmentInput(
        statement_uuid=statement.uuid,
        statement=_content_from_composed(
            composition.compose_statement(bundle, statement)
        ),
        canonical_knowledge=knowledge.knowledge_for_statement(
            bundle, statement, index
        ).render(),
    )


class StatementEnrichmentSignature(dspy.Signature):
    r"""
    Compile the supplied source members into one complete canonical statement.

    Preserve everything needed to understand what the statement communicates,
    asserts, explains, or asks: qualifiers, conditions, labels, notation, and
    all requested parts. Include relevant source context, but do not solve a
    problem or add facts. Write standalone Markdown prose and LaTeX without
    referring to "above", "below", or the source document.

    Return only the compiled statement.
    """

    source_content: content.ContentParts = dspy.InputField(
        description='Ordered statement source text and figures.'
    )
    canonical_knowledge: str = dspy.InputField(
        description='Supported canonical concepts, relations, and facts.'
    )
    statement: str = dspy.OutputField(
        description='The complete standalone canonical statement.'
    )


class StatementEnricher(module.Module):
    """Writes the canonical ``Statement.statement`` field."""

    signature = StatementEnrichmentSignature
    record_name = 'statement_enrichment'

    def encode(
        self, source_content: content.Content, canonical_knowledge: str
    ) -> dict:
        return {
            'source_content': content.ContentParts(content=source_content),
            'canonical_knowledge': canonical_knowledge,
        }

    def decode(self, prediction, **inputs) -> str:
        return module.require_text(prediction.statement, 'statement')


class StatementEnrichmentNode:
    """Compiles and stores canonical statement fields."""

    def __init__(self, enricher: StatementEnricher) -> None:
        self._enricher = enricher

    async def run(self, current_state: dict) -> dict:
        from kms.core import state

        bundle = state.to_construction_bundle(current_state)
        if not bundle.statements:
            return {
                'statements_enriched': 0,
                'statement_enrichments': [],
                'statement_hub_records': [],
                'construction_bundle': bundle,
            }
        gate = llm.gate(
            config.get_settings().stages.statement_enrichment.max_concurrent_calls
        )

        async def compile_one(statement: models.Statement):
            if not statement.uuid:
                raise ValueError(
                    'statement enrichment requires an assigned uuid'
                )
            source_content = _content_from_composed(
                composition.compose_statement(bundle, statement)
            )
            canonical_knowledge = knowledge.knowledge_for_statement(
                bundle, statement, bundle.knowledge_index
            ).render()
            async with gate:
                return await self._enricher.aforward(
                    source_content=source_content,
                    canonical_knowledge=canonical_knowledge,
                )

        compiled = await asyncio.gather(
            *(compile_one(stmt) for stmt in bundle.statements)
        )
        statements_by_uuid = {
            statement.uuid: statement
            for statement in bundle.statements
            if statement.uuid
        }
        for stmt, text in zip(bundle.statements, compiled, strict=True):
            if stmt.uuid:
                statements_by_uuid[stmt.uuid].statement = text.strip()
        vectors = await embeddings.embedder().embed(
            [content.Content.from_text(text) for text in compiled]
        )
        source = bundle.source.key or ''
        enrichments = [
            {
                'uuid': stmt.uuid,
                'description': text,
                'statement': text,
                'embedding': vector,
            }
            for stmt, text, vector in zip(
                bundle.statements, compiled, vectors, strict=True
            )
            if stmt.uuid
        ]
        records = [
            models.StatementHubRecord(
                uuid=stmt.uuid,
                source=source,
                description=text,
                embedding=vector,
            )
            for stmt, text, vector in zip(
                bundle.statements, compiled, vectors, strict=True
            )
            if stmt.uuid
        ]
        bundle.statement_enrichments = enrichments
        bundle.statement_hub_records = records
        return {
            'statements_enriched': len(compiled),
            'statement_enrichments': enrichments,
            'statement_hub_records': records,
            'construction_bundle': bundle,
        }
