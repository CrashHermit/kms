"""Enrich and embed general Statement and Procedure learning units."""

import asyncio
from collections.abc import Callable

import dspy

from kms import config
from kms.core import content, embeddings, llm, module
from kms.graph import queries, writer


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


def _knowledge_text(groups: list[dict]) -> str:
    """Formats canonical graph knowledge returned for one source unit."""
    parts: list[str] = []
    for group in groups:
        name = group.get('name') or group.get('canonical_name') or ''
        description = group.get('description') or ''
        predicate_name = group.get('predicate_name') or ''
        predicate_description = group.get('predicate_description') or ''
        fact_name = group.get('fact_name') or ''
        fact_description = group.get('fact_description') or ''
        if name and description:
            parts.append(f'{name}: {description}')
        if predicate_name and predicate_description:
            parts.append(f'{predicate_name}: {predicate_description}')
        if fact_name and fact_description:
            parts.append(f'FACT: {fact_name}: {fact_description}')
    return '\n'.join(dict.fromkeys(parts))


async def _statement_items(
    session_factory: Callable, source: str
) -> list[dict]:
    """Loads statement records with their source-local graph context."""
    rows = await queries.statement_enrichment_items(session_factory, source)
    for row in rows:
        composed = await queries.compose_statement(
            row['statement_uuid'], session_factory
        )
        row['content'] = composed
        row['canonical_knowledge'] = _knowledge_text(
            await queries.statement_knowledge(
                row['statement_uuid'], session_factory
            )
        )
    return rows


async def _procedure_items(
    session_factory: Callable, source: str
) -> list[dict]:
    """Loads procedure records with their statement and graph context."""
    rows = await queries.procedure_enrichment_items(session_factory, source)
    for row in rows:
        procedure = await queries.compose_procedure(
            row['procedure_uuid'], session_factory
        )
        row['procedure_content'] = procedure
        if row['statement_uuid']:
            statement = await queries.compose_statement(
                row['statement_uuid'], session_factory
            )
            row['statement_content'] = statement
            row['canonical_knowledge'] = _knowledge_text(
                await queries.statement_knowledge(
                    row['statement_uuid'], session_factory
                )
            )
        else:
            row['statement_content'] = {'text': '', 'pictures': []}
            row['canonical_knowledge'] = ''
    return rows


def _content_from_composed(composed: dict) -> content.Content:
    """Builds multimodal content from a composed graph record."""
    return content.Content.from_text_and_pictures(
        composed.get('text', ''), composed.get('pictures', [])
    )


def _procedure_content(composed: dict) -> content.Content:
    """Builds procedure content from source members and ordered steps."""
    parts: list[content.TextPart | content.ImagePart] = []
    source_content = _content_from_composed(composed)
    parts.extend(source_content.parts)
    steps = composed.get('steps') or []
    if steps:
        parts.append(content.TextPart(text='ORDERED STEPS:'))
        for step in steps:
            parts.append(
                content.TextPart(text=f'{step["index"] + 1}. {step["text"]}')
            )
    return content.Content(parts=parts)


class StatementEnrichmentNode:
    """Enriches and embeds every persisted Statement."""

    def __init__(
        self,
        session_factory: Callable | None,
        enricher: StatementEnricher,
    ) -> None:
        self._session_factory = session_factory
        self._enricher = enricher

    async def run(self, current_state: dict) -> dict:
        """Writes statement descriptions and embeddings to the graph."""
        if not self._session_factory:
            return {}
        rows = await _statement_items(
            self._session_factory, current_state['source']
        )
        if not rows:
            return {'statements_enriched': 0}
        gate = llm.gate(
            config.get_settings().stages.statement_enrichment.max_concurrent_calls
        )

        async def enrich(row: dict) -> str:
            async with gate:
                return await self._enricher.aforward(
                    statement=_content_from_composed(row['content']),
                    canonical_knowledge=row['canonical_knowledge'],
                )

        descriptions = await _ordered_results(rows, enrich)
        embedder = embeddings.embedder()
        vectors = await embedder.embed(
            [
                _with_description(
                    _content_from_composed(row['content']), description
                )
                for row, description in zip(rows, descriptions, strict=True)
            ]
        )
        await writer.persist_statement_enrichment(
            [
                {
                    'uuid': row['statement_uuid'],
                    'description': description,
                    'embedding': vector,
                }
                for row, description, vector in zip(
                    rows, descriptions, vectors, strict=True
                )
            ],
            session_factory=self._session_factory,
        )
        return {'statements_enriched': len(rows)}


class ProcedureEnrichmentNode:
    """Enriches and embeds every persisted Procedure."""

    def __init__(
        self,
        session_factory: Callable | None,
        enricher: ProcedureEnricher,
    ) -> None:
        self._session_factory = session_factory
        self._enricher = enricher

    async def run(self, current_state: dict) -> dict:
        """Writes procedure descriptions and embeddings to the graph."""
        if not self._session_factory:
            return {}
        rows = await _procedure_items(
            self._session_factory, current_state['source']
        )
        if not rows:
            return {'procedures_enriched': 0}
        gate = llm.gate(
            config.get_settings().stages.procedure_enrichment.max_concurrent_calls
        )

        async def enrich(row: dict) -> str:
            async with gate:
                return await self._enricher.aforward(
                    statement=_content_from_composed(row['statement_content']),
                    procedure=_procedure_content(row['procedure_content']),
                    canonical_knowledge=row['canonical_knowledge'],
                )

        descriptions = await asyncio.gather(*(enrich(row) for row in rows))
        embedder = embeddings.embedder()
        vectors = await embedder.embed(
            [
                _with_description(
                    content.Content(
                        parts=[
                            *(
                                _content_from_composed(
                                    row['statement_content']
                                ).parts
                            ),
                            *(
                                _procedure_content(
                                    row['procedure_content']
                                ).parts
                            ),
                        ]
                    ),
                    description,
                )
                for row, description in zip(rows, descriptions, strict=True)
            ]
        )
        await writer.persist_procedure_enrichment(
            [
                {
                    'uuid': row['procedure_uuid'],
                    'description': description,
                    'embedding': vector,
                }
                for row, description, vector in zip(
                    rows, descriptions, vectors, strict=True
                )
            ],
            session_factory=self._session_factory,
        )
        return {'procedures_enriched': len(rows)}


async def _ordered_results(rows: list[dict], callback: Callable) -> list:
    """Runs callbacks concurrently while returning results in row order."""
    return list(await asyncio.gather(*(callback(row) for row in rows)))
