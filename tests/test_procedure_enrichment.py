import asyncio
from types import SimpleNamespace

import dspy

from kms.construction import procedure_enrichment, statement_enrichment
from kms.core import content, models


def test_statement_enricher_emits_the_canonical_statement_field():
    prediction = SimpleNamespace(statement='Complete statement.')

    assert statement_enrichment.StatementEnricher.decode(
        None, prediction, source_content=content.Content.from_text('source'),
        canonical_knowledge='knowledge'
    ) == 'Complete statement.'


def test_procedure_need_router_decodes_a_boolean_gate():
    statement = content.Content.from_text('Solve this.')
    assert procedure_enrichment.ProcedureNeedRouter.decode(
        None,
        SimpleNamespace(needs_procedure=True),
        statement=statement,
    ) is True
    assert procedure_enrichment.ProcedureNeedRouter.decode(
        None,
        SimpleNamespace(needs_procedure=False),
        statement=statement,
    ) is False


def test_procedure_router_and_writer_encode_multimodal_statement():
    image = dspy.Image(url='data:image/png;base64,AAAA')
    statement = content.Content.from_parts(['Use this diagram.', image])

    router_input = procedure_enrichment.ProcedureNeedRouter.encode(
        object(), statement
    )
    writer_input = procedure_enrichment.ProcedureWriter.encode(
        object(), statement, None, 'knowledge'
    )

    assert isinstance(router_input['statement'], content.ContentParts)
    assert isinstance(writer_input['statement'], content.ContentParts)
    assert len(writer_input['statement'].content.parts) == 2


def test_procedure_enricher_does_not_write_when_router_rejects():
    enricher = object.__new__(procedure_enrichment.ProcedureEnricher)
    calls = []

    class Router:
        async def aforward(self, **kwargs):
            calls.append(('router', kwargs))
            return False

    class Writer:
        async def aforward(self, **kwargs):
            calls.append(('writer', kwargs))
            return 'should not be called'

    enricher.router = Router()
    enricher.writer = Writer()
    item = models.ProcedureEnrichmentInput(
        source='book',
        statement_uuid='statement-1',
        statement=content.Content.from_text('Define a graph.'),
    )

    needed, procedure = asyncio.run(enricher.compile(item))

    assert (needed, procedure) == (False, None)
    assert [name for name, _ in calls] == ['router']


def test_procedure_enricher_writes_existing_and_generated_procedures():
    enricher = object.__new__(procedure_enrichment.ProcedureEnricher)

    class Router:
        async def aforward(self, **kwargs):
            return True

    class Writer:
        async def aforward(self, **kwargs):
            return 'Complete procedure.'

    enricher.router = Router()
    enricher.writer = Writer()
    item = models.ProcedureEnrichmentInput(
        source='book',
        statement_uuid='statement-1',
        statement=content.Content.from_text('Prove the claim.'),
        canonical_knowledge='definition',
    )

    needed, procedure = asyncio.run(enricher.compile(item))

    assert needed is True
    assert procedure == 'Complete procedure.'
