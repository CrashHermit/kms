from types import SimpleNamespace

import asyncio

from kms.construction import statement_enrichment, procedure_enrichment
from kms.core import content, models


def test_statement_enricher_emits_the_canonical_statement_field():
    prediction = SimpleNamespace(statement='Complete statement.')

    assert statement_enrichment.StatementEnricher.decode(
        None, prediction, source_content=content.Content.from_text('source'),
        canonical_knowledge='knowledge'
    ) == 'Complete statement.'


def test_procedure_need_router_decodes_a_boolean_gate():
    assert procedure_enrichment.ProcedureNeedRouter.decode(
        None, SimpleNamespace(needs_procedure=True), statement='Solve this.'
    ) is True
    assert procedure_enrichment.ProcedureNeedRouter.decode(
        None, SimpleNamespace(needs_procedure=False), statement='Define this.'
    ) is False


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
