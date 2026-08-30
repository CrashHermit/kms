import asyncio
from types import SimpleNamespace

from kms.construction import procedure_enrichment, statement_enrichment
from kms.core import models
from kms.postprocessing.learning import procedures as procedure_generation


def test_statement_enricher_emits_the_canonical_statement_field():
    prediction = SimpleNamespace(statement='Complete statement.')

    assert statement_enrichment.StatementEnricher.decode(
        None, prediction, source_content=[], canonical_knowledge='knowledge'
    ) == 'Complete statement.'


def test_source_procedure_writer_encodes_source_text_nodes():
    statement = [
        models.TextNodeInput(
            local_index=0, node_type='paragraph', node_text='Prove this.'
        )
    ]
    source_procedure = [
        models.TextNodeInput(
            local_index=1, node_type='paragraph', node_text='Apply the theorem.'
        )
    ]
    writer_input = procedure_enrichment.SourceProcedureWriter.encode(
        object(), statement, source_procedure, 'knowledge'
    )

    assert writer_input['statement'] == statement
    assert writer_input['source_procedure'] == source_procedure


def test_source_procedure_enricher_skips_missing_source_procedure():
    enricher = object.__new__(procedure_enrichment.ProcedureEnricher)
    calls = []

    class Writer:
        async def aforward(self, **kwargs):
            calls.append(kwargs)
            return 'should not be called'

    enricher.writer = Writer()
    item = models.ProcedureEnrichmentInput(
        source='book',
        statement_uuid='statement-1',
        statement=(
            models.TextNodeInput(
                local_index=0, node_type='paragraph', node_text='Define a graph.'
            ),
        ),
    )

    compiled, procedure = asyncio.run(enricher.compile(item))

    assert (compiled, procedure) == (False, None)
    assert calls == []


def test_source_procedure_enricher_compiles_existing_source_procedure():
    enricher = object.__new__(procedure_enrichment.ProcedureEnricher)

    class Writer:
        async def aforward(self, **kwargs):
            return 'Canonical source procedure.'

    enricher.writer = Writer()
    item = models.ProcedureEnrichmentInput(
        source='book',
        statement_uuid='statement-1',
        procedure_uuid='procedure-1',
        statement=(
            models.TextNodeInput(
                local_index=0, node_type='paragraph', node_text='Prove the claim.'
            ),
        ),
        procedure=(
            models.TextNodeInput(
                local_index=1, node_type='paragraph', node_text='Proof.'
            ),
        ),
        canonical_knowledge='definition',
    )

    compiled, procedure = asyncio.run(enricher.compile(item))

    assert compiled is True
    assert procedure == 'Canonical source procedure.'


def test_solution_router_decodes_a_boolean_gate():
    statement = [
        models.TextNodeInput(
            local_index=0, node_type='paragraph', node_text='Solve this.'
        )
    ]

    assert procedure_generation.ProcedureNeedRouter.decode(
        None, SimpleNamespace(needs_procedure=True), statement=statement
    ) is True
    assert procedure_generation.ProcedureNeedRouter.decode(
        None, SimpleNamespace(needs_procedure=False), statement=statement
    ) is False


def test_generated_solution_has_generated_kind():
    generator = object.__new__(procedure_generation.SolutionProcedureGenerator)

    class Router:
        async def aforward(self, **kwargs):
            return True

    class Writer:
        async def aforward(self, **kwargs):
            return 'Generated solution.'

    generator.router = Router()
    generator.writer = Writer()
    item = models.ProcedureEnrichmentInput(
        source='book',
        statement_uuid='statement-1',
        statement=(
            models.TextNodeInput(
                local_index=0, node_type='paragraph', node_text='Prove the claim.'
            ),
        ),
        canonical_knowledge='definition',
    )

    procedure = asyncio.run(generator.generate(item))

    assert procedure is not None
    assert procedure.kind is models.ProcedureKind.GENERATED
    assert procedure.procedure == 'Generated solution.'
