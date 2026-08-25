"""Command-line entry point for disposable graph maintenance."""

import argparse
import asyncio
import json

from kms import runtime
from kms.construction import (
    entity_hubs,
    local_procedure_hubs,
    local_statement_hubs,
    maintenance,
    predicate_hubs,
)
from kms.core import llm
from kms.graph import maintenance as graph_maintenance


def _parser() -> argparse.ArgumentParser:
    """Build the maintenance command parser."""
    parser = argparse.ArgumentParser(
        description='Rebuild disposable KMS graph knowledge.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    source = subparsers.add_parser(
        'rebuild-source',
        help='Rebuild source-local hubs from durable records.',
    )
    source.add_argument('source')
    meta = subparsers.add_parser(
        'rebuild-meta',
        help='Rebuild cross-source entity and predicate hubs.',
    )
    meta.add_argument(
        '--learning',
        action='store_true',
        help='Also rebuild global statement and procedure hubs.',
    )
    meta.add_argument(
        '--statements',
        action='store_true',
        help='Rebuild the GlobalStatementHub layer.',
    )
    meta.add_argument(
        '--procedures',
        action='store_true',
        help='Rebuild the GlobalProcedureHub layer.',
    )
    subparsers.add_parser(
        'rebuild-embeddings',
        help='Re-embed all persisted vector-indexed graph records.',
    )
    return parser


def _models() -> dict[str, object]:
    """Build the model objects required by maintenance operations."""
    entity_language_model = llm.module_lm('entity_hub_builder')
    predicate_language_model = llm.module_lm('predicate_hub_builder')
    statement_language_model = llm.module_lm('statement_hub_builder')
    procedure_language_model = llm.module_lm('procedure_hub_builder')
    return {
        'entity_language_model': entity_language_model,
        'entity_adjudicator': entity_hubs.EntityHubAdjudicator(
            language_model=entity_language_model
        ),
        'entity_synthesizer': entity_hubs.EntityHubSynthesizer(
            language_model=entity_language_model
        ),
        'predicate_language_model': predicate_language_model,
        'predicate_adjudicator': predicate_hubs.PredicateHubAdjudicator(
            language_model=predicate_language_model
        ),
        'predicate_synthesizer': predicate_hubs.PredicateHubSynthesizer(
            language_model=predicate_language_model
        ),
        'statement_adjudicator': local_statement_hubs.LocalStatementHubAdjudicator(
            language_model=statement_language_model
        ),
        'statement_synthesizer': local_statement_hubs.LocalStatementHubSynthesizer(
            language_model=statement_language_model
        ),
        'procedure_adjudicator': local_procedure_hubs.LocalProcedureHubAdjudicator(
            language_model=procedure_language_model
        ),
        'procedure_synthesizer': local_procedure_hubs.LocalProcedureHubSynthesizer(
            language_model=procedure_language_model
        ),
    }
async def _run(arguments: argparse.Namespace) -> dict:
    """Run the selected maintenance operation in a shared runtime."""
    async with runtime.Runtime() as application:
        session_factory = application.session_factory()
        if not session_factory:
            raise RuntimeError('Neo4j is not configured')
        if arguments.command == 'rebuild-embeddings':
            return await graph_maintenance.rebuild_embeddings(
                session_factory=session_factory
            )

        models = _models()
        if arguments.command == 'rebuild-meta':
            return await maintenance.rebuild_meta(
                session_factory=session_factory,
                entity_language_model=models['entity_language_model'],
                entity_adjudicator=models['entity_adjudicator'],
                entity_synthesizer=models['entity_synthesizer'],
                predicate_language_model=models['predicate_language_model'],
                predicate_adjudicator=models['predicate_adjudicator'],
                predicate_synthesizer=models['predicate_synthesizer'],
                statement_adjudicator=models['statement_adjudicator'],
                statement_synthesizer=models['statement_synthesizer'],
                procedure_adjudicator=models['procedure_adjudicator'],
                procedure_synthesizer=models['procedure_synthesizer'],
                include_statement_learning=(
                    arguments.learning or arguments.statements
                ),
                include_procedure_learning=(
                    arguments.learning or arguments.procedures
                ),
            )
        return await maintenance.rebuild_source(
            arguments.source,
            session_factory=session_factory,
            **models,
        )


def run() -> None:
    """Parse arguments and print maintenance counts as JSON."""
    arguments = _parser().parse_args()
    print(json.dumps(asyncio.run(_run(arguments)), indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
