"""Command-line entry point for disposable graph maintenance."""

import argparse
import asyncio
import json
from importlib import import_module

from kms import runtime
from kms.construction import (
    local_entity_hubs,
    local_event_hubs,
    local_predicate_hubs,
    local_procedure_hubs,
    local_statement_hubs,
    maintenance,
)
from kms.core import llm, serve
from kms.core import models as core_models
from kms.graph import maintenance as graph_maintenance
from kms.postprocessing.learning import (
    cards,
    entity_cards,
    event_cards,
    procedure_cards,
    triplet_cards,
)

global_maintenance = import_module('kms.postprocessing.global.maintenance')
global_statement_hubs = import_module(
    'kms.postprocessing.global.statement_hubs'
)
global_procedure_hubs = import_module(
    'kms.postprocessing.global.procedure_hubs'
)
global_event_hubs = import_module('kms.postprocessing.global.event_hubs')
global_predicate_hubs = import_module(
    'kms.postprocessing.global.predicate_hubs'
)


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
    for command in ('entity', 'event', 'triplet', 'procedure'):
        card = subparsers.add_parser(
            f'generate-{command}-cards',
            help=f'Generate {command} component-owned cards for one local hub.',
        )
        card.add_argument('hub_uuid')
    subparsers.add_parser(
        'rebuild-embeddings',
        help='Re-embed all persisted vector-indexed graph records.',
    )
    return parser


def _card_designer(
    kind: core_models.CardTargetKind,
) -> cards.ComponentCardDesigner:
    """Build the card designer for one component type."""
    modules = {
        core_models.CardTargetKind.ENTITY: (
            entity_cards.EntityCardSuitability,
            entity_cards.EntityCardGenerator,
            entity_cards.EntityCardVerifier,
            (
                'entity_card_suitability',
                'entity_card_generator',
                'entity_card_verifier',
            ),
        ),
        core_models.CardTargetKind.EVENT: (
            event_cards.EventCardSuitability,
            event_cards.EventCardGenerator,
            event_cards.EventCardVerifier,
            (
                'event_card_suitability',
                'event_card_generator',
                'event_card_verifier',
            ),
        ),
        core_models.CardTargetKind.TRIPLET: (
            triplet_cards.TripletCardSuitability,
            triplet_cards.TripletCardGenerator,
            triplet_cards.TripletCardVerifier,
            (
                'triplet_card_suitability',
                'triplet_card_generator',
                'triplet_card_verifier',
            ),
        ),
        core_models.CardTargetKind.PROCEDURE: (
            procedure_cards.ProcedureCardSuitability,
            procedure_cards.ProcedureCardGenerator,
            procedure_cards.ProcedureCardVerifier,
            (
                'procedure_card_suitability',
                'procedure_card_generator',
                'procedure_card_verifier',
            ),
        ),
    }
    suitability, generator, verifier, lm_names = modules[kind]
    return {
        core_models.CardTargetKind.ENTITY: entity_cards.EntityCardDesigner,
        core_models.CardTargetKind.EVENT: event_cards.EventCardDesigner,
        core_models.CardTargetKind.TRIPLET: triplet_cards.TripletCardDesigner,
        core_models.CardTargetKind.PROCEDURE: procedure_cards.ProcedureCardDesigner,
    }[kind](
        suitability(llm.module_lm(lm_names[0])),
        generator(llm.module_lm(lm_names[1])),
        verifier(llm.module_lm(lm_names[2])),
    )


def _models() -> dict[str, object]:
    """Build the model objects required by aggregate rebuild operations."""
    entity_language_model = llm.module_lm('entity_hub_builder')
    predicate_language_model = llm.module_lm('predicate_hub_builder')
    statement_language_model = llm.module_lm('statement_hub_builder')
    procedure_language_model = llm.module_lm('procedure_hub_builder')
    event_language_model = llm.module_lm('event_hub_builder')
    triplet_language_model = llm.module_lm('triplet_hub_builder')
    return {
        'entity_language_model': entity_language_model,
        'entity_synthesizer': local_entity_hubs.EntityHubSynthesizer(
            language_model=entity_language_model
        ),
        'event_synthesizer': local_event_hubs.EventHubSynthesizer(
            language_model=event_language_model
        ),
        'global_event_adjudicator': global_event_hubs.EventHubAdjudicator(
            language_model=event_language_model
        ),
        'global_event_synthesizer': global_event_hubs.EventHubSynthesizer(
            language_model=event_language_model
        ),
        'event_language_model': event_language_model,
        'triplet_language_model': triplet_language_model,
        'predicate_synthesizer': local_predicate_hubs.PredicateHubSynthesizer(
            language_model=predicate_language_model
        ),
        'global_predicate_adjudicator': global_predicate_hubs.PredicateHubAdjudicator(
            language_model=predicate_language_model
        ),
        'global_predicate_synthesizer': global_predicate_hubs.PredicateHubSynthesizer(
            language_model=predicate_language_model
        ),
        'predicate_language_model': predicate_language_model,
        'statement_synthesizer': local_statement_hubs.StatementHubSynthesizer(
            language_model=statement_language_model
        ),
        'procedure_synthesizer': local_procedure_hubs.ProcedureHubSynthesizer(
            language_model=procedure_language_model
        ),
        'global_statement_adjudicator': global_statement_hubs.GlobalStatementHubAdjudicator(
            language_model=statement_language_model
        ),
        'global_statement_synthesizer': global_statement_hubs.GlobalStatementHubSynthesizer(
            language_model=statement_language_model
        ),
        'global_procedure_adjudicator': global_procedure_hubs.GlobalProcedureHubAdjudicator(
            language_model=procedure_language_model
        ),
        'global_procedure_synthesizer': global_procedure_hubs.GlobalProcedureHubSynthesizer(
            language_model=procedure_language_model
        ),
    }


async def _generate_cards(
    *,
    session_factory,
    target_kind: core_models.CardTargetKind,
    hub_uuid: str,
    designer: cards.ComponentCardDesigner,
) -> dict:
    """Generate cards for one explicitly selected component kind."""
    return await cards.create_cards_for_hub(
        session_factory=session_factory,
        target_kind=target_kind,
        hub_uuid=hub_uuid,
        designer=designer,
    )


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

        with serve.model_manager_context(application.model_manager):
            card_commands = {
                'generate-entity-cards': core_models.CardTargetKind.ENTITY,
                'generate-event-cards': core_models.CardTargetKind.EVENT,
                'generate-triplet-cards': core_models.CardTargetKind.TRIPLET,
                'generate-procedure-cards': core_models.CardTargetKind.PROCEDURE,
            }
            if arguments.command in card_commands:
                target_kind = card_commands[arguments.command]
                return await _generate_cards(
                    session_factory=session_factory,
                    target_kind=target_kind,
                    hub_uuid=arguments.hub_uuid,
                    designer=_card_designer(target_kind),
                )
            models = _models()
            if arguments.command == 'rebuild-meta':
                return await global_maintenance.rebuild(
                    session_factory=session_factory,
                    entity_language_model=models['entity_language_model'],
                    entity_synthesizer=models['entity_synthesizer'],
                    event_language_model=models['event_language_model'],
                    event_adjudicator=models['global_event_adjudicator'],
                    event_synthesizer=models['global_event_synthesizer'],
                    predicate_language_model=models['predicate_language_model'],
                    predicate_adjudicator=models[
                        'global_predicate_adjudicator'
                    ],
                    predicate_synthesizer=models[
                        'global_predicate_synthesizer'
                    ],
                    statement_adjudicator=models[
                        'global_statement_adjudicator'
                    ],
                    statement_synthesizer=models[
                        'global_statement_synthesizer'
                    ],
                    procedure_adjudicator=models[
                        'global_procedure_adjudicator'
                    ],
                    procedure_synthesizer=models[
                        'global_procedure_synthesizer'
                    ],
                    include_statements=arguments.learning
                    or arguments.statements,
                    include_procedures=arguments.learning
                    or arguments.procedures,
                )
            if arguments.command == 'rebuild-source':
                return await maintenance.rebuild_source(
                    arguments.source,
                    session_factory=session_factory,
                    entity_language_model=models['entity_language_model'],
                    entity_synthesizer=models['entity_synthesizer'],
                    predicate_language_model=models['predicate_language_model'],
                    predicate_synthesizer=models['predicate_synthesizer'],
                    event_synthesizer=models['event_synthesizer'],
                    triplet_language_model=models['triplet_language_model'],
                    statement_synthesizer=models['statement_synthesizer'],
                    procedure_synthesizer=models['procedure_synthesizer'],
                )
            raise ValueError(
                f'unknown maintenance command: {arguments.command}'
            )


def run() -> None:
    """Parse arguments and print maintenance counts as JSON."""
    arguments = _parser().parse_args()
    print(json.dumps(asyncio.run(_run(arguments)), indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
