from kms.construction import local_procedure_hubs
from kms.graph import queries, writer

GlobalProcedureHubAdjudicator = local_procedure_hubs.LocalProcedureHubAdjudicator
GlobalProcedureHubSynthesizer = local_procedure_hubs.LocalProcedureHubSynthesizer


async def rebuild(
    *,
    session_factory,
    adjudicator: GlobalProcedureHubAdjudicator,
    synthesizer: GlobalProcedureHubSynthesizer,
) -> dict:
    """Rebuild global procedure hubs from persisted local procedure hubs."""
    rows = await queries.all_source_procedure_hubs(session_factory)
    typed = local_procedure_hubs._records(rows)
    result = await local_procedure_hubs._build(
        '', typed, adjudicator=adjudicator, synthesizer=synthesizer, meta=True
    )
    await writer.clear_global_procedure_hubs(session_factory=session_factory)
    await writer.persist_global_procedure_hubs(
        result['hubs'], session_factory=session_factory
    )
    return {'global_hubs': len(result['hubs']), 'local_hubs': result['records']}
