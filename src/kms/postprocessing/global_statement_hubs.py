from kms.construction import local_statement_hubs
from kms.graph import queries, writer

GlobalStatementHubAdjudicator = local_statement_hubs.LocalStatementHubAdjudicator
GlobalStatementHubSynthesizer = local_statement_hubs.LocalStatementHubSynthesizer


async def rebuild(
    *,
    session_factory,
    adjudicator: GlobalStatementHubAdjudicator,
    synthesizer: GlobalStatementHubSynthesizer,
) -> dict:
    """Rebuild global statement hubs from persisted local statement hubs."""
    rows = await queries.all_source_statement_hubs(session_factory)
    typed = local_statement_hubs._records(rows)
    result = await local_statement_hubs._build(
        '', typed, adjudicator=adjudicator, synthesizer=synthesizer, meta=True
    )
    await writer.clear_global_statement_hubs(session_factory=session_factory)
    await writer.persist_global_statement_hubs(
        result['hubs'], session_factory=session_factory
    )
    return {'global_hubs': len(result['hubs']), 'local_hubs': result['records']}
