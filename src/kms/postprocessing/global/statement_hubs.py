import dspy
from pydantic import BaseModel, Field

from kms.core import models, module
from kms.graph import local_statement_hubs, queries, writer

from . import builder


class GlobalStatementHubResult(BaseModel):
    canonical_name: str = Field(
        description='A concise name for the shared statement.'
    )
    description: str = Field(
        description='A standalone description supported by all supplied local hubs.'
    )


class GlobalStatementHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one reusable global canonical statement from local statement
    hubs that express the same meaning across multiple sources. Preserve shared
    meaning and qualifiers. Do not invent facts or resolve questions.
    """

    request: models.EvidenceInput = dspy.InputField()
    result: GlobalStatementHubResult = dspy.OutputField()


class GlobalStatementHubAdjudicationSignature(dspy.Signature):
    r"""
    Decide whether two local statement hubs express the same canonical meaning
    across sources. Return False for merely related, broader, narrower, or
    differently qualified statements.
    """

    pair: models.TextPairInput = dspy.InputField()
    result: models.MergeDecision = dspy.OutputField()


class GlobalStatementHubSynthesizer(module.Module):
    signature = GlobalStatementHubSynthesisSignature
    record_name = 'global_statement_hub_synthesizer'

    def encode(self, evidence: list[str]) -> dict:
        return {'request': models.EvidenceInput(evidence=evidence)}

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = GlobalStatementHubResult.model_validate(prediction.result)
        return (
            module.require_text(result.canonical_name, 'canonical_name'),
            module.require_text(result.description, 'description'),
        )


class GlobalStatementHubAdjudicator(module.Module):
    signature = GlobalStatementHubAdjudicationSignature
    record_name = 'global_statement_hub_adjudicator'

    def encode(self, left: str, right: str) -> dict:
        return {'pair': models.TextPairInput(left=left, right=right)}

    def decode(self, prediction, **inputs) -> bool:
        result = models.MergeDecision.model_validate(prediction.result)
        return module.require_bool(result.should_merge, 'should_merge')


def _records(rows: list[dict]) -> tuple[models.HubRecord, ...]:
    return tuple(
        models.HubRecord(
            uuid=row['uuid'],
            name=row.get('name', row['uuid']),
            description=row.get('description'),
            embedding=list(row['embedding']),
            source=row['source'],
        )
        for row in rows
    )


async def rebuild(
    *,
    session_factory,
    adjudicator: GlobalStatementHubAdjudicator,
    synthesizer: GlobalStatementHubSynthesizer,
) -> dict:
    """Rebuild global statement hubs from persisted local statement hubs."""
    records = _records(await queries.all_source_statement_hubs(session_factory))
    result = await builder.build(
        records,
        stage_name='statement_hubs',
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    for hub in result['hubs']:
        hub['uuid'] = local_statement_hubs.global_hub_uuid(
            hub['members']
        )
    await writer.clear_global_statement_hubs(session_factory=session_factory)
    await writer.persist_global_statement_hubs(
        result['hubs'], session_factory=session_factory
    )
    return {'global_hubs': len(result['hubs']), 'local_hubs': result['records']}
