import dspy
from pydantic import BaseModel, Field

from kms.core import models, module
from kms.graph import local_procedure_hubs, queries, writer

from . import builder


class GlobalProcedureHubResult(BaseModel):
    canonical_name: str = Field(
        description='A concise name for the shared method.'
    )
    description: str = Field(
        description='A standalone description supported by all supplied local hubs.'
    )


class GlobalProcedureHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one reusable global canonical method from local procedure hubs
    that share a reasoning or action pattern. Preserve only supported content.
    Do not invent steps or merge merely related methods.
    """

    request: models.EvidenceInput = dspy.InputField()
    result: GlobalProcedureHubResult = dspy.OutputField()


class GlobalProcedureHubAdjudicationSignature(dspy.Signature):
    r"""
    Decide whether two local procedure hubs use the same reusable method across
    sources. Return False for merely related methods or shared results.
    """

    pair: models.TextPairInput = dspy.InputField()
    result: models.MergeDecision = dspy.OutputField()


class GlobalProcedureHubSynthesizer(module.Module):
    signature = GlobalProcedureHubSynthesisSignature
    record_name = 'global_procedure_hub_synthesizer'

    def encode(self, evidence: list[str]) -> dict:
        return {'request': models.EvidenceInput(evidence=evidence)}

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = GlobalProcedureHubResult.model_validate(prediction.result)
        return (
            module.require_text(result.canonical_name, 'canonical_name'),
            module.require_text(result.description, 'description'),
        )


class GlobalProcedureHubAdjudicator(module.Module):
    signature = GlobalProcedureHubAdjudicationSignature
    record_name = 'global_procedure_hub_adjudicator'

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
    adjudicator: GlobalProcedureHubAdjudicator,
    synthesizer: GlobalProcedureHubSynthesizer,
) -> dict:
    """Rebuild global procedure hubs from persisted local procedure hubs."""
    records = _records(await queries.all_source_procedure_hubs(session_factory))
    result = await builder.build(
        records,
        stage_name='procedure_hubs',
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    for hub in result['hubs']:
        hub['uuid'] = local_procedure_hubs.global_hub_uuid(
            hub['members']
        )
    await writer.clear_global_procedure_hubs(session_factory=session_factory)
    await writer.persist_global_procedure_hubs(
        result['hubs'], session_factory=session_factory
    )
    return {'global_hubs': len(result['hubs']), 'local_hubs': result['records']}
