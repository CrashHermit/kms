import dspy
from pydantic import BaseModel, Field

from kms.construction import learning_hub_builder
from kms.core import module
from kms.graph import queries, writer


class ProcedureHubResult(BaseModel):
    canonical_name: str = Field(
        description='A concise name for the shared method.'
    )
    description: str = Field(
        description=(
            'A standalone learner-facing description of the shared method.'
        )
    )


class ProcedureHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one reusable source-local learning method from procedures that
    use the same general reasoning or action pattern.

    The supplied descriptions are enriched procedures. Explain the shared
    goal and method without copying source-specific names, values, answers, or
    navigation. Preserve supported conditions and mathematical notation. Do
    not invent steps or collapse merely related but different methods.
    """

    evidence: list[str] = dspy.InputField(
        description='Descriptions of procedures assigned to one local hub.'
    )
    result: ProcedureHubResult = dspy.OutputField(
        description='A reusable name and learner-facing method description.'
    )


class ProcedureHubAdjudicationSignature(dspy.Signature):
    r"""
    Decide whether two enriched procedures use the same reusable method within
    one source.

    Return True only when their general reasoning or action pattern is the
    same. Return False when they merely concern the same subject, share a
    result, or use materially different methods.
    """

    left: str = dspy.InputField(description='The first enriched procedure.')
    right: str = dspy.InputField(description='The second enriched procedure.')
    should_merge: bool = dspy.OutputField(
        description='Whether both procedures belong to one local ProcedureHub.'
    )


class ProcedureHubSynthesizer(module.Module):
    signature = ProcedureHubSynthesisSignature
    record_name = 'procedure_hub_synthesizer'

    def encode(self, evidence: list[str]) -> dict:
        return {'evidence': evidence}

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = prediction.result
        return result.canonical_name, result.description


class ProcedureHubAdjudicator(module.Module):
    signature = ProcedureHubAdjudicationSignature
    record_name = 'procedure_hub_adjudicator'

    def encode(self, left: str, right: str) -> dict:
        return {'left': left, 'right': right}

    def decode(self, prediction, **inputs) -> bool:
        return prediction.should_merge


class ProcedureHubNode:
    def __init__(
        self,
        session_factory,
        adjudicator: ProcedureHubAdjudicator,
        synthesizer: ProcedureHubSynthesizer,
    ) -> None:
        self._session_factory = session_factory
        self._adjudicator = adjudicator
        self._synthesizer = synthesizer

    async def run(self, current_state: dict) -> dict:
        if not self._session_factory:
            return {}
        source = current_state.get('source')
        if not source:
            return {}
        records = await queries.learning_hub_items(
            self._session_factory,
            'procedure',
            source,
        )
        result = await learning_hub_builder.build_source_hubs(
            'procedure',
            source,
            records,
            adjudicator=self._adjudicator,
            synthesizer=self._synthesizer,
        )
        await writer.clear_learning_hubs(
            'procedure', source, session_factory=self._session_factory
        )
        await writer.persist_learning_hubs(
            'procedure', result['hubs'], session_factory=self._session_factory
        )
        return {
            'procedure_hubs_created': len(result['hubs']),
            'procedures_clustered': result['records'],
        }


async def rebuild(
    source: str,
    *,
    session_factory,
    adjudicator: ProcedureHubAdjudicator,
    synthesizer: ProcedureHubSynthesizer,
) -> dict:
    records = await queries.learning_hub_items(
        session_factory,
        'procedure',
        source,
    )
    result = await learning_hub_builder.build_source_hubs(
        'procedure',
        source,
        records,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    await writer.clear_learning_hubs(
        'procedure', source, session_factory=session_factory
    )
    await writer.persist_learning_hubs(
        'procedure', result['hubs'], session_factory=session_factory
    )
    return {
        'procedure_hubs': len(result['hubs']),
        'procedures': result['records'],
    }


async def rebuild_meta(
    *,
    session_factory,
    adjudicator: ProcedureHubAdjudicator,
    synthesizer: ProcedureHubSynthesizer,
) -> dict:
    """Rebuild cross-source MetaProcedureHub records."""
    return await learning_hub_builder.rebuild_meta_hubs(
        'procedure',
        session_factory=session_factory,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
