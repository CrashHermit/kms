import dspy
from pydantic import BaseModel, Field

from kms.construction import learning_hub_builder
from kms.core import module
from kms.graph import queries, writer


class StatementHubResult(BaseModel):
    canonical_name: str = Field(
        description='A concise name for the shared statement.'
    )
    description: str = Field(
        description=(
            'A standalone learner-facing description of the shared statement.'
        )
    )


class StatementHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one reusable source-local learning target from statements that
    express the same claim, fact, theorem, explanation, or question pattern.

    The supplied descriptions are already enriched source-level statements.
    Preserve the supported meaning, qualifiers, and mathematical notation.
    Generalize across the supplied statements only where they share meaning.
    Do not mention the source, passages, statement identifiers, or procedures.
    Do not solve a question or invent facts.
    """

    evidence: list[str] = dspy.InputField(
        description='Descriptions of statements assigned to one local hub.'
    )
    result: StatementHubResult = dspy.OutputField(
        description='A reusable name and learner-facing statement description.'
    )


class StatementHubAdjudicationSignature(dspy.Signature):
    r"""
    Decide whether two enriched statements express the same reusable learning
    target within one source.

    Return True only when they communicate the same claim, fact, theorem,
    explanation, or question pattern with equivalent meaning. Return False for
    merely related, sequential, broader, narrower, or differently solved
    statements. Ignore whether either statement has a procedure.
    """

    left: str = dspy.InputField(description='The first enriched statement.')
    right: str = dspy.InputField(description='The second enriched statement.')
    should_merge: bool = dspy.OutputField(
        description='Whether both statements belong to one local StatementHub.'
    )


class StatementHubSynthesizer(module.Module):
    signature = StatementHubSynthesisSignature
    record_name = 'statement_hub_synthesizer'

    def encode(self, evidence: list[str]) -> dict:
        return {'evidence': evidence}

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = prediction.result
        return result.canonical_name, result.description


class StatementHubAdjudicator(module.Module):
    signature = StatementHubAdjudicationSignature
    record_name = 'statement_hub_adjudicator'

    def encode(self, left: str, right: str) -> dict:
        return {'left': left, 'right': right}

    def decode(self, prediction, **inputs) -> bool:
        return prediction.should_merge


class StatementHubNode:
    def __init__(
        self,
        session_factory,
        adjudicator: StatementHubAdjudicator,
        synthesizer: StatementHubSynthesizer,
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
            'statement',
            source,
        )
        result = await learning_hub_builder.build_source_hubs(
            'statement',
            source,
            records,
            adjudicator=self._adjudicator,
            synthesizer=self._synthesizer,
        )
        await writer.clear_learning_hubs(
            'statement', source, session_factory=self._session_factory
        )
        await writer.persist_learning_hubs(
            'statement', result['hubs'], session_factory=self._session_factory
        )
        return {
            'statement_hubs_created': len(result['hubs']),
            'statements_clustered': result['records'],
        }


async def rebuild(
    source: str,
    *,
    session_factory,
    adjudicator: StatementHubAdjudicator,
    synthesizer: StatementHubSynthesizer,
) -> dict:
    records = await queries.learning_hub_items(
        session_factory,
        'statement',
        source,
    )
    result = await learning_hub_builder.build_source_hubs(
        'statement',
        source,
        records,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    await writer.clear_learning_hubs(
        'statement', source, session_factory=session_factory
    )
    await writer.persist_learning_hubs(
        'statement', result['hubs'], session_factory=session_factory
    )
    return {
        'statement_hubs': len(result['hubs']),
        'statements': result['records'],
    }


async def rebuild_meta(
    *,
    session_factory,
    adjudicator: StatementHubAdjudicator,
    synthesizer: StatementHubSynthesizer,
) -> dict:
    """Rebuild cross-source MetaStatementHub records."""
    return await learning_hub_builder.rebuild_meta_hubs(
        'statement',
        session_factory=session_factory,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
