import dspy
from pydantic import BaseModel, Field

from kms.construction import hub_builder, name_hubs
from kms.core import content, module


class PredicateHubDefinition(BaseModel):
    canonical_name: str = Field(
        description='The canonical name for the predicate relation.'
    )
    description: str = Field(
        description='A standalone learner-facing predicate description.'
    )


class PredicateHubAdjudication(BaseModel):
    decision: str = Field(
        description='Merge, Hierarchy, or Separate for predicate mentions.'
    )
    more_general: str = Field(
        default='none',
        description='left or right when the decision is Hierarchy.',
    )


class PredicateHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one source-local learner-facing predicate concept from supplied
    relation phrases and their passage-grounded descriptions. Generalize only
    what the evidence supports. Do not mention the source or invent facts.
    """

    surface_forms: list[str] = dspy.InputField()
    descriptions: list[str] = dspy.InputField()
    scope: str = dspy.InputField()
    result: PredicateHubDefinition = dspy.OutputField()


class PredicateHubAdjudicationSignature(dspy.Signature):
    r"""
    Compare two predicate mentions. Return Merge for the same relation,
    Hierarchy only for a strict kind-of relation, and Separate otherwise.
    """

    left: content.ContentParts = dspy.InputField()
    right: content.ContentParts = dspy.InputField()
    scope: str = dspy.InputField()
    result: PredicateHubAdjudication = dspy.OutputField()


class PredicateHubSynthesizer(module.Module):
    signature = PredicateHubSynthesisSignature
    record_name = 'predicate_hub_synthesizer'

    def encode(
        self,
        surface_forms: list[str],
        descriptions: list[str],
        scope: str,
    ) -> dict:
        return {
            'surface_forms': surface_forms,
            'descriptions': descriptions,
            'scope': scope,
        }

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = prediction.result
        return result.canonical_name, result.description


class PredicateHubAdjudicator(module.Module):
    signature = PredicateHubAdjudicationSignature
    record_name = 'predicate_hub_adjudicator'

    def encode(
        self,
        left: content.Content,
        right: content.Content,
        scope: str,
    ) -> dict:
        return {
            'left': content.ContentParts(content=left),
            'right': content.ContentParts(content=right),
            'scope': scope,
        }

    def decode(self, prediction, **inputs) -> PredicateHubAdjudication:
        return prediction.result


class PredicateHubNode:
    def __init__(
        self,
        session_factory,
        adjudicator,
        synthesizer,
        name_language_model,
    ) -> None:
        self._session_factory = session_factory
        self._adjudicator = adjudicator
        self._synthesizer = synthesizer
        self._name_language_model = name_language_model

    async def run(self, current_state: dict) -> dict:
        if not self._session_factory:
            return {}
        source = current_state.get('source')
        if not source:
            return {}
        result = await hub_builder.assign_source_hubs(
            'predicate',
            source,
            session_factory=self._session_factory,
            adjudicator=self._adjudicator,
            synthesizer=self._synthesizer,
        )
        lexical = await name_hubs.rebuild(
            'predicate',
            source,
            language_model=self._name_language_model,
            session_factory=self._session_factory,
        )
        return {
            'predicate_assigned': result['assigned'],
            'predicate_hubs_created': result['new_hubs'],
            'predicate_name_hubs_created': lexical['name_hubs'],
        }


async def rebuild(
    source: str,
    *,
    adjudicator,
    synthesizer,
    name_language_model,
    session_factory,
) -> dict:
    return await PredicateHubNode(
        session_factory, adjudicator, synthesizer, name_language_model
    ).run({'source': source})


async def rebuild_source(
    source: str,
    *,
    language_model,
    adjudicator,
    synthesizer,
    session_factory,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild source-local predicate hubs from persisted components."""
    return await hub_builder.rebuild_hubs(
        'predicate',
        session_factory=session_factory,
        source=source,
        language_model=language_model,
        max_concurrency=max_concurrency,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )


async def rebuild_meta(
    *,
    language_model,
    adjudicator,
    synthesizer,
    session_factory,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild cross-source predicate hubs from source-local hubs."""
    return await hub_builder.rebuild_meta_hubs(
        'predicate',
        language_model=language_model,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
