"""Build source-local and cross-source EventHub concepts."""

import dspy
from pydantic import BaseModel

from kms.construction import local_entity_hubs
from kms.core import llm, models, module
from kms.graph import local_event_hubs as graph_event_hubs
from kms.graph import queries, writer


class EventHubDefinition(BaseModel):
    """Canonical event concept synthesized from event evidence."""

    canonical_name: str
    description: str


class EventHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one canonical event concept from source-grounded event
    mentions that have already been grouped as equivalent.

    Name the occurrence or event itself, not its consequence, motivation,
    property, result, or surrounding explanatory clause. Preserve the
    event's participants and role meaning in the description. Do not merge
    a battle with its victory, an appointment with the person appointed, or
    an event with a later consequence. Generalize only the common event
    supported by every supplied mention. Never invent time, participants,
    outcomes, or causal claims.
    """

    request: models.HubSynthesisInput = dspy.InputField()
    result: EventHubDefinition = dspy.OutputField()


class EventHubAdjudicationSignature(dspy.Signature):
    r"""
    Decide whether two source mentions refer to the same event concept.

    Return TRUE only when both mentions describe the same kind of occurrence
    with compatible trigger/meaning and compatible participants or roles.
    Return FALSE when one is a consequence, result, motivation, condition,
    property, state, or explanation of the other. Return FALSE when one event
    is merely related to, precedes, follows, causes, or is caused by the
    other. In particular, a battle and its victory are distinct events, as
    are an approach and the turning point it produces. Shared verbs or
    topical context are insufficient. Do not use outside knowledge.
    """
    comparison: models.HubMentionComparisonInput = dspy.InputField()
    result: models.MergeDecision = dspy.OutputField()


class EventHubSynthesizer(module.Module):
    """Event-tailored canonical event synthesis."""

    signature = EventHubSynthesisSignature
    record_name = 'event_hub_synthesizer'

    def __init__(self, language_model: dspy.LM | None = None, recorder=None) -> None:
        module.Module.__init__(
            self, language_model or llm.module_lm(self.record_name), recorder=recorder
        )

    def encode(
        self,
        surface_forms: list[str],
        descriptions: list[str],
        scope: str,
    ) -> dict:
        return {
            'request': models.HubSynthesisInput(
                surface_forms=surface_forms,
                descriptions=descriptions,
                scope=scope,
            )
        }

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = EventHubDefinition.model_validate(prediction.result)
        return (
            module.require_text(result.canonical_name, 'canonical_name'),
            module.require_text(result.description, 'description'),
        )
class EventHubAdjudicator(module.Module):
    """Event-tailored event equivalence adjudication."""

    signature = EventHubAdjudicationSignature
    record_name = 'event_hub_adjudicator'

    def __init__(self, language_model: dspy.LM | None = None, recorder=None) -> None:
        module.Module.__init__(
            self, language_model or llm.module_lm(self.record_name), recorder=recorder
        )

    def encode(
        self,
        left: models.HubMentionInput,
        right: models.HubMentionInput,
        scope: str,
    ) -> dict:
        return {
            'comparison': models.HubMentionComparisonInput(
                left=left, right=right, scope=scope
            )
        }

    def decode(self, prediction, **inputs) -> bool:
        result = models.MergeDecision.model_validate(prediction.result)
        return module.require_bool(result.should_merge, 'should_merge')

async def rebuild(
    source: str,
    *,
    session_factory,
    language_model,
    adjudicator,
    synthesizer,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild local EventHubs from persisted Event components."""
    spec = local_entity_hubs.EntityHubSpec(
        graph=graph_event_hubs,
        stage_name='event_hubs',
        hub_id_factory=local_entity_hubs.source_hub_id_factory(graph_event_hubs),
        source_resolver=lambda records: source,
        adjudication_context='Compare event mentions by event meaning, trigger, and roles.',
        synthesis_context='Synthesize a source-local semantic event hub.',
        all_components=queries.all_event_components,
        clear_hubs=writer.clear_event_hubs,
        persist_hubs=writer.persist_event_hubs,
        rebuild_names=lambda *args, **kwargs: None,
        rebuild_triplets=lambda *args, **kwargs: None,
    )
    records = await queries.all_event_components(session_factory, source)
    result = await local_entity_hubs.build_hubs(
        records,
        spec=spec,
        max_concurrency=max_concurrency,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
    )
    await writer.clear_event_hubs(source, session_factory=session_factory)
    await writer.persist_event_hubs(
        result['hubs'], session_factory=session_factory, tier='source'
    )
    return {'clusters': result['clusters'], 'records': result['records']}
