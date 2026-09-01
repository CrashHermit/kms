import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import (
    identity,
    models,
    module,
    semantic,
    state,
)


class EventDescription(BaseModel):
    term: str = Field(description='The exact input event term.')
    description: str = Field(description='A concise local event description.')


class EventEnrichmentSignature(dspy.Signature):
    r"""
    Describe each event term as the occurrence supported by the supplied
    technical passage.

    Write a concise source-local description of the event itself. Preserve
    participants and their roles when supported by the passage, as well as
    explicit temporal, spatial, duration, ordering, polarity, causal, and
    state-transition qualifiers that describe the event. Do not describe
    merely its result, consequence, motivation, or surrounding explanatory
    clause. Do not infer unsupported qualifiers or facts. Return exactly one
    description per input term in order.

    """

    request: models.TermEnrichmentInput = dspy.InputField(
        description=(
            'The enrichment request. Describe only target_node.text as an '
            'event occurrence. context_before and context_after provide '
            'reference context only.'
        )
    )
    description: str = dspy.OutputField(
        description='One local description of the supplied event occurrence.'
    )


class EventEnricher(module.Module):
    signature = EventEnrichmentSignature
    record_name = 'event_enrichment'

    def encode(self, request: models.TermEnrichmentInput) -> dict[str, object]:
        """Passes the validated enrichment request to the signature."""
        return {'request': request}

    def decode(self, prediction, **inputs) -> list[EventDescription]:
        """Returns one validated description for the singleton event term."""
        description = module.require_text(prediction.description, 'description')
        terms = inputs['request'].terms
        if len(terms) != 1:
            raise ValueError(
                f'event enrichment expects one input term, got {len(terms)}'
            )
        return [EventDescription(term=terms[0], description=description)]


async def enrich(
    nodes,
    triplets,
    enricher: EventEnricher,
) -> dict[int, dict[str, str | None]]:
    """Describes only event endpoints at each triplet evidence position."""
    terms_by_position: dict[int, set[str]] = {}
    for triplet in triplets:
        for position in triplet.evidence_positions:
            if triplet.subject_kind is models.NodeKind.EVENT:
                terms_by_position.setdefault(position, set()).add(
                    triplet.subject
                )
            if triplet.object_kind is models.NodeKind.EVENT:
                terms_by_position.setdefault(position, set()).add(
                    triplet.object
                )
    if not terms_by_position:
        return {}
    stage = config.get_settings().stages.event_enrichment
    return await semantic.describe_terms(
        nodes,
        terms_by_position,
        enricher,
        stage.before_budget,
        stage.after_budget,
        stage.max_concurrent_calls,
    )


class EventEnrichmentNode:
    """Enriches typed event endpoints and prepares event hub components."""

    def __init__(self, enricher: EventEnricher) -> None:
        self._enricher = enricher

    async def run(self, current_state: dict) -> dict:
        bundle = state.to_construction_bundle(current_state)
        triplets = bundle.triplets
        if not triplets:
            return {'construction_bundle': bundle}
        descriptions = await enrich(bundle.nodes, triplets, self._enricher)
        if not descriptions:
            bundle.event_descriptions = {}
            bundle.event_embeddings = {}
            bundle.event_hub_components = []
            return {
                'event_descriptions': {},
                'event_embeddings': {},
                'event_hub_components': [],
                'construction_bundle': bundle,
            }
        vectors = await semantic.embed_descriptions(descriptions)
        source = bundle.source.key or ''
        bundle.event_descriptions = descriptions
        bundle.event_embeddings = vectors
        bundle.event_hub_components = [
            models.HubComponent(
                uuid=identity.event_uuid(source, position, name),
                source=source,
                node_id=position,
                name=name,
                description=descriptions.get(position, {}).get(name),
                embedding=list(vectors.get(position, {}).get(name, [])),
            )
            for triplet in triplets
            for position in triplet.evidence_positions
            for name, kind in (
                (triplet.subject, triplet.subject_kind),
                (triplet.object, triplet.object_kind),
            )
            if kind is models.NodeKind.EVENT
            and vectors.get(position, {}).get(name) is not None
        ]
        return {
            'event_descriptions': descriptions,
            'event_embeddings': vectors,
            'event_hub_components': bundle.event_hub_components,
            'construction_bundle': bundle,
        }
