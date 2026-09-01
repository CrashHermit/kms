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


class TermDescription(BaseModel):
    term: str = Field(description='The exact input entity term.')
    description: str = Field(description='A concise local entity description.')


class EntityEnrichmentSignature(dspy.Signature):
    r"""
    Describe each entity term in the supplied technical passage.

    Write a concise source-local gloss of what each noun phrase, object, or
    concept means in this passage. Resolve notation and local references, but
    do not create a general definition, merge synonyms, or invent unsupported
    facts. Return exactly one description per input term in order.
    """

    request: models.TermEnrichmentInput = dspy.InputField(
        description=(
            'The enrichment request. Describe only target_node.text. '
            'context_before and context_after provide reference context only.'
        )
    )
    description: str = dspy.OutputField(
        description='One local description for the supplied term.'
    )


class EntityEnricher(module.Module):
    signature = EntityEnrichmentSignature
    record_name = 'entity_enrichment'

    def encode(self, request: models.TermEnrichmentInput) -> dict[str, object]:
        """Passes the validated enrichment request to the signature."""
        return {'request': request}

    def decode(self, prediction, **inputs) -> list[TermDescription]:
        """Returns one validated description for the singleton input term."""
        description = module.require_text(prediction.description, 'description')
        terms = inputs['request'].terms
        if len(terms) != 1:
            raise ValueError(
                f'entity enrichment expects one input term, got {len(terms)}'
            )
        return [TermDescription(term=terms[0], description=description)]


async def enrich(
    nodes,
    triplets,
    enricher: EntityEnricher,
) -> dict[int, dict[str, str | None]]:
    terms_by_position: dict[int, set[str]] = {}
    for triplet in triplets:
        for position in triplet.evidence_positions:
            if triplet.subject_kind is models.NodeKind.ENTITY:
                terms_by_position.setdefault(position, set()).add(
                    triplet.subject
                )
            if triplet.object_kind is models.NodeKind.ENTITY:
                terms_by_position.setdefault(position, set()).add(
                    triplet.object
                )
    if not terms_by_position:
        return {}
    stage = config.get_settings().stages.entity_enrichment
    return await semantic.describe_terms(
        nodes,
        terms_by_position,
        enricher,
        stage.before_budget,
        stage.after_budget,
        stage.max_concurrent_calls,
    )


class EntityEnrichmentNode:
    def __init__(self, enricher: EntityEnricher) -> None:
        self._enricher = enricher

    async def run(self, current_state: dict) -> dict:
        bundle = state.to_construction_bundle(current_state)
        triplets = bundle.triplets
        if not triplets:
            return {'construction_bundle': bundle}
        descriptions = await enrich(bundle.nodes, triplets, self._enricher)
        vectors = await semantic.embed_descriptions(descriptions)
        source = bundle.source.key or ''
        bundle.entity_descriptions = descriptions
        bundle.entity_embeddings = vectors
        bundle.entity_hub_components = [
            models.HubComponent(
                uuid=identity.entity_uuid(source, position, name),
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
            if kind is models.NodeKind.ENTITY
            and vectors.get(position, {}).get(name) is not None
        ]
        return {
            'entity_descriptions': descriptions,
            'entity_embeddings': vectors,
            'entity_hub_components': bundle.entity_hub_components,
            'construction_bundle': bundle,
        }
