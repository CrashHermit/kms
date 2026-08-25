import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.core import (
    context_window,
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

    context_before: list[semantic.TermContextNodeInput] = dspy.InputField(
        description='Ordered preceding text context; reference only.'
    )
    target_node: semantic.TermContextNodeInput = dspy.InputField(
        description='The target node containing the local entity term context.'
    )
    context_after: list[semantic.TermContextNodeInput] = dspy.InputField(
        description='Ordered following text context; reference only.'
    )
    terms: list[str] = dspy.InputField(description='Exact entity terms.')
    description: str = dspy.OutputField(
        description='One local description for the supplied term.'
    )


class EntityEnricher(module.Module):
    signature = EntityEnrichmentSignature
    record_name = 'entity_enrichment'

    def encode(
        self,
        context_before: list[context_window.ContextNode],
        target_node: context_window.ContextNode,
        context_after: list[context_window.ContextNode],
        terms: list[str],
    ) -> dict[str, object]:
        return {
            'context_before': [
                semantic.term_context_input(node, index)
                for index, node in enumerate(context_before)
            ],
            'target_node': semantic.term_context_input(target_node),
            'context_after': [
                semantic.term_context_input(node, index)
                for index, node in enumerate(context_after)
            ],
            'terms': terms,
        }

    def decode(self, prediction, **inputs) -> list[TermDescription]:
        """Returns one validated description for the singleton input term."""
        description = module.require_text(
            prediction.description, 'description'
        )
        terms = inputs['terms']
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
            terms_by_position.setdefault(position, set()).update(
                (triplet.subject, triplet.object)
            )
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
            for name in dict.fromkeys((triplet.subject, triplet.object))
            if vectors.get(position, {}).get(name) is not None
        ]
        return {
            'entity_descriptions': descriptions,
            'entity_embeddings': vectors,
            'entity_hub_components': bundle.entity_hub_components,
            'construction_bundle': bundle,
        }
