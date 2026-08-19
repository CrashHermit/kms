import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.construction import hub_inputs
from kms.core import content, module, semantic, state


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

    passage: content.ContentParts = dspy.InputField(
        description='The passage with optional figures.'
    )
    terms: list[str] = dspy.InputField(description='Exact entity terms.')
    descriptions: list[TermDescription] = dspy.OutputField(
        description='One local description per term in input order.'
    )


class EntityEnricher(module.Module):
    signature = EntityEnrichmentSignature
    record_name = 'entity_enrichment'

    def encode(self, passage: content.Content, terms: list[str]) -> dict:
        return {
            'passage': content.ContentParts(content=passage),
            'terms': terms,
        }

    def decode(self, prediction, **inputs):
        return module.as_list(prediction.descriptions)


async def enrich(
    nodes,
    triplets,
    enricher: EntityEnricher,
) -> dict[int, dict[str, str | None]]:
    terms_by_node: dict[int, set[str]] = {}
    for triplet in triplets:
        for node_id in triplet.node_ids:
            terms_by_node.setdefault(node_id, set()).update(
                (triplet.subject, triplet.object)
            )
    stage = config.get_settings().stages.entity_enrichment
    return await semantic.describe_terms(
        nodes,
        terms_by_node,
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
        bundle.entity_hub_components = list(
            hub_inputs.build_hub_components(
                kind='entity',
                source=source,
                triplets=triplets,
                descriptions=descriptions,
                embeddings=vectors,
            )
        )
        return {
            'entity_descriptions': descriptions,
            'entity_embeddings': vectors,
            'entity_hub_components': bundle.entity_hub_components,
            'construction_bundle': bundle,
        }
