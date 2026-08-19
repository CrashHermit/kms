import dspy
from pydantic import BaseModel, Field

from kms import config
from kms.construction import hub_inputs
from kms.core import content, module, semantic, state


class TermDescription(BaseModel):
    term: str = Field(description='The exact input predicate term.')
    description: str = Field(
        description='A concise local predicate description.'
    )


class PredicateEnrichmentSignature(dspy.Signature):
    r"""
    Describe each predicate term in the supplied technical passage.

    Explain the local relation represented by each verb phrase. Preserve its
    direction, subject and object roles, negation, conditions, causality, and
    temporal meaning. Do not create a general relation, merge synonyms, or
    invent unsupported facts. Return exactly one description per input term in
    order.
    """

    passage: content.ContentParts = dspy.InputField(
        description='The passage with optional figures.'
    )
    terms: list[str] = dspy.InputField(description='Exact predicate terms.')
    descriptions: list[TermDescription] = dspy.OutputField(
        description='One local description per term in input order.'
    )


class PredicateEnricher(module.Module):
    signature = PredicateEnrichmentSignature
    record_name = 'predicate_enrichment'

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
    enricher: PredicateEnricher,
) -> dict[int, dict[str, str | None]]:
    terms_by_node: dict[int, set[str]] = {}
    for triplet in triplets:
        for node_id in triplet.node_ids:
            terms_by_node.setdefault(node_id, set()).add(triplet.predicate)
    stage = config.get_settings().stages.predicate_enrichment
    return await semantic.describe_terms(
        nodes,
        terms_by_node,
        enricher,
        stage.before_budget,
        stage.after_budget,
        stage.max_concurrent_calls,
    )


class PredicateEnrichmentNode:
    def __init__(self, enricher: PredicateEnricher) -> None:
        self._enricher = enricher

    async def run(self, current_state: dict) -> dict:
        bundle = state.to_construction_bundle(current_state)
        triplets = bundle.triplets
        if not triplets:
            return {'construction_bundle': bundle}
        descriptions = await enrich(bundle.nodes, triplets, self._enricher)
        vectors = await semantic.embed_descriptions(descriptions)
        source = bundle.source.key or ''
        bundle.predicate_descriptions = descriptions
        bundle.predicate_embeddings = vectors
        bundle.predicate_hub_components = list(
            hub_inputs.build_hub_components(
                kind='predicate',
                source=source,
                triplets=triplets,
                descriptions=descriptions,
                embeddings=vectors,
            )
        )
        return {
            'predicate_descriptions': descriptions,
            'predicate_embeddings': vectors,
            'predicate_hub_components': bundle.predicate_hub_components,
            'construction_bundle': bundle,
        }
