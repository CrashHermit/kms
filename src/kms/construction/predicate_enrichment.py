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

    context_before: list[semantic.TermContextNodeInput] = dspy.InputField(
        description='Ordered preceding text context; reference only.'
    )
    target_node: semantic.TermContextNodeInput = dspy.InputField(
        description='The target node containing the local predicate context.'
    )
    context_after: list[semantic.TermContextNodeInput] = dspy.InputField(
        description='Ordered following text context; reference only.'
    )
    terms: list[str] = dspy.InputField(description='Exact predicate terms.')
    description: str = dspy.OutputField(
        description='One local description for the supplied term.'
    )


class PredicateEnricher(module.Module):
    signature = PredicateEnrichmentSignature
    record_name = 'predicate_enrichment'

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
                f'predicate enrichment expects one input term, got {len(terms)}'
            )
        return [TermDescription(term=terms[0], description=description)]


async def enrich(
    nodes,
    triplets,
    enricher: PredicateEnricher,
) -> dict[int, dict[str, str | None]]:
    terms_by_position: dict[int, set[str]] = {}
    for triplet in triplets:
        for position in triplet.evidence_positions:
            terms_by_position.setdefault(position, set()).add(triplet.predicate)
    stage = config.get_settings().stages.predicate_enrichment
    return await semantic.describe_terms(
        nodes,
        terms_by_position,
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
        bundle.predicate_hub_components = [
            models.HubComponent(
                uuid=identity.predicate_uuid(
                    triplet.occurrence_uuids[position]
                ),
                source=source,
                node_id=position,
                name=name,
                description=descriptions.get(position, {}).get(name),
                embedding=list(vectors.get(position, {}).get(name, [])),
            )
            for triplet in triplets
            for position in triplet.evidence_positions
            for name in (triplet.predicate,)
            if triplet.occurrence_uuids.get(position) is not None
            and vectors.get(position, {}).get(name) is not None
        ]
        return {
            'predicate_descriptions': descriptions,
            'predicate_embeddings': vectors,
            'predicate_hub_components': bundle.predicate_hub_components,
            'construction_bundle': bundle,
        }
