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
    term: str = Field(description='The exact input predicate term.')
    description: str = Field(
        description='A concise local predicate description.'
    )


class PredicateEnrichmentSignature(dspy.Signature):
    r"""
    Give each supplied predicate term a compact, source-grounded meaning.

    The predicate is already the graph relation. Describe only the relation's
    semantics, not the whole source sentence. Preserve direction, negation,
    modality, causality, and temporal meaning when explicitly present. Do not
    restate the subject, object, formula, point, vector, or surrounding
    exercise.

    Output one short description for the one supplied term:
    - 2–12 plain-language words;
    - no LaTeX, equations, names, values, or source-specific details;
    - no sentence with a subject and object copied from the passage;
    - no new facts, examples, explanations, or generic textbook commentary.

    Examples:
    - term: `is defined as` → `Defines a term by its expression.`
    - term: `is orthogonal to` → `Expresses orthogonality between objects.`
    - term: `equals` → `States equality between two values.`
    """

    request: models.TermEnrichmentInput = dspy.InputField(
        description=(
            'The enrichment request. Describe only target_node.text. '
            'context_before and context_after provide reference context only.'
        )
    )
    description: str = dspy.OutputField(
        description=(
            'A compact 2–12-word semantic description of the relation. '
            'Do not include formulas, entities, values, or source details.'
        )
    )


class PredicateEnricher(module.Module):
    signature = PredicateEnrichmentSignature
    record_name = 'predicate_enrichment'

    def encode(self, request: models.TermEnrichmentInput) -> dict[str, object]:
        """Passes the validated enrichment request to the signature."""
        return {'request': request}

    def decode(self, prediction, **inputs) -> list[TermDescription]:
        """Returns one model-generated description for the singleton term."""
        description = module.require_text(prediction.description, 'description')
        terms = inputs['request'].terms
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
