import dspy
from pydantic import BaseModel, Field

from kms.construction import hub_engine
from kms.core import content, module
from kms.graph import entity_hubs as entity_graph
from kms.graph import queries, writer

SOURCE_SPEC = hub_engine.HubBuildSpec(
    domain='entity',
    tier='source',
    graph=entity_graph,
    stage_name='entity_hubs',
    hub_id_factory=hub_engine.source_hub_id_factory(entity_graph),
    source_resolver=hub_engine._source_scope,
    record_adapter=hub_engine.component_records,
    adjudication_context='Compare durable component mentions within one source.',
    synthesis_context='Synthesize a source-local semantic hub.',
    all_components=queries.all_entity_components,
    clear_hubs=writer.clear_entity_hubs,
    persist_hubs=writer.persist_entity_hubs,
    rebuild_names=hub_engine._rebuild_names_callback('entity'),
    rebuild_triplets=hub_engine._rebuild_triplets_callback(),
)

META_SPEC = hub_engine.HubBuildSpec(
    domain='entity',
    tier='meta',
    graph=entity_graph,
    stage_name='entity_hubs',
    hub_id_factory=hub_engine.meta_hub_id_factory(entity_graph),
    source_resolver=hub_engine._no_source,
    record_adapter=hub_engine.source_hub_records,
    adjudication_context='Compare source-local hubs across different sources.',
    synthesis_context='Synthesize a cross-source semantic hub.',
    all_source_hubs=queries.all_entity_source_hubs,
    qualified_meta_hub_uuids=queries.qualified_entity_meta_hub_uuids,
    index_name='meta_entity_hub_embedding',
    clear_hubs=writer.clear_entity_meta_hubs,
    persist_hubs=writer.persist_entity_hubs,
    attach_meta_hubs=writer.attach_entity_meta_hubs,
    clear_invalid_meta_hubs=writer.clear_invalid_entity_meta_hubs,
    rebuild_meta_names=hub_engine._rebuild_meta_names_callback('entity'),
    rebuild_triplets=hub_engine._rebuild_triplets_callback(),
)


class EntityHubDefinition(BaseModel):
    canonical_name: str = Field(
        description='The canonical name for the entity concept.'
    )
    description: str = Field(
        description='A standalone learner-facing entity description.'
    )


class EntityHubAdjudication(BaseModel):
    decision: str = Field(
        description='Merge, Hierarchy, or Separate for the two entity mentions.'
    )
    more_general: str = Field(
        default='none',
        description='left or right when the decision is Hierarchy.',
    )


class EntityHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one source-local learner-facing entity concept from supplied
    surface forms and their passage-grounded descriptions. Generalize only
    what the evidence supports. Do not mention the source or invent facts.
    """

    surface_forms: list[str] = dspy.InputField()
    descriptions: list[str] = dspy.InputField()
    scope: str = dspy.InputField()
    result: EntityHubDefinition = dspy.OutputField()


class EntityHubAdjudicationSignature(dspy.Signature):
    r"""
    Compare two entity mentions. Return Merge for the same concept, Hierarchy
    only for a strict kind-of relation, and Separate otherwise. Composition
    and part-whole relations are Separate.
    """

    left: content.ContentParts = dspy.InputField()
    right: content.ContentParts = dspy.InputField()
    scope: str = dspy.InputField()
    result: EntityHubAdjudication = dspy.OutputField()


class EntityHubSynthesizer(module.Module):
    signature = EntityHubSynthesisSignature
    record_name = 'entity_hub_synthesizer'

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


class EntityHubAdjudicator(module.Module):
    signature = EntityHubAdjudicationSignature
    record_name = 'entity_hub_adjudicator'

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

    def decode(self, prediction, **inputs) -> EntityHubAdjudication:
        return prediction.result


async def assign_source_hubs(
    bundle,
    *,
    max_concurrency: int | None = None,
    adjudicator,
    synthesizer,
) -> dict:
    """Assign entity components to source-local EntityHubs."""
    return await hub_engine.assign_source_hubs(
        bundle,
        spec=SOURCE_SPEC,
        max_concurrency=max_concurrency,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
        include_outputs=True,
    )


class EntityHubNode:
    def __init__(
        self,
        adjudicator,
        synthesizer,
    ) -> None:
        self._adjudicator = adjudicator
        self._synthesizer = synthesizer

    async def run(self, current_state: dict) -> dict:
        from kms.core import state

        construction_bundle = state.to_construction_bundle(current_state)
        bundle = construction_bundle.entity_hub_bundle
        if bundle is None:
            return {'construction_bundle': construction_bundle}
        result = await assign_source_hubs(
            bundle,
            adjudicator=self._adjudicator,
            synthesizer=self._synthesizer,
        )
        construction_bundle.entity_hub_assignments = result.get(
            'assignments', []
        )
        construction_bundle.entity_hub_records = result.get('hubs', [])
        return {
            'entity_hub_assignments': construction_bundle.entity_hub_assignments,
            'entity_hub_records': construction_bundle.entity_hub_records,
            'construction_bundle': construction_bundle,
        }


async def rebuild_source(
    source: str,
    *,
    language_model,
    adjudicator,
    synthesizer,
    session_factory,
    max_concurrency: int | None = None,
) -> dict:
    """Rebuild source-local entity hubs from persisted components."""
    return await hub_engine.rebuild_hubs(
        spec=SOURCE_SPEC,
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
    """Rebuild cross-source entity hubs from source-local hubs."""
    return await hub_engine.rebuild_meta_hubs(
        spec=META_SPEC,
        language_model=language_model,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
        session_factory=session_factory,
        max_concurrency=max_concurrency,
    )
