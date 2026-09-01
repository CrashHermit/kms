"""Shared langgraph state schema for the ingestion pipeline."""

import copy
import operator
from typing import Annotated, TypedDict

from kms.core import models


class State(TypedDict, total=False):
    """The mutable state threaded through every graph node.

    All keys are optional; worker results are merged in via the
    Annotated reducer fields. ``source`` identifies the source being ingested,
    ``documents`` carries mutable page documents, and ``nodes`` carries the
    flattened parsed stream before persistence.
    """

    pdf_path: str
    output_dir: str
    pages: list[int] | None
    ocr_response_path: str | None
    source: models.Source
    construction_bundle: Annotated[
        models.ConstructionBundle,
        lambda old, new: new if new is not None else old,
    ]
    documents: list[models.Document]
    nodes: list[models.SourceNode]
    spans: list[list[int]]
    instructions: list[models.Instruction]
    statements: list[models.Statement]
    procedures: list[models.Procedure]
    triplets: list[models.Triplet]
    knowledge_index: models.KnowledgeIndex
    statement_enrichment_inputs: list[models.StatementEnrichmentInput]
    procedure_enrichment_inputs: list[models.ProcedureEnrichmentInput]
    statement_hub_records: list[models.StatementHubRecord]
    procedure_hub_records: list[models.ProcedureHubRecord]
    entity_hub_bundle: models.HubBuildBundle
    event_hub_bundle: models.HubBuildBundle
    predicate_hub_bundle: models.HubBuildBundle
    entity_hub_components: list[models.HubComponent]
    event_hub_components: list[models.HubComponent]
    predicate_hub_components: list[models.HubComponent]
    entity_hub_assignments: list[dict]
    event_hub_assignments: list[dict]
    predicate_hub_assignments: list[dict]
    entity_hub_records: list[dict]
    event_hub_records: list[dict]
    predicate_hub_records: list[dict]
    statement_enrichments: list[dict]
    procedure_enrichments: list[dict]
    statement_hubs: list[dict]
    procedure_hubs: list[dict]
    entity_descriptions: dict[int, dict[str, str | None]]
    event_descriptions: dict[int, dict[str, str | None]]
    predicate_descriptions: dict[int, dict[str, str | None]]
    entity_embeddings: dict[int, dict[str, list[float]]]
    event_embeddings: dict[int, dict[str, list[float]]]
    predicate_embeddings: dict[int, dict[str, list[float]]]
    generated_procedures: list[models.Procedure]
    procedure_links: list[models.ProcedureLink]
    statements_enriched: int
    procedures_enriched: int
    statement_hubs_created: int
    procedure_hubs_created: int
    triplet_hubs_created: int
    triplets_clustered: int
    statement_hub_diagnostics: dict[str, int]
    procedure_hub_diagnostics: dict[str, int]
    projected: bool
    statements_clustered: int
    procedures_clustered: int
    block_correction_results: Annotated[
        list[tuple[int, list[models.SourceNode]]], operator.add
    ]
    format_results: Annotated[list[tuple[int, int, str]], operator.add]
    extract_results: Annotated[
        list[tuple[int, list[models.SourceNode]]], operator.add
    ]
    seam_even_results: Annotated[
        list[tuple[int, list[models.SourceNode]]], operator.add
    ]
    seam_odd_results: Annotated[
        list[tuple[int, list[models.SourceNode]]], operator.add
    ]
    image_seam_even_results: Annotated[
        list[tuple[int, list[models.SourceNode]]], operator.add
    ]
    image_seam_odd_results: Annotated[
        list[tuple[int, list[models.SourceNode]]], operator.add
    ]


def to_construction_bundle(current_state: State) -> models.ConstructionBundle:
    """Projects durable construction data out of workflow state.

    This is a pure boundary conversion. Workflow inputs, reducer scratch
    fields, and progress counters intentionally remain in ``State``.
    """
    existing_bundle = current_state.get('construction_bundle')
    if existing_bundle is not None:
        return copy.deepcopy(existing_bundle)

    source = current_state.get('source')
    if source is None or not source.key or not source.key.strip():
        raise ValueError('state must contain a non-empty source')

    return models.ConstructionBundle(
        source=copy.deepcopy(source),
        nodes=copy.deepcopy(current_state.get('nodes', [])),
        instructions=copy.deepcopy(current_state.get('instructions', [])),
        statements=copy.deepcopy(current_state.get('statements', [])),
        procedures=copy.deepcopy(current_state.get('procedures', [])),
        triplets=copy.deepcopy(current_state.get('triplets', [])),
        knowledge_index=copy.deepcopy(current_state.get('knowledge_index')),
        statement_enrichment_inputs=copy.deepcopy(
            current_state.get('statement_enrichment_inputs', [])
        ),
        procedure_enrichment_inputs=copy.deepcopy(
            current_state.get('procedure_enrichment_inputs', [])
        ),
        statement_hub_records=copy.deepcopy(
            current_state.get('statement_hub_records', [])
        ),
        procedure_hub_records=copy.deepcopy(
            current_state.get('procedure_hub_records', [])
        ),
        entity_hub_bundle=copy.deepcopy(current_state.get('entity_hub_bundle')),
        event_hub_bundle=copy.deepcopy(current_state.get('event_hub_bundle')),
        predicate_hub_bundle=copy.deepcopy(
            current_state.get('predicate_hub_bundle')
        ),
        entity_hub_components=copy.deepcopy(
            current_state.get('entity_hub_components', [])
        ),
        event_hub_components=copy.deepcopy(
            current_state.get('event_hub_components', [])
        ),
        predicate_hub_components=copy.deepcopy(
            current_state.get('predicate_hub_components', [])
        ),
        entity_hub_assignments=copy.deepcopy(
            current_state.get('entity_hub_assignments', [])
        ),
        event_hub_assignments=copy.deepcopy(
            current_state.get('event_hub_assignments', [])
        ),
        predicate_hub_assignments=copy.deepcopy(
            current_state.get('predicate_hub_assignments', [])
        ),
        entity_hub_records=copy.deepcopy(
            current_state.get('entity_hub_records', [])
        ),
        event_hub_records=copy.deepcopy(
            current_state.get('event_hub_records', [])
        ),
        predicate_hub_records=copy.deepcopy(
            current_state.get('predicate_hub_records', [])
        ),
        statement_enrichments=copy.deepcopy(
            current_state.get('statement_enrichments', [])
        ),
        procedure_enrichments=copy.deepcopy(
            current_state.get('procedure_enrichments', [])
        ),
        statement_hubs=copy.deepcopy(current_state.get('statement_hubs', [])),
        procedure_hubs=copy.deepcopy(current_state.get('procedure_hubs', [])),
        generated_procedures=copy.deepcopy(
            current_state.get('generated_procedures', [])
        ),
        procedure_links=copy.deepcopy(current_state.get('procedure_links', [])),
        entity_descriptions=copy.deepcopy(
            current_state.get('entity_descriptions', {})
        ),
        event_descriptions=copy.deepcopy(
            current_state.get('event_descriptions', {})
        ),
        predicate_descriptions=copy.deepcopy(
            current_state.get('predicate_descriptions', {})
        ),
        entity_embeddings=copy.deepcopy(
            current_state.get('entity_embeddings', {})
        ),
        event_embeddings=copy.deepcopy(
            current_state.get('event_embeddings', {})
        ),
        predicate_embeddings=copy.deepcopy(
            current_state.get('predicate_embeddings', {})
        ),
    )
