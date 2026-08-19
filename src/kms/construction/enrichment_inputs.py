"""Build typed enrichment inputs from in-memory construction state."""

from kms.construction import enrichment
from kms.core import models, state


def build_statement_inputs(
    bundle: models.ConstructionBundle,
) -> list[models.StatementEnrichmentInput]:
    """Builds enrichment inputs for every statement in a bundle."""
    return [
        enrichment.statement_enrichment_input(
            bundle, statement, bundle.knowledge_index
        )
        for statement in bundle.statements
    ]


def build_procedure_inputs(
    bundle: models.ConstructionBundle,
) -> list[models.ProcedureEnrichmentInput]:
    """Builds enrichment inputs for every procedure in a bundle."""
    return [
        enrichment.procedure_enrichment_input(
            bundle, procedure, bundle.knowledge_index
        )
        for procedure in bundle.procedures
    ]


class StatementEnrichmentInputNode:
    """Builds statement enrichment inputs without graph access."""

    async def run(self, current_state: state.State) -> dict:
        """Builds and stores source-local statement inputs in the bundle."""
        bundle = state.to_construction_bundle(current_state)
        inputs = build_statement_inputs(bundle)
        bundle.statement_enrichment_inputs = inputs
        return {
            'statement_enrichment_inputs': inputs,
            'construction_bundle': bundle,
        }


class ProcedureEnrichmentInputNode:
    """Builds procedure enrichment inputs without graph access."""

    async def run(self, current_state: state.State) -> dict:
        """Builds and stores source-local procedure inputs in the bundle."""
        bundle = state.to_construction_bundle(current_state)
        inputs = build_procedure_inputs(bundle)
        bundle.procedure_enrichment_inputs = inputs
        return {
            'procedure_enrichment_inputs': inputs,
            'construction_bundle': bundle,
        }
