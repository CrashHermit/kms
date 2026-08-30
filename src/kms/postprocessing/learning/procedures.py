"""Generate derived solution procedures after graph construction."""

import dspy

from kms.core import identity, models, module, recording


class ProcedureNeedRouterSignature(dspy.Signature):
    r"""
    Decide whether the supplied statement requires a worked solution.

    Return True when fulfilling it requires a proof, derivation, calculation,
    construction, verification, or other worked solution. Return False for
    definitions, notation, explanatory prose, headings, and assertions that
    can be understood without producing work. Return only True or False.
    """

    statement: list[models.TextNodeInput] = dspy.InputField(
        description='The complete canonical statement as ordered text nodes.'
    )
    needs_procedure: bool = dspy.OutputField(
        description='Whether the statement requires a generated worked solution.'
    )


class ProcedureNeedRouter(module.Module):
    """Routes statements that require a generated worked solution."""

    signature = ProcedureNeedRouterSignature
    record_name = 'procedure_need_router'

    def encode(self, statement: list[models.TextNodeInput]) -> dict:
        return {'statement': statement}

    def decode(self, prediction, **inputs) -> bool:
        """Returns the validated solution-routing decision."""
        return module.require_bool(
            prediction.needs_procedure, 'needs_procedure'
        )


class SolutionProcedureWriterSignature(dspy.Signature):
    r"""
    Write one complete generated solution for the supplied statement.

    The statement is authoritative about the goal and requested parts. If
    source procedure content is supplied, preserve its supported reasoning and
    use it as evidence while rewriting it into a complete standalone solution.
    If it is empty, write the correct proof, solution, derivation, calculation,
    or construction needed by the statement. Make meaningful reasoning
    explicit, answer every requested part, and do not invent facts.

    Return continuous prose with Markdown LaTeX. Return only the solution.
    """

    statement: list[models.TextNodeInput] = dspy.InputField(
        description='The complete canonical statement as ordered text nodes.'
    )
    source_procedure: list[models.TextNodeInput] = dspy.InputField(
        description='Existing source procedure text nodes, if any.'
    )
    canonical_knowledge: str = dspy.InputField(
        description='Supported canonical concepts, relations, and facts.'
    )
    procedure: str = dspy.OutputField(
        description='The complete generated worked solution.'
    )


class SolutionProcedureWriter(module.Module):
    """Writes generated solution procedure content."""

    signature = SolutionProcedureWriterSignature
    record_name = 'solution_procedure_writer'

    def encode(
        self,
        statement: list[models.TextNodeInput],
        source_procedure: list[models.TextNodeInput],
        canonical_knowledge: str,
    ) -> dict:
        return {
            'statement': statement,
            'source_procedure': source_procedure,
            'canonical_knowledge': canonical_knowledge,
        }

    def decode(self, prediction, **inputs) -> str:
        return module.require_text(prediction.procedure, 'procedure')


class SolutionProcedureGenerator:
    """Generates ``ProcedureKind.GENERATED`` solutions in post-processing."""

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        self.router = ProcedureNeedRouter(language_model, recorder=recorder)
        self.writer = SolutionProcedureWriter(language_model, recorder=recorder)

    async def generate(
        self, item: models.ProcedureEnrichmentInput
    ) -> models.Procedure | None:
        """Generates one solution procedure, or returns None when unnecessary."""
        statement = list(item.statement)
        if not any(node.node_text.strip() for node in statement):
            return None
        needs_procedure = await self.router.aforward(statement=statement)
        if not needs_procedure:
            return None
        result = await self.writer.aforward(
            statement=statement,
            source_procedure=list(item.procedure or ()),
            canonical_knowledge=item.canonical_knowledge,
        )
        text = result.strip()
        if not text:
            return None
        procedure_uuid = identity.procedure_uuid(
            item.source,
            [],
            0,
            statement_uuid_value=item.statement_uuid,
            kind=models.ProcedureKind.GENERATED.value,
        )
        return models.Procedure(
            block=[],
            index=0,
            kind=models.ProcedureKind.GENERATED,
            uuid=procedure_uuid,
            statement_uuid=item.statement_uuid,
            procedure=text,
        )
