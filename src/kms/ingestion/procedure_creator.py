import asyncio
import logging
from collections.abc import Callable

import dspy

from kms import search
from kms.core import content, llm, models
from kms.graph import procedures, queries, writer

logger = logging.getLogger(__name__)


class NeedsJudgeSignature(dspy.Signature):
    r"""
    You are given a STATEMENT from a textbook.  Decide whether it needs
    a procedure — something that works it out.  The statement is given
    as its text followed by any figures that belong to it.

    A STATEMENT that only names or describes (a definition, a notation
    convention, a remark, a catalogue of facts) does NOT need a
    procedure.  Answer False.

    A STATEMENT that asks for something to be done, asserts something
    to be shown, or poses a problem the reader must resolve DOES need
    a procedure.  This includes theorems, propositions, lemmas,
    exercises, worked examples, and any claim followed by 'Prove',
    'Show', 'Find', 'Compute', 'Solve', 'Determine', or similar.

    Answer only True or False.
    """

    parts: content.ContentParts = dspy.InputField(
        description='The statement: its text followed by any figures.'
    )
    needs_procedure: bool = dspy.OutputField(
        description='True if this statement needs a procedure.'
    )


class NeedsJudge(dspy.Module):
    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.judge = dspy.Predict(NeedsJudgeSignature)
        self.set_lm(language_model or llm.pipeline_lm())

    async def aforward(self, parts: content.Content) -> bool:
        result = await self.judge.acall(
            parts=content.ContentParts(content=parts)
        )
        return result.needs_procedure

    def forward(self, parts: content.Content) -> bool:
        return asyncio.run(self.aforward(parts))


class ProcedureWriterSignature(dspy.Signature):
    r"""
    You are given a STATEMENT that needs a procedure — a proof, a
    solution, a derivation, a worked calculation, or any working-out
    that resolves what the statement posed.  The statement is given as
    its text followed by any figures that belong to it.  You are also
    given ENTITY DEFINITIONS that explain the concepts involved.

    Write the complete procedure as continuous prose.  Do not number
    steps and do not break it into a list — a later pass will split it.

    Use the entity definitions and the statement's figures to ground
    your reasoning.  Do not invent concepts not in the definitions.
    Every inference must be explicit and every gap filled.
    """

    parts: content.ContentParts = dspy.InputField(
        description='The statement: its text followed by any figures.'
    )
    entity_definitions: str = dspy.InputField(
        description='A formatted list of entity/concept definitions '
        'relevant to the statement.'
    )
    procedure: str = dspy.OutputField(
        description='The complete procedure — proof, solution, '
        'derivation, or calculation — written as continuous prose.'
    )


class ProcedureWriter(dspy.Module):
    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.writer = dspy.Predict(ProcedureWriterSignature)
        self.set_lm(language_model or llm.pipeline_lm())

    async def aforward(
        self, parts: content.Content, entity_definitions: str
    ) -> str:
        result = await self.writer.acall(
            parts=content.ContentParts(content=parts),
            entity_definitions=entity_definitions,
        )
        return result.procedure

    def forward(self, parts: content.Content, entity_definitions: str) -> str:
        return asyncio.run(self.aforward(parts, entity_definitions))


class StepSplitterSignature(dspy.Signature):
    r"""
    You are given a procedure (a proof, solution, derivation, or
    calculation) written as continuous prose.  Break it into an ordered
    list of logical steps.

    Each step is ONE logical move: a substitution, an application of a
    definition, a case split, an algebraic manipulation, a conclusion.

    Rules:
    - Preserve all notation and wording exactly as written.
    - Do not add, remove, or rephrase content.
    - Do not number the steps — the list order is the numbering.
    - Cut on natural logical boundaries, not on sentence boundaries.

    Return the steps as a list of strings.
    """

    procedure: str = dspy.InputField(
        description='The full procedure as continuous prose.'
    )
    steps: list[str] = dspy.OutputField(
        description='The procedure broken into ordered logical steps.'
    )


class StepSplitter(dspy.Module):
    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.splitter = dspy.Predict(StepSplitterSignature)
        self.set_lm(language_model or llm.pipeline_lm())

    async def aforward(self, procedure: str) -> list[str]:
        result = await self.splitter.acall(procedure=procedure)
        return list(result.steps or [])

    def forward(self, procedure: str) -> list[str]:
        return asyncio.run(self.aforward(procedure))


def _format_entity_definitions(entity_groups: list[search.SearchGroup]) -> str:
    parts: list[str] = []
    for group in entity_groups:
        if group.sub_query:
            parts.append(f'CONCEPT: {group.sub_query}')
        for result in group.results:
            name = result.properties.get('canonical_name', '')
            desc = result.properties.get('description', '')
            if name and desc:
                parts.append(f'{name}: {desc}')
    return '\n'.join(parts)


async def create_procedures(
    session_factory: Callable,
    *,
    language_model: dspy.LM | None = None,
    top_k: int = 10,
) -> int:
    orphans = await queries.orphan_statements(session_factory)
    if not orphans:
        logger.info('No orphan statements found.')
        return 0

    judge = NeedsJudge(language_model)
    writer_module = ProcedureWriter(language_model)
    splitter = StepSplitter(language_model)
    created = 0

    for orphan in orphans:
        statement_uuid = orphan['uuid']
        source = orphan['source']

        composed = await queries.compose_statement(
            statement_uuid, session_factory
        )
        statement_text = composed['text']
        if not statement_text.strip():
            logger.debug('Skipping empty statement %s', statement_uuid)
            continue

        parts = content.Content.from_text_and_pictures(
            statement_text, composed.get('pictures', [])
        )

        needs_procedure = await judge.aforward(parts)
        if not needs_procedure:
            logger.debug(
                'Skipping statement %s: no procedure needed',
                statement_uuid,
            )
            continue

        entity_groups = await search.search(
            query=statement_text,
            index_name='entity_hub_embedding',
            text_field='description',
            session_factory=session_factory,
            top_k=top_k,
        )
        entity_defs = _format_entity_definitions(entity_groups)

        procedure_text = await writer_module.aforward(parts, entity_defs)
        if not procedure_text.strip():
            logger.debug(
                'Skipping statement %s: empty procedure',
                statement_uuid,
            )
            continue

        steps = await splitter.aforward(procedure_text)
        if not steps:
            logger.debug(
                'Skipping statement %s: no steps extracted',
                statement_uuid,
            )
            continue

        procedure = models.Procedure(
            block=[],
            index=0,
            steps=[
                models.Step(text=text, index=i) for i, text in enumerate(steps)
            ],
        )
        procedure_list: list[models.Procedure] = [procedure]

        await writer.persist_procedures(
            procedure_list,
            source,
            session_factory=session_factory,
        )

        procedure_uuid = procedures.procedure_uuid(
            source, [], 0, statement_uuid=statement_uuid
        )
        now = writer.utcnow_iso()
        async with session_factory() as session:
            await session.run(
                queries.MERGE_HAS_PROCEDURE,
                pairs=[
                    {
                        'statement': statement_uuid,
                        'procedure': procedure_uuid,
                    }
                ],
                now=now,
            )

        created += 1
        logger.info(
            'Created procedure for statement %s with %d steps.',
            statement_uuid,
            len(steps),
        )

    return created
