"""Creates procedures for orphan statements that need working-out."""

import logging
from collections.abc import Callable

import dspy

from kms import config
from kms.core import content, llm, models, module, search
from kms.graph import queries, writer
from kms.graph.procedures import procedure_uuid

logger = logging.getLogger(__name__)


class NeedsJudgeSignature(dspy.Signature):
    r"""
    TASK

    Decide whether this textbook statement needs a newly written procedure:
    a proof, solution, derivation, calculation, or other working-out that
    resolves what the statement asks. The statement is followed by any
    figures that belong to it.

    Return True when a learner must do work to satisfy the statement. This
    includes a question or task asking the learner to prove, show, find,
    compute, solve, determine, derive, verify, sketch, or explain something;
    an exercise or worked example with a result to obtain; and a theorem,
    proposition, or lemma presented with a claim that must be established.
    The request may be implicit: a declarative problem such as "Find the
    velocity components ..." still needs a procedure even without a question
    mark.

    Return False when the statement can be understood without carrying out a
    proof or solution. This includes definitions, notation conventions,
    descriptions, lists or catalogues of facts, remarks, explanatory prose,
    headings, learning objectives, and assertions that state a fact without
    asking the reader to establish or calculate anything. Mathematical
    notation alone does not make a procedure necessary.

    DECISION TEST: Would a learner need to produce an intermediate argument,
    calculation, construction, or verification to fulfill this statement? If
    yes, return True. If understanding the statement is sufficient, return
    False.

    Return only True or False.
    """

    parts: content.ContentParts = dspy.InputField(
        description='The statement: its text followed by any figures.'
    )
    needs_procedure: bool = dspy.OutputField(
        description='True if this statement needs a procedure.'
    )


class NeedsJudge(module.Module):
    """Decides whether a statement needs a procedure at all."""

    signature = NeedsJudgeSignature
    record_name = 'needs_judge'

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__(language_model or llm.module_lm('procedure_creator'))

    def encode(self, parts: content.Content) -> dict:
        """Builds the judge-signature kwargs for one statement."""
        return {'parts': content.ContentParts(content=parts)}

    def decode(self, prediction, **inputs) -> bool:
        """True when the statement poses something to be worked out."""
        return prediction.needs_procedure


class ProcedureWriterSignature(dspy.Signature):
    r"""
    TASK

    You are given a textbook STATEMENT that needs a procedure: a proof,
    solution, derivation, worked calculation, or other reasoning that resolves
    what the statement asks. The statement is followed by any figures that
    belong to it. ENTITY DEFINITIONS provide relevant facts and terminology.

    Write the complete, correct solution to THIS statement only. Use the
    statement, its figures, and the entity definitions; do not solve a
    neighboring exercise or import a task from surrounding textbook text.
    Answer every explicitly requested part, preserving each part label and
    keeping independent parts distinguishable. Make every inference explicit:
    introduce the rule or definition being used, perform each meaningful
    calculation, and explain how each result leads to the next. Include the
    final result and its conditions. Do not invent facts, assumptions,
    concepts, or data that are not supported by the inputs.

    Write one continuous piece of prose with equations where needed. Do not
    number steps or format the response as a list; a later pass will divide
    this one solution into learnable stages. Do not give only a hint, outline,
    summary, or final answer. Do not merge independent subproblems into a
    single unexplained conclusion.

    LATEX FORMAT IS REQUIRED. Write all mathematical notation in LaTeX.
    Use `$...$` for inline mathematics and `$$...$$` for display mathematics.
    Put each display equation on its own lines. Never leave mathematical
    notation bare, use Unicode math symbols when a LaTeX command exists, or
    use `\\(...\\)` / `\\[...\\]` delimiters. Preserve the meaning of every
    expression while choosing the delimiters.
    """

    parts: content.ContentParts = dspy.InputField(
        description='The statement: its text followed by any figures.'
    )
    entity_definitions: str = dspy.InputField(
        description='A formatted list of entity/concept definitions '
        'relevant to the statement.'
    )
    procedure: str = dspy.OutputField(
        description=(
            'The complete correct solution in continuous prose with '
            'Markdown LaTeX: `$...$` inline and `$$...$$` display.'
        )
    )


class ProcedureWriter(module.Module):
    """Writes a complete procedure for a statement, as continuous prose."""

    signature = ProcedureWriterSignature
    record_name = 'procedure_writer'

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__(language_model or llm.module_lm('procedure_creator'))

    def encode(self, parts: content.Content, entity_definitions: str) -> dict:
        """Builds the writer-signature kwargs for one statement."""
        return {
            'parts': content.ContentParts(content=parts),
            'entity_definitions': entity_definitions,
        }

    def decode(self, prediction, **inputs) -> str:
        """Returns the written procedure text for the statement."""
        return prediction.procedure


class StepSplitterSignature(dspy.Signature):
    r"""
    TASK

    Split the complete procedure below into an ordered list of learnable
    stages. The procedure may be a proof, solution, derivation, calculation,
    or other working-out.

    OUTPUT CONTRACT

    Every output item must be a verbatim, contiguous excerpt of the input.
    Copy the input; do not rewrite it. Use every sentence, equation, and
    reasoning clause exactly once across the stages. Do not summarize, explain,
    answer, or add commentary.

    The first stage must begin with the beginning of the input. The final
    stage must end with the end of the input. Never return only the answer or
    final conclusion. A multi-phase procedure must have multiple stages: keep
    the setup, each major piece of reasoning, and the conclusion. Return one
    stage only when the entire input is genuinely one inseparable action.

    STAGE BOUNDARIES

    Start a new stage when the procedure:
    - sets up the goal or introduces the relevant definition;
    - applies a distinct rule, formula, substitution, or calculation;
    - reaches an important intermediate result;
    - changes strategy or begins a new case; or
    - draws the final conclusion.

    Preserve explicit part labels such as `(a)`, `(b)`, or `Part 1` as
    boundaries. Independent subproblems must not be placed in one stage,
    even when they use the same method. A stage may contain several tightly
    coupled sentences, but it must represent one coherent action and its
    result. Keep an action with the reasoning and result it produces; do not
    split tightly coupled algebra into microscopic fragments.

    Each stage should be understandable and useful to a learner on its own,
    while remaining a contiguous part of the original procedure.

    MATH AND MARKUP

    Preserve every mathematical expression exactly. Use `$...$` for inline
    math and `$$...$$` for display math. Never remove, invent, or convert math
    delimiters. Never use bare mathematical notation or `\(...\)` /
    `\[...\]` delimiters.

    FINAL CHECK

    Before returning the list, check that:
    1. the first stage contains the input's opening;
    2. the last stage contains the input's ending;
    3. no input text was omitted, summarized, duplicated, or reordered; and
    4. a multi-phase procedure was not collapsed into one conclusion-only
       stage.

    Return only the list of stage strings.
    """

    procedure: str = dspy.InputField(
        description='The full procedure as continuous prose.'
    )
    steps: list[str] = dspy.OutputField(
        description=(
            'The complete procedure copied into coherent, ordered '
            'pedagogical stages; never a summary or final answer only.'
        )
    )


class StepSplitter(module.Module):
    """Splits a written procedure into ordered pedagogical stages."""

    signature = StepSplitterSignature
    record_name = 'step_splitter'

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__(language_model or llm.module_lm('procedure_creator'))

    def encode(self, procedure: str) -> dict:
        """Builds the step-splitter kwargs for one procedure."""
        return {'procedure': procedure}

    def decode(self, prediction, **inputs) -> list[str]:
        """Returns the model's pedagogical stages."""
        return module.as_list(prediction.steps)


def _format_entity_definitions(entity_groups: list[search.SearchGroup]) -> str:
    """Formats search groups into the entity-definition prompt block."""
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
    top_k: int | None = None,
) -> int:
    """Creates procedures for all orphaned statements that need one.

    For each statement that passes the needs-procedure judge, the
    statement is composed, relevant entity definitions are searched,
    a procedure is written and split into steps, then persisted and
    linked to the statement.

    Args:
        session_factory: Async callable returning a Neo4j session.
        language_model: LM for judging, writing, and splitting.
        top_k: Number of entity definitions to retrieve per statement.

    Returns:
        The number of procedures created.
    """
    if top_k is None:
        top_k = config.get_settings().stages.procedure.entity_definition_top_k
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

        needs_procedure = await judge.aforward(parts=parts)
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
            source=source,
            top_k=top_k,
            language_model=language_model,
        )
        entity_defs = _format_entity_definitions(entity_groups)

        procedure_text = await writer_module.aforward(
            parts=parts, entity_definitions=entity_defs
        )
        if not procedure_text.strip():
            logger.debug(
                'Skipping statement %s: empty procedure',
                statement_uuid,
            )
            continue

        steps = await splitter.aforward(procedure=procedure_text)
        if not steps:
            logger.debug(
                'Skipping statement %s: no steps extracted',
                statement_uuid,
            )
            continue

        procedure = models.Procedure(
            block=[],
            index=0,
            statement_uuid=statement_uuid,
            steps=[
                models.Step(text=text, index=i) for i, text in enumerate(steps)
            ],
        )
        procedures: list[models.Procedure] = [procedure]

        await writer.persist_procedures(
            procedures,
            source,
            session_factory=session_factory,
        )

        procedure_uuid_value = procedure_uuid(
            source,
            procedure.block,
            procedure.index,
            statement_uuid=procedure.statement_uuid,
        )
        now = writer.utcnow_iso()
        async with session_factory() as session:
            await session.run(
                queries.MERGE_HAS_PROCEDURE,
                pairs=[
                    {
                        'statement': statement_uuid,
                        'procedure': procedure_uuid_value,
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
