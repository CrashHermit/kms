"""Turn source statements into reusable and actionable learning.

EntityHub and PredicateHub descriptions provide canonical concept and relation
knowledge, while TripletHub descriptions provide reusable canonical facts.
This module uses that knowledge to explain how a learner solves a particular
statement: procedures give solution-level reasoning and Steps preserve
coherent learner-facing actions, justifications, and results.
"""

import logging
from collections.abc import Callable

import dspy

from kms import config
from kms.core import content, llm, models, module, search
from kms.graph import procedures, queries, writer

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
    belong to it. ENTITY DEFINITIONS provide canonical, reusable concepts and
    terminology. They are abstraction-layer knowledge, not copied source
    passages; use them as background without inventing unsupported facts.

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
        description=(
            'Canonical, reusable concept and relation definitions relevant '
            'to the statement; learner-facing background knowledge, not raw '
            'source excerpts.'
        )
    )
    procedure: str = dspy.OutputField(
        description=(
            'The complete correct solution in continuous prose with '
            'Markdown LaTeX: `$...$` inline and `$$...$$` display.'
        )
    )


class ProcedureWriter(module.Module):
    """Writes solution-level learning from a statement and canonical context."""

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

    Split the complete PROCEDURE below into an ordered list of learnable
    stages. The STATEMENT gives the learner's goal. ENTITY DEFINITIONS give
    canonical, reusable concepts and relations that support the reasoning;
    they are background abstractions, not raw source excerpts. The procedure
    may be a proof, solution, derivation, calculation, or other working-out.

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

    statement: content.ContentParts = dspy.InputField(
        description='The statement and any figures it contains.'
    )
    procedure: content.ContentParts = dspy.InputField(
        description='The complete procedure and any figures it contains.'
    )
    entity_definitions: str = dspy.InputField(
        description=(
            'Canonical, reusable concept and relation descriptions relevant '
            'to the procedure.'
        )
    )
    steps: list[str] = dspy.OutputField(
        description=(
            'The complete procedure copied into coherent, ordered '
            'pedagogical stages; never a summary or final answer only.'
        )
    )


class StepSplitter(module.Module):
    """Splits solution reasoning into coherent learner-facing stages."""

    signature = StepSplitterSignature
    record_name = 'step_splitter'

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__(language_model or llm.module_lm('procedure_creator'))

    def encode(
        self,
        statement: content.Content,
        procedure: content.Content,
        entity_definitions: str,
    ) -> dict:
        """Builds the step-splitter kwargs for one procedure."""
        return {
            'statement': content.ContentParts(content=statement),
            'procedure': content.ContentParts(content=procedure),
            'entity_definitions': entity_definitions,
        }

    def decode(self, prediction, **inputs) -> list[str]:
        """Returns coherent stages that teach the procedure's reasoning."""
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


async def _split_procedure(
    splitter: StepSplitter,
    statement_parts: content.Content,
    procedure_parts: content.Content,
    entity_definitions: str,
) -> list[str]:
    """Splits a procedure with its statement and search context."""
    return await splitter.aforward(
        statement=statement_parts,
        procedure=procedure_parts,
        entity_definitions=entity_definitions,
    )


async def _create_generated_procedure(
    statement_uuid: str,
    source: str,
    statement_parts: content.Content,
    entity_definitions: str,
    writer_module: ProcedureWriter,
    splitter: StepSplitter,
    session_factory: Callable,
) -> list[str]:
    """Writes, splits, persists, and links a missing procedure."""
    procedure_text = await writer_module.aforward(
        parts=statement_parts, entity_definitions=entity_definitions
    )
    if not procedure_text.strip():
        return []

    steps = await _split_procedure(
        splitter,
        statement_parts,
        content.Content.from_text(procedure_text),
        entity_definitions,
    )
    if not steps:
        return []

    procedure = models.Procedure(
        block=[],
        index=0,
        statement_uuid=statement_uuid,
        steps=[
            models.Step(text=text, index=index)
            for index, text in enumerate(steps)
        ],
    )
    await writer.persist_procedures(
        [procedure], source, session_factory=session_factory
    )

    procedure_uuid_value = procedures.procedure_uuid(
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
    return steps


async def create_procedures(
    session_factory: Callable,
    *,
    language_model: dspy.LM | None = None,
    top_k: int | None = None,
) -> int:
    """Materializes attached procedures and creates missing procedures.

    Existing procedures with source members are split into learnable steps.
    Canonical hub descriptions supply reusable background knowledge; the
    procedure and its steps supply task-specific application and reasoning.
    Statements without a procedure are judged, enriched with canonical search
    results, and given a newly written procedure when they need one.

    Args:
        session_factory: Async callable returning a Neo4j session.
        language_model: LM for judging, writing, and splitting.
        top_k: Number of entity definitions to retrieve per statement.

    Returns:
        The number of procedures materialized or created.
    """
    if top_k is None:
        top_k = config.get_settings().stages.procedure.entity_definition_top_k
    work_items = await queries.statement_procedure_work_items(session_factory)
    if not work_items:
        logger.info('No statements found.')
        return 0

    statements: dict[str, dict] = {}
    for work_item in work_items:
        statement = statements.setdefault(
            work_item['statement_uuid'],
            {'source': work_item['source'], 'procedures': []},
        )
        if work_item['procedure_uuid'] is not None:
            statement['procedures'].append(work_item)

    judge = NeedsJudge(language_model)
    writer_module = ProcedureWriter(language_model)
    splitter = StepSplitter(language_model)
    processed = 0

    for statement_uuid, statement_data in statements.items():
        source = statement_data['source']
        composed_statement = await queries.compose_statement(
            statement_uuid, session_factory
        )
        statement_text = composed_statement['text']
        if not statement_text.strip():
            logger.debug('Skipping empty statement %s', statement_uuid)
            continue

        statement_parts = content.Content.from_text_and_pictures(
            statement_text, composed_statement.get('pictures', [])
        )
        attached_procedures = statement_data['procedures']
        if not attached_procedures:
            needs_procedure = await judge.aforward(parts=statement_parts)
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
        entity_definitions = _format_entity_definitions(entity_groups)

        if not attached_procedures:
            steps = await _create_generated_procedure(
                statement_uuid,
                source,
                statement_parts,
                entity_definitions,
                writer_module,
                splitter,
                session_factory,
            )
            if steps:
                processed += 1
                logger.info(
                    'Created procedure for statement %s with %d steps.',
                    statement_uuid,
                    len(steps),
                )
            continue

        for procedure_data in attached_procedures:
            if procedure_data['has_steps']:
                continue
            composed_procedure = await queries.compose_procedure(
                procedure_data['procedure_uuid'], session_factory
            )
            if not composed_procedure['member_count']:
                logger.warning(
                    'Procedure %s has neither steps nor members.',
                    procedure_data['procedure_uuid'],
                )
                continue

            procedure_parts = content.Content.from_text_and_pictures(
                composed_procedure['text'],
                composed_procedure.get('pictures', []),
            )
            steps = await _split_procedure(
                splitter,
                statement_parts,
                procedure_parts,
                entity_definitions,
            )
            if not steps:
                logger.warning(
                    'Procedure %s produced no steps.',
                    procedure_data['procedure_uuid'],
                )
                continue
            await writer.persist_procedure_steps(
                procedure_data['procedure_uuid'],
                [
                    models.Step(text=text, index=index)
                    for index, text in enumerate(steps)
                ],
                source,
                session_factory=session_factory,
            )
            processed += 1
            logger.info(
                'Materialized procedure %s with %d steps.',
                procedure_data['procedure_uuid'],
                len(steps),
            )

    return processed
