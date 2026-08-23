"""Assigns instruction governance to statement-level construction units."""

import asyncio
import logging

from kms import config
from kms.construction import governance_judge
from kms.core import content, models, state, walker

logger = logging.getLogger(__name__)


def _compose_nodes(nodes: list[models.Node]) -> content.Content:
    """Composes ordered nodes into multimodal content."""
    if not nodes:
        return content.Content.from_text('')

    parts: list[content.TextPart | content.ImagePart] = []
    for node in nodes:
        if node.image_path:
            image = content.load_image(
                node.image_path,
                max_dim=config.get_settings().image.max_dim,
            )
            if image:
                parts.append(content.ImagePart(image=image))
        if node.content:
            parts.append(content.TextPart(text=node.content))
    return content.Content(parts=parts)


def _member_nodes(
    member_positions: list[int], nodes: list[models.Node]
) -> list[models.Node]:
    """Returns existing member nodes in the unit's declared order."""
    return [nodes[position] for position in member_positions]


def compose_instruction_content(
    instruction: models.Instruction,
    nodes: list[models.Node],
) -> content.Content:
    """Composes an instruction's member nodes into multimodal content."""
    return _compose_nodes(_member_nodes(instruction.member_positions, nodes))


def compose_statement_content(
    statement: models.Statement,
    nodes: list[models.Node],
) -> content.Content:
    """Composes a statement's complete member content."""
    return _compose_nodes(_member_nodes(statement.member_positions, nodes))


def _positions(
    member_positions: list[int], nodes: list[models.Node]
) -> list[int]:
    """Returns stream positions for member positions."""
    return member_positions


def _span_positions(
    member_positions: list[int], nodes: list[models.Node]
) -> tuple[int, int] | None:
    """Returns the inclusive stream span for a set of member positions."""
    if not member_positions:
        return None
    return min(member_positions), max(member_positions)


def _marked_statement_window(
    nodes: list[models.Node],
    target_positions: list[int],
    backward_budget: int,
    forward_budget: int,
) -> list[walker.WindowNode]:
    """Builds a static context window around statement members."""
    return walker.marked_window(
        nodes,
        target_positions,
        backward_budget=backward_budget,
        forward_budget=forward_budget,
        marker='statement',
    )


class GovernanceStatementWalkerNode:
    """Assigns instruction governance by walking statements in document order."""

    def __init__(
        self,
        judge: governance_judge.GovernanceJudge,
        backward_budget: int | None = None,
        forward_budget: int | None = None,
        threshold: float = 0.5,
        max_concurrent_calls: int | None = None,
    ) -> None:
        governance_config = config.get_settings().stages.governance
        self.judge = judge
        self.backward_budget = (
            backward_budget
            if backward_budget is not None
            else governance_config.backward_context_budget
        )
        self.forward_budget = (
            forward_budget
            if forward_budget is not None
            else governance_config.forward_context_budget
        )
        self.threshold = threshold
        self.max_concurrent_calls = (
            max_concurrent_calls
            if max_concurrent_calls is not None
            else governance_config.max_concurrent_calls
        )

    async def _judge_statement(
        self,
        instruction_content: content.Content,
        statement: models.Statement,
        nodes: list[models.Node],
        start: int,
        end: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[models.Statement, bool, float]:
        """Judges one complete statement with its surrounding context."""
        statement_content = compose_statement_content(statement, nodes)
        target_positions = _positions(statement.member_positions, nodes)
        context_window = _marked_statement_window(
            nodes,
            target_positions,
            backward_budget=self.backward_budget,
            forward_budget=self.forward_budget,
        )
        async with semaphore:
            governs_result, confidence = await self.judge.acall(
                instruction_directive=instruction_content,
                statement_content=statement_content,
                context_window=context_window,
            )
        return statement, governs_result, confidence

    async def run(self, current_state: state.State) -> dict:
        """Walks statements and assigns applicable instruction identities."""
        nodes = current_state.get('nodes', [])
        instructions = current_state.get('instructions', [])
        statements = current_state.get('statements', [])

        if not instructions or not statements:
            return {'statements': statements}

        instruction_data = []
        for instruction in instructions:
            span = _span_positions(
                instruction.member_positions or instruction.block, nodes
            )
            if span is not None:
                instruction_data.append((span[0], span[1], instruction))
        instruction_data.sort(key=lambda item: item[0])

        statement_data = []
        for statement in statements:
            span = _span_positions(
                statement.member_positions or statement.block, nodes
            )
            if span is not None:
                statement_data.append((span[0], span[1], statement))
        statement_data.sort(key=lambda item: item[0])

        semaphore = asyncio.Semaphore(self.max_concurrent_calls)
        statement_cursor = 0
        evaluated = 0
        for instruction_index, (
            _instruction_start,
            instruction_end,
            instruction,
        ) in enumerate(instruction_data):
            if instruction.uuid is None:
                raise ValueError('instruction is missing its assigned uuid')

            domain_start = instruction_end + 1
            domain_end = (
                instruction_data[instruction_index + 1][0]
                if instruction_index + 1 < len(instruction_data)
                else len(nodes)
            )
            cursor = domain_start
            candidates: list[tuple[models.Statement, int, int]] = []

            while statement_cursor < len(statement_data):
                statement_start, statement_end, statement = statement_data[
                    statement_cursor
                ]
                if statement_start < cursor:
                    statement_cursor += 1
                    continue
                if statement_start >= domain_end:
                    break
                candidates.append(
                    (statement, statement_start, statement_end + 1)
                )
                cursor = statement_end + 1
                statement_cursor += 1

            if not candidates:
                continue

            instruction_content = compose_instruction_content(
                instruction, nodes
            )
            results = await asyncio.gather(
                *(
                    self._judge_statement(
                        instruction_content,
                        statement,
                        nodes,
                        start,
                        end,
                        semaphore,
                    )
                    for statement, start, end in candidates
                )
            )
            evaluated += len(results)
            for statement, governs_result, confidence in results:
                if (
                    governs_result
                    and confidence >= self.threshold
                    and instruction.uuid not in statement.instruction_uuids
                ):
                    statement.instruction_uuids.append(instruction.uuid)

        logger.info(
            'governance walker: %d instructions -> %d statements evaluated',
            len(instruction_data),
            evaluated,
        )
        return {'statements': statements}
