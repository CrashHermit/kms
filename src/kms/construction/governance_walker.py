"""Assigns instruction governance to statement-level construction units."""

import asyncio
import logging

from kms import config
from kms.construction import governance_judge
from kms.core import context_window, models, state

logger = logging.getLogger(__name__)


def _member_nodes(
    member_positions: list[int], nodes: list[models.SourceNode]
) -> list[models.SourceNode]:
    """Returns existing member nodes in the unit's declared order."""
    return [nodes[position] for position in member_positions]


def _member_window(
    member_positions: list[int], nodes: list[models.SourceNode]
) -> list[context_window.ContextNode]:
    """Projects ordered source members into a local node window."""
    return context_window.project_nodes(_member_nodes(member_positions, nodes))


def _positions(
    member_positions: list[int], nodes: list[models.SourceNode]
) -> list[int]:
    """Returns stream positions for member positions."""
    return member_positions


def _span_positions(
    member_positions: list[int], nodes: list[models.SourceNode]
) -> tuple[int, int] | None:
    """Returns the inclusive stream span for a set of member positions."""
    if not member_positions:
        return None
    return min(member_positions), max(member_positions)


def _statement_context_parts(
    nodes: list[models.SourceNode],
    target_positions: list[int],
    backward_budget: int,
    forward_budget: int,
) -> tuple[list[context_window.ContextNode], list[context_window.ContextNode]]:
    """Builds directional context around explicit statement members."""
    if not target_positions:
        return [], []
    start = min(target_positions)
    end = max(target_positions)
    return (
        context_window.project_nodes(
            context_window.nodes_before(nodes, start, backward_budget)
        ),
        context_window.project_nodes(
            context_window.nodes_after(nodes, end, forward_budget)
        ),
    )


class GovernanceStatementWalkerNode:
    """Assigns instruction governance by walking statements in document order."""

    def __init__(
        self,
        judge: governance_judge.GovernanceJudge,
        backward_budget: int | None = None,
        forward_budget: int | None = None,
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
        self.max_concurrent_calls = (
            max_concurrent_calls
            if max_concurrent_calls is not None
            else governance_config.max_concurrent_calls
        )

    async def _judge_statement(
        self,
        instruction_nodes: list[context_window.ContextNode],
        statement: models.Statement,
        nodes: list[models.SourceNode],
        semaphore: asyncio.Semaphore,
    ) -> tuple[models.Statement, bool]:
        """Judges one complete statement with its surrounding context."""
        statement_nodes = _member_window(statement.member_positions, nodes)
        target_positions = _positions(statement.member_positions, nodes)
        context_before, context_after = _statement_context_parts(
            nodes,
            target_positions,
            backward_budget=self.backward_budget,
            forward_budget=self.forward_budget,
        )
        async with semaphore:
            governs_result = await self.judge.acall(
                instruction_nodes=instruction_nodes,
                context_before=context_before,
                statement_nodes=statement_nodes,
                context_after=context_after,
            )
        return statement, governs_result

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

            instruction_nodes = _member_window(
                instruction.member_positions or instruction.block, nodes
            )
            results = await asyncio.gather(
                *(
                    self._judge_statement(
                        instruction_nodes,
                        statement,
                        nodes,
                        semaphore,
                    )
                    for statement, _start, _end in candidates
                )
            )
            evaluated += len(results)
            for statement, governs_result in results:
                if (
                    governs_result
                    and instruction.uuid not in statement.instruction_uuids
                ):
                    statement.instruction_uuids.append(instruction.uuid)

        logger.info(
            'governance walker: %d instructions -> %d statements evaluated',
            len(instruction_data),
            evaluated,
        )
        return {'statements': statements}
