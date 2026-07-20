r"""
Entity attribution — the Stage 2 pass that fills each entity's role attributes (the
AutoMathKG `bodylist` action labels) on top of the Stage 1 groupings.

For every entity it labels each member node with the role it plays and lifts the
`number`/`instruction` metadata up from the member nodes:

  * Theorem  — the LLM splits members into `statement` vs `proof` (repeatable).
  * Problem  — the LLM splits a gathered worked example into `statement` vs `solution`
               (repeatable); an atomic 1:1-wrapped exercise is a single `statement`.
  * Definition — every member is `statement` (mechanical, no LLM).

Only multi-member Theorems and Problems need the LLM (there is a statement/proof or
statement/solution boundary to find). Definitions and single-member entities are all
`statement` and are handled mechanically in collect. As in the governor/grouper, the
LLM never touches ids: it sees member contents in order and returns a role per member,
which collect zips back onto the entity's members.
"""

import dspy
from langgraph.types import Send

from .state import State, EntityType, EntityRole
from .llm import text_lm

# Roles the LLM may assign, by entity type. The statement comes first; the secondary
# role (proof for theorems, solution for problems) covers the rest.
_SECONDARY = {EntityType.THEOREM: EntityRole.PROOF, EntityType.PROBLEM: EntityRole.SOLUTION}


class Signature(dspy.Signature):
    r"""
    Label the role each member node plays within a single mathematical entity.

    You are given the entity's type and its member nodes, in document order. Assign
    each member exactly one role:

    - For a `theorem`: `statement` — the claim being asserted — or `proof` — a part of
      its proof. The statement comes first; the proof nodes follow it. A theorem may
      have several proof nodes (label each `proof`); it may also have none.
    - For a `problem` (a worked example): `statement` — the problem being posed — or
      `solution` — a part of the worked-out solution. The statement comes first; the
      solution nodes follow. A problem may have several solution nodes.

    Return `roles`: one role per member, in the SAME order and the SAME length as the
    members given.
    """

    entity_type: str = dspy.InputField(description="The entity's type: 'theorem' or 'problem'.")
    members: list[str] = dspy.InputField(
        description="The member node contents, in document order."
    )
    roles: list[str] = dspy.OutputField(
        description="One role per member, in order, same length as members."
    )


class Module(dspy.Module):
    def __init__(self, lm: dspy.LM | None = None):
        super().__init__()
        self.labeler = dspy.ChainOfThought(Signature)
        self.set_lm(lm or text_lm())

    async def aforward(self, entity_type: EntityType, members: list[str]) -> list[str]:
        result = await self.labeler.acall(entity_type=entity_type.value, members=members)
        secondary = _SECONDARY[entity_type]
        allowed = {EntityRole.STATEMENT.value, secondary.value}

        def positional(i: int) -> str:
            # Safe fallback: first member is the statement, the rest take the secondary
            # role (proof/solution) — the usual statement-then-proof/solution layout.
            return EntityRole.STATEMENT.value if i == 0 else secondary.value

        roles = list(result.roles or [])
        # On any length mismatch, fall back entirely to the positional layout rather
        # than misalign the role↔member mapping.
        if len(roles) != len(members):
            return [positional(i) for i in range(len(members))]
        return [
            r.strip().lower() if (r or "").strip().lower() in allowed else positional(i)
            for i, r in enumerate(roles)
        ]


class EntityAttributorNode:
    def __init__(self, module: Module | None = None):
        self.module = module or Module()

    def dispatch(self, state: State) -> list[Send] | str:
        """Fan out one worker per entity that has a role boundary to find — a Theorem
        or Problem with more than one member. Definitions and single-member entities
        are all `statement`, handled mechanically in collect."""
        node_by_id = {n.id: n for n in state.get("nodes", [])}
        sends: list[Send] = []
        for entity in state.get("entities", []):
            if entity.type in _SECONDARY and len(entity.members) > 1:
                contents = [
                    (node_by_id[m.node_id].content or "")
                    for m in entity.members if m.node_id in node_by_id
                ]
                sends.append(Send("entity_attributor_worker", {
                    "entity_id": entity.id,
                    "entity_type": entity.type,
                    "members": contents,
                }))
        return sends or "entity_attributor_collect"

    async def worker(self, state: dict) -> dict:
        """Label one entity's members with statement/proof or statement/solution."""
        roles = await self.module.aforward(state["entity_type"], state["members"])
        return {"attribute_results": [(state["entity_id"], roles)]}

    def collect(self, state: State) -> dict:
        """Apply the LLM's role labels to the entities that got them, label everything
        else `statement` mechanically, and lift number/instruction from member nodes."""
        node_by_id = {n.id: n for n in state.get("nodes", [])}
        roles_by_entity = dict(state.get("attribute_results", []))

        for entity in state.get("entities", []):
            labels = roles_by_entity.get(entity.id)
            if labels is not None and len(labels) == len(entity.members):
                for member, role in zip(entity.members, labels):
                    member.role = EntityRole(role)
            else:
                # Definitions, single-member entities, or a missing result: all statement.
                for member in entity.members:
                    member.role = EntityRole.STATEMENT

            # Lift metadata off the member nodes. Only atomic problem nodes carry a
            # number/instruction (set by the refiner/governor); other nodes carry None,
            # so this is a no-op for definitions, theorems, and gathered examples.
            if entity.members:
                first = node_by_id.get(entity.members[0].node_id)
                if first is not None:
                    entity.number = first.number
                    entity.instruction = first.instruction

        return {"entities": state.get("entities", [])}
