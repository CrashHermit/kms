r"""
Entity attribution — the Stage 2 pass that fills each entity's role attributes (the
AutoMathKG `bodylist` action labels) on top of the Stage 1 groupings.

For every entity it labels each member node with the role it plays and lifts the
`number`/`instruction` metadata up from the member nodes:

  * Theorem  — the LLM splits members into `statement` vs `proof` (repeatable).
  * Problem  — the LLM splits a gathered worked example into `statement` vs `solution`
               (repeatable); an atomic 1:1-wrapped exercise is a single `statement`.
  * Definition — every member is `statement` (mechanical, no LLM).

The statement→proof/solution boundary is usually an explicit "Proof"/"Solution" marker
node, so collect splits there deterministically with no LLM (this is reliable where the
LLM was not). The LLM is only used for a multi-member Theorem/Problem that has no such
marker; Definitions and single-member entities are all `statement`. As in the
governor/grouper, the LLM never touches ids: it sees member contents in order and
returns a role per member, which collect zips back onto the entity's members.
"""

import re

import dspy
from langgraph.types import Send

from .state import State, EntityType, EntityRole
from .llm import text_lm
from . import capture

# Roles the LLM may assign, by entity type. The statement comes first; the secondary
# role (proof for theorems, solution for problems) covers the rest.
_SECONDARY = {EntityType.THEOREM: EntityRole.PROOF, EntityType.PROBLEM: EntityRole.SOLUTION}


# A node that opens with "Proof." / "Solution:" and keeps going — the run-in proof/
# solution style (e.g. Judson's "*Proof.* Suppose that ..."), as opposed to a bare
# "### Solution" heading node.
_RUN_IN_MARKER = re.compile(r"^(?:proof|solution)\b\s*[.:]", re.IGNORECASE)


def _is_marker(content: str | None) -> bool:
    """True if a node marks the boundary between an entity's statement and its
    proof/solution — either a bare 'Proof'/'Solution' heading node, or a node that
    opens run-in with 'Proof.'/'Solution:'. In both cases everything before this node
    is the statement and this node begins the proof/solution."""
    if not content:
        return False
    # Strip leading heading/emphasis markup so the first word is visible.
    stripped = content.strip().lstrip("#").strip().lstrip("*").strip()
    lowered = stripped.lower()
    # Bare heading: the node is just "Proof"/"Solution" (+ optional trailing markup).
    bare = lowered.rstrip("*").strip().rstrip(".:").strip()
    if bare in ("proof", "solution"):
        return True
    # Run-in: the node opens with "Proof."/"Solution:" and continues.
    return bool(_RUN_IN_MARKER.match(lowered))


def _marker_index(entity, node_by_id) -> int | None:
    """Position of the first member that is an explicit Proof/Solution marker, provided
    it has statement nodes before it. This gives a deterministic statement→proof/solution
    split (everything before the marker is statement, the marker and after are the
    secondary role) so the common case needs no LLM. Returns None when there is no such
    marker, leaving the judgment to the LLM."""
    for i, member in enumerate(entity.members):
        node = node_by_id.get(member.node_id)
        if node is not None and _is_marker(node.content):
            return i if i > 0 else None
    return None


class Signature(dspy.Signature):
    r"""
    Label the role each member node plays within a single mathematical entity.

    You are given the entity's type and its member nodes, in document order. Split the
    members into an opening STATEMENT run followed by a proof/solution run. The split
    point is the explicit "Proof" / "Solution" marker: every node up to (but NOT
    including) that marker is `statement`; the marker node and every node after it are
    the proof/solution.

    - For a `theorem`: `statement` covers the theorem's label/title heading AND its
      claim — every node before the "Proof" heading. `proof` covers the "Proof" heading
      and every node of the proof after it. A theorem may have several proof nodes, or
      none at all (then every member is `statement`).
    - For a `problem` (a worked example): `statement` covers the example's label/title
      heading AND the question being posed — every node before the "Solution" heading.
      `solution` covers the "Solution" heading and every node after it. A problem may
      have several solution nodes.

    The statement usually spans MULTIPLE nodes — commonly a label node ("Example 2.30"),
    a title node ("Classifying a Discontinuity"), and the actual claim/question node.
    Do NOT stop at the first node: keep labelling `statement` until you reach the
    Proof/Solution marker. If there is no explicit marker, judge where the claim or
    question ends and its justification begins.

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
        """Fan out one worker only for entities whose statement→proof/solution split
        can't be found mechanically — a multi-member Theorem/Problem with no explicit
        Proof/Solution marker. Marker-delimited entities are split deterministically in
        collect; Definitions and single-member entities are all `statement`."""
        node_by_id = {n.id: n for n in state.get("nodes", [])}
        sends: list[Send] = []
        for entity in state.get("entities", []):
            if (
                entity.type in _SECONDARY
                and len(entity.members) > 1
                and _marker_index(entity, node_by_id) is None
            ):
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
            secondary = _SECONDARY.get(entity.type)
            split = _marker_index(entity, node_by_id) if secondary else None
            labels = roles_by_entity.get(entity.id)
            if split is not None:
                # Deterministic split at the Proof/Solution marker: statement before it,
                # the secondary role from it onward.
                for i, member in enumerate(entity.members):
                    member.role = EntityRole.STATEMENT if i < split else secondary
            elif labels is not None and len(labels) == len(entity.members):
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

            # Capture a role-labelling example (inputs = type + member contents, output
            # = the final roles). The roles are post-split, so marker-delimited entities
            # yield high-quality labels for the role signature — not just LLM calls.
            if capture.enabled() and secondary and len(entity.members) > 1:
                contents = [
                    (node_by_id[m.node_id].content or "")
                    for m in entity.members if m.node_id in node_by_id
                ]
                if len(contents) == len(entity.members):
                    capture.record(
                        "entity_attributor",
                        {"entity_type": entity.type.value, "members": contents},
                        {"roles": [m.role.value for m in entity.members if m.role]},
                    )

        return {"entities": state.get("entities", [])}
