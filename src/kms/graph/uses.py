"""
Graph representation of step-level references — the ``:USES`` edges (see ``docs/UNIFIED-KG.md``,
"Edges (math-first)").

A reference is, physically, invoked at a *step*: "By the Mean Value Theorem, …" happens in one line
of a proof. AutoMathKG (and this pipeline's referencer) records references at the *entity* level, and
the entity-level ``(:Entity)-[:REFERENCES]->(:Canonical)`` edge stays as the **rollup / floor**. This
module adds the finer ``(:Event)-[:USES {tactic}]->(:Entity:Canonical)`` edge on top, locating which
proof step invoked which reference.

How (math-first v1): a **deterministic name match**. For each of the entity's refs, wherever the
target name appears (as a whole word, case-insensitive) in a proof step's text, a ``:USES`` edge is
drawn from that step's ``:Event`` to the reference's canonical. This is a cheap heuristic — the same
spirit as the name-normalized canonical — and it is purely additive: a ref whose target isn't located
in any step simply has no ``:USES`` edge and is still covered by the entity-level ``:REFERENCES``
rollup. A later stage can replace the match with a proper per-step extraction (the referencer running
per-event); the edge shape doesn't change, only how the rows are produced.

Only *proof* steps are considered — solutions have no ``bodylist``/events in the model, and statement
structure isn't reified into events — so a Problem's refs stay entity-level only for now.
"""

import re

from kms.core.models import Entity
from kms.graph.procedures import proof_events
from kms.graph.references import canonical_uuid


def _mentions(text: str, target: str) -> bool:
    """True if ``target`` appears in ``text`` as a whole token (case-insensitive), so "Set" matches
    "Set" and "a Set" but not "subset" / "Reset". Whitespace in the target is collapsed; an empty
    target never matches."""
    needle = " ".join(target.split())
    if not needle:
        return False
    pattern = rf"(?<![0-9A-Za-z]){re.escape(needle)}(?![0-9A-Za-z])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def uses_rows(entities: list[Entity], source: str) -> list[dict]:
    """The ``{event, canonical, tactic}`` rows for the ``:USES`` edges: one per (proof step, reference)
    where the reference's target name is mentioned in the step's text, de-duplicated by (event,
    canonical). The event uuid matches the ``:Event`` the procedural layer wrote; the canonical uuid
    matches the ``:Entity:Canonical`` the reference layer wrote."""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        if not entity.refs:
            continue
        for event, step in proof_events(entity, source):
            for ref in entity.refs:
                if not _mentions(step.description, ref.target):
                    continue
                canonical = canonical_uuid(ref.kind, ref.target)
                key = (event, canonical)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"event": event, "canonical": canonical, "tactic": ref.tactic})
    return rows
