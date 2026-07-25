r"""
Dependency finder — the concept-level prerequisite graph, ``(:Concept)-[:DEPENDS_ON]->(:Concept)``.

This is the edge the design chose *instead* of a taxonomic ``:BROADER`` hierarchy
(``docs/GENERALIZATION.md``, "Drop :BROADER / MSC; add :DEPENDS_ON"): "you need Y to define/prove/
understand X" answers the curriculum question a taxonomy only approximates, and — unlike an imported
MSC tree — it is *groundable* in what the pipeline already extracts.

Three commitments, each of them a finding from the probes rather than a preference:

1. **Grounded, not free-listed.** Candidate pairs are not invented by asking a model "what does
   eigenvalue depend on?" — batch free-listing drifted to *related* concepts and sometimes named a
   co-concept (``eigenvector`` for ``eigenvalue``) as a prerequisite. Instead the candidates are the
   concept-level rollup of the entity-level ``:REFERENCES`` graph: entity A cites entity B, so A's
   concepts are candidate dependents of B's concepts. The graph thus converges on ONE relationship —
   dependency — at two granularities, mirroring ``:REFERENCES`` (entity) vs ``:USES`` (step).
2. **Pairwise judgment.** Every candidate is judged on its own, with the edge defined *strictly* as a
   definitional prerequisite. Pairwise judgments were crisp where free-listing was not, including
   necessity reasoning ("compactness depends on metric space?" → no, topological spaces suffice).
3. **Cycle-guarded.** Co-defined concepts (eigenvalue ↔ eigenvector) can each look like the other's
   prerequisite. A prerequisite graph that loops is useless for sequencing, so edges are admitted
   best-evidenced first and any edge that would close a cycle is dropped.

A reference target is resolved to an in-corpus entity by the same cheap **nominal** title match the
``:REALIZES`` layer uses (``graph.realizes``) — a citation to "vector space" is grounded by the book's
own definition of it. A citation to something the book never defines contributes no candidate; that is
the honest outcome, not a gap to paper over with a guess.

Entry point ``find_dependencies(entities, module)`` (async): returns the judged, cycle-guarded
``models.Dependency`` list. Persistence-agnostic — turning them into edges is the entity persister's job.
"""

import asyncio

import dspy

from kms.core import llm, models, state

# How deep into each side's concept list a candidate pair may draw. The conceptualizer emits
# specific-first, so the head of the list carries the dependency signal; the tail is the broad field
# tag, which pairs into near-vacuous candidates ("algebra depends on algebra") and multiplies the
# judgment cost for nothing.
PAIR_DEPTH = 2

# Ceiling on how many distinct candidate pairs are judged, best-evidenced first. A resource limit at
# the edge of the system (one LLM call per pair), not part of the rule.
MAX_CANDIDATES = 400


class JudgeDependency(dspy.Signature):
    r"""
    Decide whether one concept is a PREREQUISITE of another: is `prerequisite` something a reader
    must already understand in order to define, state, or prove `dependent`?

    Answer yes ONLY for a genuine definitional or logical prerequisite — the concept the definition
    of `dependent` (or its proof) is BUILT ON. Ask: could you define or establish `dependent`
    without `prerequisite`? If yes, the answer is NO.

    Answer NO for:
      * merely RELATED or adjacent concepts, or two concepts that just appear together;
      * a CO-DEFINED partner — a concept introduced alongside `dependent` in the same breath
        (eigenvalue and eigenvector define each other; neither is the other's prerequisite);
      * a SPECIALIZATION or example of `dependent` (a square matrix is not a prerequisite of a
        matrix);
      * the REVERSE direction — if `dependent` is what `prerequisite` is built on, answer NO;
      * a stronger structure than needed. Compactness does not depend on "metric space": it is
        defined for topological spaces, and a metric space is merely one setting where it appears.
        Name the weakest thing that is actually required.

    Domain-neutral: judge physics, biology, or mathematics concepts the same way, by what the
    subject itself requires.
    """

    dependent: str = dspy.InputField(
        description='The concept that might require the other.'
    )
    prerequisite: str = dspy.InputField(
        description='The concept that might be required.'
    )
    depends: bool = dspy.OutputField(
        description='True only if `prerequisite` must be understood first to define/prove `dependent`.'
    )


class Module(dspy.Module):
    """Runs the pairwise prerequisite judgment for one candidate pair."""

    def __init__(self, language_model: dspy.LM | None = None) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(JudgeDependency)
        self.set_lm(language_model or llm.text_lm())

    async def depends(self, dependent: str, prerequisite: str) -> bool:
        """True when `prerequisite` must be understood first to define/prove `dependent`."""
        result = await self.judge.acall(
            dependent=dependent, prerequisite=prerequisite
        )
        return bool(result.depends)


def _normalize(name: str) -> str:
    """The clustering key for a concept or title: lowercased, whitespace-collapsed. Deliberately the
    same normalization the graph tier's concept and canonical uuids use, so a pair judged here keys
    onto the vertices ``graph.concepts`` minted."""
    return ' '.join((name or '').split()).lower()


def _by_title(entities: list[models.Entity]) -> dict[str, models.Entity]:
    """Index the overlay by normalized title, so a reference target resolves to the entity the book
    defines it in. First mention wins; untitled entities are skipped."""
    index: dict[str, models.Entity] = {}
    for entity in entities:
        key = _normalize(entity.title or '')
        if key:
            index.setdefault(key, entity)
    return index


def candidate_pairs(
    entities: list[models.Entity], depth: int = PAIR_DEPTH
) -> list[tuple[str, str, int]]:
    """The reference-grounded candidate prerequisite pairs as ``(dependent, prerequisite, support)``,
    most-supported first.

    For every reference whose target resolves to an in-corpus entity, each of the citing entity's
    leading concepts is a candidate dependent of each of the target entity's leading concepts.
    ``support`` counts how many references grounded the pair — the evidence the cycle guard breaks
    ties on. Self-pairs (both sides naming the same concept) are dropped.
    """
    index = _by_title(entities)
    support: dict[tuple[str, str], int] = {}
    names: dict[str, str] = {}  # normalized key -> first spelling seen
    for entity in entities:
        for ref in entity.refs:
            target = index.get(_normalize(ref.target))
            if target is None or target is entity:
                continue
            for dependent in entity.concepts[:depth]:
                for prerequisite in target.concepts[:depth]:
                    a, b = _normalize(dependent), _normalize(prerequisite)
                    if not a or not b or a == b:
                        continue
                    names.setdefault(a, dependent)
                    names.setdefault(b, prerequisite)
                    support[(a, b)] = support.get((a, b), 0) + 1
    ranked = sorted(support.items(), key=lambda item: -item[1])
    return [(names[a], names[b], count) for (a, b), count in ranked]


def _acyclic(
    dependencies: list[models.Dependency],
) -> list[models.Dependency]:
    """Admit dependencies best-evidenced first, dropping any edge that would close a cycle.

    A prerequisite graph must be a DAG — a loop makes "what do I learn first?" unanswerable — and
    co-defined concepts are exactly the pair that produces one. Reachability is recomputed over the
    edges admitted so far, so the *first* (better-supported) direction of a co-defined pair survives
    and its mirror is dropped.
    """
    edges: dict[str, set[str]] = {}
    kept: list[models.Dependency] = []

    def reaches(start: str, goal: str) -> bool:
        seen, stack = {start}, [start]
        while stack:
            for nxt in edges.get(stack.pop(), ()):
                if nxt == goal:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    for dependency in dependencies:
        a = _normalize(dependency.dependent)
        b = _normalize(dependency.prerequisite)
        if reaches(b, a):  # b already needs a — adding a→b would close a cycle
            continue
        edges.setdefault(a, set()).add(b)
        kept.append(dependency)
    return kept


async def find_dependencies(
    entities: list[models.Entity],
    module: Module | None = None,
    max_candidates: int = MAX_CANDIDATES,
) -> list[models.Dependency]:
    """Judge the reference-grounded candidate pairs and return the surviving, cycle-guarded
    dependencies, best-evidenced first.

    The judgments are independent, so they run concurrently; the cycle guard is applied afterwards
    over the survivors in support order. Returns an empty list when nothing grounded a candidate —
    a corpus whose citations never resolve to its own definitions has no evidence to roll up.
    """
    candidates = candidate_pairs(entities)[:max_candidates]
    if not candidates:
        return []
    module = module or Module()
    verdicts = await asyncio.gather(
        *(
            module.depends(dependent, prerequisite)
            for dependent, prerequisite, _ in candidates
        )
    )
    judged = [
        models.Dependency(
            dependent=dependent, prerequisite=prerequisite, support=support
        )
        for (dependent, prerequisite, support), depends in zip(
            candidates, verdicts, strict=True
        )
        if depends
    ]
    return _acyclic(judged)


class DependencyFinderNode:
    """Builds the concept-level ``:DEPENDS_ON`` graph from the conceptualized overlay.

    Runs after the conceptualizer (it rolls up concepts) and before the entity persister, which
    writes the edges. Writes the ``concept_dependencies`` channel — the one channel whose unit is a
    concept rather than an entity."""

    def __init__(self, module: Module | None = None) -> None:
        self.module = module or Module()

    async def run(self, state: state.State) -> dict:
        """Judges the reference-grounded candidate pairs into the concept prerequisite graph."""
        dependencies = await find_dependencies(
            state.get('entities', []), module=self.module
        )
        return {'concept_dependencies': dependencies}
