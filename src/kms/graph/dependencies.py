"""
Graph representation of the concept prerequisite layer — the ``:DEPENDS_ON`` edges between concepts
(see ``docs/GENERALIZATION.md``, "Drop :BROADER / MSC; add :DEPENDS_ON").

``(:Concept)-[:DEPENDS_ON]->(:Concept)`` reads "you need the target to define/prove/understand the
source". It is what the design chose *instead* of a taxonomic ``:BROADER`` edge, for three reasons
worth restating where the code lives:

* It answers the actual goal — curriculum / prerequisite sequencing — that an "is-a-kind-of"
  hierarchy only approximates.
* It is **groundable**: it is the concept-level rollup of the entity-level ``:REFERENCES`` graph the
  referencer already extracts, so the graph converges on ONE relationship — dependency — at two
  granularities, mirroring the existing ``:REFERENCES`` (entity) vs ``:USES`` (step) pattern.
* ``:BROADER`` was an imported MSC taxonomy — math-only by construction, and therefore exactly the
  kind of domain vocabulary the generalization removes. The flat multi-tag concepts
  (``graph.concepts``) already cover the categorization it would have served.

This module is the pure mapping half: it turns the dependency finder's ``core.models.Dependency``
records into edge rows keyed on the same global concept uuids ``graph.concepts`` mints, so a
prerequisite edge lands on the very vertex an ``:INSTANCE_OF`` edge points at. The driver lives in
``graph.db`` and the writes in ``graph.writer``.

Both endpoints are MATCHed, never minted: a dependency naming a concept no entity instantiates draws
no edge. That keeps concept minting the conceptualizer's sole job and stops a prerequisite judgment
from inventing vertices the corpus has no evidence for.
"""

from kms.core import models
from kms.graph.concepts import TOPIC_CONCEPT, concept_uuid, normalize_concept


def dependency_rows(dependencies: list[models.Dependency]) -> list[dict]:
    """The ``{dependent, prerequisite, support}`` uuid rows for the ``:DEPENDS_ON`` edges, one per
    dependency, de-duplicated by the (dependent, prerequisite) concept pair — the first row for a
    pair wins, which is the highest-support one because the dependency finder emits in support order.
    A self-dependency (both ends normalizing to the same concept) is dropped: it would be a loop, and
    a prerequisite graph must stay acyclic."""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for dependency in dependencies:
        dependent = normalize_concept(dependency.dependent)
        prerequisite = normalize_concept(dependency.prerequisite)
        if not dependent or not prerequisite or dependent == prerequisite:
            continue
        key = (dependent, prerequisite)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                'dependent': concept_uuid(TOPIC_CONCEPT, dependency.dependent),
                'prerequisite': concept_uuid(
                    TOPIC_CONCEPT, dependency.prerequisite
                ),
                'support': dependency.support,
            }
        )
    return rows
