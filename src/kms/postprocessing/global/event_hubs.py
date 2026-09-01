"""Build cross-source global EventHubs from local EventHubs."""

import dspy
from pydantic import BaseModel

from kms import config
from kms.core import embeddings, llm, models, module
from kms.graph import queries, writer


class EventHubDefinition(BaseModel):
    """Canonical event concept synthesized from event evidence."""

    canonical_name: str
    description: str


class EventHubSynthesisSignature(dspy.Signature):
    r"""
    Synthesize one canonical event concept from source-grounded event
    mentions that have already been grouped as equivalent.

    Name the occurrence or event itself, not its consequence, motivation,
    property, result, or surrounding explanatory clause. Preserve the
    event's participants and role meaning in the description. Do not merge
    a battle with its victory, an appointment with the person appointed, or
    an event with a later consequence. Generalize only the common event
    supported by every supplied mention. Never invent time, participants,
    outcomes, or causal claims.
    """

    request: models.HubSynthesisInput = dspy.InputField()
    result: EventHubDefinition = dspy.OutputField()


class EventHubAdjudicationSignature(dspy.Signature):
    r"""
    Decide whether two source mentions refer to the same event concept.

    Return TRUE only when both mentions describe the same kind of occurrence
    with compatible trigger/meaning and compatible participants or roles.
    Return FALSE when one is a consequence, result, motivation, condition,
    property, state, or explanation of the other. Return FALSE when one event
    is merely related to, precedes, follows, causes, or is caused by the
    other. In particular, a battle and its victory are distinct events, as
    are an approach and the turning point it produces. Shared verbs or
    topical context are insufficient. Do not use outside knowledge.
    """

    comparison: models.HubMentionComparisonInput = dspy.InputField()
    result: models.MergeDecision = dspy.OutputField()


class EventHubSynthesizer(module.Module):
    """Global event-tailored canonical event synthesis."""

    signature = EventHubSynthesisSignature
    record_name = 'event_hub_synthesizer'

    def __init__(
        self, language_model: dspy.LM | None = None, recorder=None
    ) -> None:
        module.Module.__init__(
            self,
            language_model or llm.module_lm(self.record_name),
            recorder=recorder,
        )

    def encode(
        self, surface_forms: list[str], descriptions: list[str], scope: str
    ) -> dict:
        return {
            'request': models.HubSynthesisInput(
                surface_forms=surface_forms,
                descriptions=descriptions,
                scope=scope,
            )
        }

    def decode(self, prediction, **inputs) -> tuple[str, str]:
        result = EventHubDefinition.model_validate(prediction.result)
        return (
            module.require_text(result.canonical_name, 'canonical_name'),
            module.require_text(result.description, 'description'),
        )


class EventHubAdjudicator(module.Module):
    """Global event equivalence adjudication."""

    signature = EventHubAdjudicationSignature
    record_name = 'event_hub_adjudicator'

    def __init__(
        self, language_model: dspy.LM | None = None, recorder=None
    ) -> None:
        module.Module.__init__(
            self,
            language_model or llm.module_lm(self.record_name),
            recorder=recorder,
        )

    def encode(
        self,
        left: models.HubMentionInput,
        right: models.HubMentionInput,
        scope: str,
    ) -> dict:
        return {
            'comparison': models.HubMentionComparisonInput(
                left=left, right=right, scope=scope
            )
        }

    def decode(self, prediction, **inputs) -> bool:
        result = models.MergeDecision.model_validate(prediction.result)
        return module.require_bool(result.should_merge, 'should_merge')


def _local_hub_records(rows: list[dict]) -> list[dict]:
    return [
        {
            'uuid': row['uuid'],
            'name': row['canonical_name'],
            'aliases': row.get('aliases') or [],
            'description': row.get('description'),
            'embedding': row.get('embedding'),
            'source': row.get('source'),
        }
        for row in rows
    ]


async def _build_global_event_hubs(
    records: list[dict],
    *,
    adjudicator,
    synthesizer,
    max_concurrency: int | None,
) -> dict:
    """Group event hubs with global event-boundary adjudication."""
    if not records:
        return {'clusters': 0, 'records': 0, 'hubs': []}
    stage = config.get_settings().stages.event_hubs
    if any(not record.get('embedding') for record in records):
        raise RuntimeError('event hubs: records must have embeddings')
    adjacency = {index: [] for index in range(len(records))}
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if (
                embeddings.cosine_similarity(
                    records[left]['embedding'], records[right]['embedding']
                )
                >= stage.recall_threshold
            ):
                adjacency[left].append(right)
                adjacency[right].append(left)
    components = []
    visited = set()
    for start in range(len(records)):
        if start in visited:
            continue
        component = []
        stack = [start]
        while stack:
            index = stack.pop()
            if index in visited:
                continue
            visited.add(index)
            component.append(records[index])
            stack.extend(adjacency[index])
        components.append(component)
    gate = llm.gate(max_concurrency)
    clusters = []
    for component in components:
        parent = list(range(len(component)))

        def find(index: int, parent=parent) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left in range(len(component)):
            for right in range(left + 1, len(component)):
                score = embeddings.cosine_similarity(
                    component[left]['embedding'], component[right]['embedding']
                )
                merge = score >= stage.merge_above
                if stage.separate_below < score < stage.merge_above:
                    async with gate:
                        merge = await adjudicator.aforward(
                            left=models.HubMentionInput.from_record(
                                component[left]
                            ),
                            right=models.HubMentionInput.from_record(
                                component[right]
                            ),
                            scope='Compare event mentions by event meaning, trigger, and roles.',
                        )
                if merge:
                    left_root, right_root = find(left), find(right)
                    if left_root != right_root:
                        parent[right_root] = left_root
        grouped = {}
        for index, record in enumerate(component):
            grouped.setdefault(find(index), []).append(record)
        clusters.extend(grouped.values())
    definitions = []
    for cluster in clusters:
        forms = list(
            dict.fromkeys(
                form
                for member in cluster
                for form in [member['name'], *member.get('aliases', [])]
                if form
            )
        )
        descriptions = sorted(
            {
                member['description']
                for member in cluster
                if member.get('description')
            }
        )
        async with gate:
            name, description = await synthesizer.aforward(
                surface_forms=forms,
                descriptions=descriptions,
                scope='Synthesize a cross-source semantic event hub.',
            )
        definitions.append({'canonical_name': name, 'description': description})
    vectors = await embeddings.embedder().embed(
        [
            f'{definition["canonical_name"]}: {definition["description"]}'
            for definition in definitions
        ]
    )
    hubs = []
    for cluster, definition, vector in zip(
        clusters, definitions, vectors, strict=True
    ):
        members = [member['uuid'] for member in cluster]
        hubs.append(
            {
                'uuid': _global_hub_uuid(members),
                'source': None,
                'canonical_name': definition['canonical_name'],
                'aliases': sorted(
                    {
                        alias
                        for member in cluster
                        for alias in [
                            member['name'],
                            *member.get('aliases', []),
                        ]
                        if alias
                    }
                ),
                'description': definition['description'],
                'embedding': vector,
                'members': members,
            }
        )
    return {'clusters': len(clusters), 'records': len(records), 'hubs': hubs}


def _global_hub_uuid(members: list[str]) -> str:
    from uuid import NAMESPACE_URL, uuid5

    return uuid5(
        NAMESPACE_URL, f'meta#event_hub#{"|".join(sorted(members))}'
    ).hex


async def align_global_hubs(
    *,
    session_factory,
    language_model,
    adjudicator,
    synthesizer,
    max_concurrency: int | None = None,
) -> dict:
    """Align local event hubs into deterministic cross-source EventHubs."""
    rows = await queries.all_event_source_hubs(session_factory)
    records = _local_hub_records(rows)
    sources = {
        record.get('source') for record in records if record.get('source')
    }
    if len(sources) < 2:
        return {'aligned': 0, 'new_hubs': 0}
    result = await _build_global_event_hubs(
        records,
        adjudicator=adjudicator,
        synthesizer=synthesizer,
        max_concurrency=max_concurrency,
    )
    result['hubs'] = [
        hub
        for hub in result['hubs']
        if len(
            {
                member.get('source')
                for member in records
                if member['uuid'] in hub.get('members', [])
            }
        )
        >= 2
    ]
    await writer.clear_event_global_hubs(session_factory=session_factory)
    await writer.persist_event_hubs(
        result['hubs'], session_factory=session_factory, tier='meta'
    )
    assignments = [
        {'source_hub': member, 'meta_hub': hub['uuid']}
        for hub in result['hubs']
        for member in hub.get('members', [])
    ]
    await writer.attach_event_global_hubs(
        assignments, aliases=[], session_factory=session_factory
    )
    return {'aligned': len(assignments), 'new_hubs': len(result['hubs'])}
