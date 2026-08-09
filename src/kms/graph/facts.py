from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

FACT_LABEL = 'Fact'


def fact_uuid(source: str, node_ids: list[int], index: int) -> str:
    return uuid5(
        NAMESPACE_URL, f'{source}#fact#{nodes.block_key(node_ids)}#{index}'
    ).hex


def fact_properties(fact: models.AtomicFact, source: str, index: int) -> dict:
    properties = {
        'uuid': fact_uuid(source, fact.node_ids, index),
        'source': nodes.source_uuid(source),
        'text': fact.text,
        'index': index,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def fact_rows(facts: list[models.AtomicFact], source: str) -> list[dict]:
    return [
        fact_properties(fact, source, index) for index, fact in enumerate(facts)
    ]


def evidence_pairs(facts: list[models.AtomicFact], source: str) -> list[dict]:
    return [
        {
            'node': nodes.node_uuid(source, node_id),
            'fact': fact_uuid(source, fact.node_ids, index),
        }
        for index, fact in enumerate(facts)
        for node_id in fact.node_ids
    ]

