import json

import pytest

from kms.core import models
from kms.training import fact_dataset as fact_training


def test_load_examples_preserves_typed_dspy_shape(tmp_path):
    path = tmp_path / 'facts.jsonl'
    path.write_text(
        json.dumps(
            {
                'id': 'one',
                'source': 'book.pdf',
                'request': {
                    'context_before': [],
                    'target_node': {
                        'index': 1,
                        'node_type': 'paragraph',
                        'text': 'A fact.',
                    },
                    'context_after': [],
                },
                'facts': [{'text': 'A fact.'}],
            }
        )
        + '\n'
    )
    examples = fact_training.load_examples(path)
    assert len(examples) == 1
    assert isinstance(examples[0].request, models.FactExtractionInput)
    assert examples[0].request.target_node.text == 'A fact.'
    assert examples[0].facts == [models.AtomicFact(text='A fact.')]
    assert examples[0].inputs().toDict()['request'] == (
        examples[0].request.model_dump()
    )
def test_load_examples_rejects_missing_structured_fields(tmp_path):
    path = tmp_path / 'facts.jsonl'
    path.write_text('{"facts": []}\n')
    with pytest.raises(ValueError, match='line 1'):
        fact_training.load_examples(path)
