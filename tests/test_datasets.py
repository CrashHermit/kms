"""Reading traces back as `dspy.Example`s, grouped by stage.

The MLflow store is stood in for: `_tagged_spans` and the example-building loop are the
logic worth pinning, and both take plain objects. Loading from a real store is exercised
end to end elsewhere (it needs the mlflow extra); these run anywhere.
"""

from kms.core import datasets, tracing


class _Span:
    """Stands in for an MLflow span."""

    def __init__(self, attributes, inputs=None, outputs=None):
        self.attributes = attributes
        self.inputs = inputs
        self.outputs = outputs


class _Trace:
    """Stands in for an MLflow trace: a bag of spans under `.data.spans`."""

    def __init__(self, *spans):
        self.data = type('Data', (), {'spans': list(spans)})()


def _tagged(stage, inputs, outputs):
    return _Span({tracing.STAGE_ATTR: stage}, inputs, outputs)


def test_only_tagged_spans_are_selected():
    # one logical Predict call also emits ChatAdapter/LM spans, and a ChainOfThought adds
    # an outer wrapper — none of them tagged, none of them a training example
    trace = _Trace(
        _Span({}, {'signature': 'x -> y'}, None),  # ChatAdapter.format
        _tagged('group_finder', {'current_nodes': []}, {'spans': []}),
        _Span({}, {'messages': []}, ['...']),  # LM.__call__
    )
    spans = datasets._tagged_spans([trace])
    assert len(spans) == 1
    assert spans[0].attributes[tracing.STAGE_ATTR] == 'group_finder'


def test_a_span_with_no_attributes_is_skipped():
    assert datasets._tagged_spans([_Trace(_Span(None))]) == []


def test_examples_are_grouped_by_stage(monkeypatch):
    monkeypatch.setattr(
        datasets,
        '_load_traces',
        lambda source: [
            _Trace(
                _tagged('group_finder', {'current_nodes': [1]}, {'spans': []}),
                _tagged('group_finder', {'current_nodes': [2]}, {'spans': []}),
                _tagged('role_typer', {'contents': 'c'}, {'role': 'entity'}),
            )
        ],
    )
    by_stage = datasets.examples_by_stage('traces/book')
    assert sorted(by_stage) == ['group_finder', 'role_typer']
    assert len(by_stage['group_finder']) == 2


def test_an_example_marks_the_calls_inputs_as_input_keys(monkeypatch):
    monkeypatch.setattr(
        datasets,
        '_load_traces',
        lambda source: [
            _Trace(
                _tagged(
                    'role_typer',
                    {'contents': 'Theorem 1.1'},
                    {'reasoning': 'it asserts', 'role': 'entity'},
                )
            )
        ],
    )
    example = datasets.examples_by_stage('traces/book')['role_typer'][0]
    assert example.inputs().toDict() == {'contents': 'Theorem 1.1'}
    # ChainOfThought's reasoning is signal worth training on, so it stays a label
    assert example.labels().toDict() == {
        'reasoning': 'it asserts',
        'role': 'entity',
    }


def test_a_half_recorded_call_is_dropped(monkeypatch):
    # a failed call has inputs but no outputs; it trains nothing
    monkeypatch.setattr(
        datasets,
        '_load_traces',
        lambda source: [
            _Trace(
                _tagged('role_typer', {'contents': 'c'}, None),
                _tagged('role_typer', None, {'role': 'entity'}),
            )
        ],
    )
    assert datasets.examples_by_stage('traces/book') == {}


def test_stage_counts_summarises_a_sweep(monkeypatch):
    monkeypatch.setattr(
        datasets,
        '_load_traces',
        lambda source: [
            _Trace(
                _tagged('role_typer', {'contents': 'a'}, {'role': 'entity'}),
                _tagged('role_typer', {'contents': 'b'}, {'role': 'procedure'}),
                _tagged('block_typer', {'contents': 'c'}, {'type': 'theorem'}),
            )
        ],
    )
    assert datasets.stage_counts('traces/book') == {
        'block_typer': 1,
        'role_typer': 2,
    }
