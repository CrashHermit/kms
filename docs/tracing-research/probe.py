"""Probe: what do Phoenix/OpenInference spans actually contain for kms-shaped DSPy calls?

Mimics three real kms stages:
  - group_finder: pydantic DTO list in, pydantic DTO list out, ChainOfThought
  - role_typer:   plain Predict, literal out
  - corrector:    dspy.Image input (the base64 payload question)
"""

import json
from typing import Literal

import dspy
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace_api.set_tracer_provider(provider)
DSPyInstrumentor().instrument(tracer_provider=provider)


class WindowNode(BaseModel):
    position: int
    content: str
    role: str


class Span(BaseModel):
    start: int
    end: int


class GroupFinder(dspy.Signature):
    """Cut the node stream into untyped spans."""

    window: list[WindowNode] = dspy.InputField()
    spans: list[Span] = dspy.OutputField()


class RoleTyper(dspy.Signature):
    """Say whether a span is a block or a derivation."""

    contents: str = dspy.InputField()
    role: Literal['entity', 'procedure'] = dspy.OutputField()


class Corrector(dspy.Signature):
    """Correct a transcription against the page image."""

    page_image: dspy.Image = dspy.InputField()
    transcription: str = dspy.InputField()
    corrected: str = dspy.OutputField()


# A 1x1 PNG, standing in for a real rendered page.
PNG = (
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ'
    'AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
)

lm = dspy.utils.DummyLM(
    [
        {'reasoning': 'two units here', 'spans': [{'start': 0, 'end': 1}]},
        {'role': 'entity'},
        {'corrected': 'Theorem 1.1 (Cauchy).'},
    ]
)
dspy.settings.configure(lm=lm)

dspy.ChainOfThought(GroupFinder)(
    window=[
        WindowNode(position=0, content='Theorem 1.1', role='heading'),
        WindowNode(position=1, content='Proof. Let x...', role='paragraph'),
    ]
)
dspy.Predict(RoleTyper)(contents='Theorem 1.1 (Cauchy).')
dspy.Predict(Corrector)(
    page_image=dspy.Image(url=PNG), transcription='Theorem 1.l (Cauchv).'
)

print('=' * 70)
for span in exporter.get_finished_spans():
    attrs = dict(span.attributes or {})
    kind = attrs.get('openinference.span.kind')
    print(f'\nSPAN: {span.name!r}  kind={kind}')
    for key in ('input.value', 'output.value'):
        val = attrs.get(key)
        if val is None:
            continue
        text = str(val)
        print(f'  {key} ({len(text)} chars): {text[:400]}')
    extra = [
        k
        for k in attrs
        if k
        not in {
            'input.value',
            'output.value',
            'input.mime_type',
            'output.mime_type',
            'openinference.span.kind',
        }
    ]
    print(f'  other attrs: {extra[:12]}')

print('\n' + '=' * 70)
print('ROUND-TRIP TEST: can we rebuild dspy.Example from the span?')
for span in exporter.get_finished_spans():
    if span.name != 'ChainOfThought.forward':
        continue
    attrs = dict(span.attributes or {})
    try:
        raw_in = json.loads(attrs['input.value'])
        raw_out = json.loads(attrs['output.value'])
    except Exception as exc:
        print(f'  FAILED to parse: {exc}')
        break
    print(f'  parsed inputs  keys: {list(raw_in)}')
    print(f'  parsed inputs  types: {[type(v).__name__ for v in raw_in.values()]}')
    print(f'  raw input repr: {str(raw_in)[:300]}')
    print(f'  parsed outputs keys: {list(raw_out)}')
    print(f'  raw output repr: {str(raw_out)[:300]}')
    ex = dspy.Example(**raw_in, **raw_out).with_inputs(*raw_in)
    print(f'  dspy.Example built: {str(ex)[:200]}')
