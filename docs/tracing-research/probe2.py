"""Probe 2: (a) full span-name list, (b) does image masking reach Predict input.value,
(c) is the pydantic-input loss avoidable via config?"""

import json
import os

# Try the documented masking knobs BEFORE importing the instrumentor.
os.environ['OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH'] = '32'

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
DSPyInstrumentor().instrument(tracer_provider=provider)


class WindowNode(BaseModel):
    position: int
    content: str
    # A field whose *value* contains the delimiter pattern, to test repr-parsing.
    role: str


class Span(BaseModel):
    start: int
    end: int


class GroupFinder(dspy.Signature):
    """Cut the node stream into untyped spans."""

    window: list[WindowNode] = dspy.InputField()
    spans: list[Span] = dspy.OutputField()


class Corrector(dspy.Signature):
    """Correct a transcription against the page image."""

    page_image: dspy.Image = dspy.InputField()
    transcription: str = dspy.InputField()
    corrected: str = dspy.OutputField()


PNG = (
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ'
    'AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
)

lm = dspy.utils.DummyLM(
    [
        {'reasoning': 'r', 'spans': [{'start': 0, 'end': 1}]},
        {'corrected': 'ok'},
    ]
)
dspy.settings.configure(lm=lm)

# Adversarial content: contains "role=" and quotes, which repr-flattening cannot
# be unambiguously parsed back out of.
dspy.ChainOfThought(GroupFinder)(
    window=[
        WindowNode(
            position=0,
            content="Let role='x' content='y' denote the map.",
            role='paragraph',
        )
    ]
)
dspy.Predict(Corrector)(
    page_image=dspy.Image(url=PNG), transcription='t'
)

print('=== (a) ALL SPAN NAMES, in order ===')
for s in exporter.get_finished_spans():
    print(f'  {s.name!r}')

print('\n=== (b) IMAGE MASKING (OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH=32) ===')
for s in exporter.get_finished_spans():
    a = dict(s.attributes or {})
    iv = str(a.get('input.value', ''))
    if 'base64' in iv or 'CUSTOM-TYPE' in iv:
        print(f'  {s.name}: input.value STILL contains base64 -> {iv[:150]}')
    img_attrs = [k for k in a if 'image_url' in k]
    for k in img_attrs:
        print(f'  {s.name}: {k} = {str(a[k])[:90]}')

print('\n=== (c) PYDANTIC INPUT FIDELITY (adversarial content) ===')
for s in exporter.get_finished_spans():
    if s.name != 'ChainOfThought.forward':
        continue
    raw = json.loads(dict(s.attributes)['input.value'])
    item = raw['window'][0]
    print(f'  stored as {type(item).__name__}: {item!r}')
    print('  -> to rebuild WindowNode you must parse that string. Original was:')
    print(
        "     content=\"Let role='x' content='y' denote the map.\", "
        "role='paragraph'"
    )
