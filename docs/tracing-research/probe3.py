"""Probe 3: realistic page-image size. Does masking apply anywhere, and how big
does one corrector span get? Our corrector sends one rendered page PNG per page."""

import base64
import os

os.environ['OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH'] = '1000'

import dspy
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
DSPyInstrumentor().instrument(tracer_provider=provider)


class Corrector(dspy.Signature):
    """Correct a transcription against the page image."""

    page_image: dspy.Image = dspy.InputField()
    transcription: str = dspy.InputField()
    corrected: str = dspy.OutputField()


# ~600 KB of base64 — the ballpark of one rendered textbook page at the DPI a
# vision correction pass needs.
blob = base64.b64encode(os.urandom(450_000)).decode()
url = f'data:image/png;base64,{blob}'
print(f'image payload: {len(url):,} chars')

dspy.settings.configure(lm=dspy.utils.DummyLM([{'corrected': 'ok'}]))
dspy.Predict(Corrector)(page_image=dspy.Image(url=url), transcription='t')

total = 0
for s in exporter.get_finished_spans():
    a = dict(s.attributes or {})
    size = sum(len(str(k)) + len(str(v)) for k, v in a.items())
    total += size
    carries = [
        k for k, v in a.items() if 'base64' in str(v) and len(str(v)) > 10_000
    ]
    print(f'  {s.name:<32} attrs={size:>10,} chars  full-image attrs={carries}')

print(f'\nTOTAL for ONE page-correction call: {total:,} chars')
print(f'image appears ~{total / len(url):.1f}x its own size across spans')
print(f'\nExtrapolated for a 300-page book: {total * 300 / 1e9:.2f} GB of spans')
print('gRPC OTLP default max message size is 4 MB per export batch.')
