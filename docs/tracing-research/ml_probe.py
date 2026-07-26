"""Does MLflow's DSPy autolog beat our custom capture on the ONE thing that matters:
faithful inputs for dspy.Example reconstruction?

Same three kms-shaped signatures, same adversarial content as the Phoenix probes.
"""

import base64
import json
import os

import mlflow

# Local SQLite, no server process. Does trace logging work without one?
mlflow.set_tracking_uri('sqlite:///mlprobe.db')
mlflow.set_experiment('kms-probe')
mlflow.dspy.autolog()

import dspy
from pydantic import BaseModel


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
    role: str = dspy.OutputField()


class Corrector(dspy.Signature):
    """Correct a transcription against the page image."""

    page_image: dspy.Image = dspy.InputField()
    transcription: str = dspy.InputField()
    corrected: str = dspy.OutputField()


dspy.settings.configure(
    lm=dspy.utils.DummyLM(
        [
            {'reasoning': 'r', 'spans': [{'start': 0, 'end': 1}]},
            {'role': 'entity'},
            {'corrected': 'ok'},
        ]
    )
)

# Same adversarial content that defeated Phoenix's repr flattening.
dspy.ChainOfThought(GroupFinder)(
    window=[
        WindowNode(
            position=0,
            content="Let role='x' content='y' denote the map.",
            role='paragraph',
        )
    ]
)
dspy.Predict(RoleTyper)(contents='Theorem 1.1 (Cauchy).')

blob = base64.b64encode(os.urandom(450_000)).decode()
dspy.Predict(Corrector)(
    page_image=dspy.Image(url=f'data:image/png;base64,{blob}'),
    transcription='t',
)

mlflow.flush_trace_async_logging(terminate=True)
traces = mlflow.search_traces(return_type='list')
print(f'\ntraces logged (no server): {len(traces)}\n')
print('=' * 72)

for tr in traces:
    for sp in tr.data.spans:
        inputs = sp.inputs or {}
        outputs = sp.outputs
        size = len(json.dumps(inputs, default=str)) + len(
            json.dumps(outputs, default=str)
        )
        print(f'\nSPAN {sp.name!r}  type={sp.span_type}  bytes={size:,}')

        if 'window' in inputs:
            item = inputs['window'][0]
            print(f'  window[0] stored as {type(item).__name__}: {str(item)[:130]}')
            if isinstance(item, dict):
                print('  -> STRUCTURED. Rebuild is exact:')
                ex = dspy.Example(
                    window=[WindowNode(**i) for i in inputs['window']],
                    **(outputs if isinstance(outputs, dict) else {}),
                ).with_inputs('window')
                print(f'     {str(ex)[:160]}')
            else:
                print('  -> FLATTENED to string; same problem as Phoenix.')

        if 'page_image' in inputs:
            img = str(inputs['page_image'])
            print(f'  page_image stored as {len(img):,} chars: {img[:80]}')
