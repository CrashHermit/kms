"""Probe 4: end-to-end against a real Phoenix server. Does a big page-correction
span land, and can we query it back into a dataset?"""

import base64
import os
import time

import phoenix as px

session = px.launch_app()
print(f'phoenix up at {session.url}')
time.sleep(5)

os.environ['PHOENIX_COLLECTOR_ENDPOINT'] = session.url
from phoenix.otel import register

tp = register(project_name='kms-probe', auto_instrument=True, batch=False,
              protocol='http/protobuf')

import dspy


class Corrector(dspy.Signature):
    """Correct a transcription against the page image."""

    page_image: dspy.Image = dspy.InputField()
    transcription: str = dspy.InputField()
    corrected: str = dspy.OutputField()


class RoleTyper(dspy.Signature):
    """Say whether a span is a block or a derivation."""

    contents: str = dspy.InputField()
    role: str = dspy.OutputField()


dspy.settings.configure(
    lm=dspy.utils.DummyLM([{'corrected': 'ok'}, {'role': 'entity'}])
)

blob = base64.b64encode(os.urandom(450_000)).decode()
dspy.Predict(Corrector)(
    page_image=dspy.Image(url=f'data:image/png;base64,{blob}'),
    transcription='t',
)
print('sent big image span')
dspy.Predict(RoleTyper)(contents='Theorem 1.1 (Cauchy).')
print('sent small text span')

time.sleep(12)

from phoenix.client import Client
df = Client(base_url=session.url).spans.get_spans_dataframe(project_identifier='kms-probe')
print(f'\nspans landed: {len(df)}')
if len(df):
    print(f'columns: {[c for c in df.columns][:14]}')
    for _, row in df.iterrows():
        name = row.get('name')
        iv = str(row.get('attributes.input.value', ''))
        print(f'  {name:<32} input.value={len(iv):>9,} chars')
