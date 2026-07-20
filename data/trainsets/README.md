# DSPy trainsets

Captured `{inputs, outputs}` examples for the DSPy signatures, one JSONL file per
signature. Each line maps to a `dspy.Example`; load them with
`module.trainsets.load("<signature>")`.

| file | signature | example |
|---|---|---|
| `extractor.jsonl` | `extractor.Signature` | markdown → structural nodes |
| `entity_grouper.jsonl` | `entity_grouper.Signature` | window nodes → entity spans |
| `entity_attributor.jsonl` | `entity_attributor.Signature` | entity type + members → roles |

## Provenance

Seed set captured from two openly-licensed textbooks (distinct formatting):

- **OpenStax, Calculus Volume 1** (CC BY-NC-SA) — the Continuity section (worked
  Examples with `Solution` headers, checkpoints).
- **Judson, Abstract Algebra: Theory and Applications** (GNU FDL) — the polynomials
  section (Lemma/Corollary/Proposition + run-in `Proof.`).

Labels are **silver**: the pipeline's post-hardening outputs, spot-verified but not
exhaustively gold-checked. The `entity_attributor` labels are post marker-split, so
they are effectively gold for marker-delimited entities.

## Regenerating / extending

Run the pipeline with `KMS_CAPTURE_DIR` set to append examples (see
`src/module/capture.py`):

```bash
KMS_CAPTURE_DIR=data/trainsets PYTHONPATH=src uv run python -m module.pipeline <pdf> <out>
```

To grow the set, run more source pages through the (verified) pipeline and curate the
new lines. This is a small seed — enough to bootstrap few-shot demos, thin for heavy
instruction optimization.
