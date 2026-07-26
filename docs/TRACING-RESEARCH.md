# Trace capture — evaluating Arize Phoenix and MLflow against `core/tracing.py`

Research for a proposed migration: replace the custom DSPy trace capture (`src/kms/core/
tracing.py`, 226 lines, zero extra dependencies) with **Arize Phoenix**, and delete the
custom code.

**Verdict, in one line: not Phoenix; MLflow eventually; keep `tracing.py` for now.**

- **Phoenix — no.** Good tool, different job. It records our pydantic inputs as `repr`
  strings that cannot be parsed back unambiguously, which is disqualifying for the one thing
  we want traces *for*. Langfuse inherits the same flaw via the same instrumentation.
- **MLflow — yes, but not yet.** It clears every bar Phoenix failed (structured inputs,
  redactable images, no server needed) and adds optimiser-run tracking. Its marginal value
  switches on when we run a first optimiser; adopting it before then buys nothing we don't
  already have.
- **`tracing.py` — keep.** On the narrow collection job it is at least as correct as either,
  and it is already emitting the exact format an optimiser consumes.

Everything below was measured, not read off a docs page. Probe scripts and versions are in
the last section.

---

## What we need vs. what Phoenix is for

The stated goal is **collecting data for training examples in DSPy**. That is a *dataset
construction* job: every call's inputs and outputs, faithfully enough to rebuild a
`dspy.Example` and hand it to an optimiser.

Phoenix is an **observability and evaluation** platform: a span store with a UI, LLM-as-judge
evaluators, and versioned datasets for regression experiments. Those datasets are curated for
*eval*, not produced as `dspy.Example`s. Neither Phoenix's DSPy docs nor DSPy's own
observability tutorial documents a spans → `dspy.Example` path — that conversion is left to
the user, and the findings below are why it is not a formality.

The two jobs overlap in what they record, which is what makes the swap look free. It isn't.

---

## Finding 1 — pydantic inputs are stored as `repr`, not JSON (this is the blocker)

Every entity stage passes pydantic DTOs in. `group_finder` takes `list[WindowNode]`.
OpenInference records the input side by string-formatting each item:

```
input.value = {"window": ["position=0 content='Theorem 1.1' role='heading'", ...]}
```

That is `str(WindowNode(...))`, not `model_dump()`. The output side *is* real JSON
(`"spans": [{"start": 0, "end": 1}]`) — the asymmetry comes from outputs going through the
signature's `output_fields`, while inputs get a generic dump.

To rebuild `dspy.Example(window=[WindowNode(...)])` you must parse that flattened string. It
is genuinely ambiguous, not merely fiddly — node content is arbitrary textbook prose, so it
contains the delimiters. Probed with adversarial content:

```
stored as str: 'position=0 content="Let role=\'x\' content=\'y\' denote the map."
                role=\'paragraph\''
```

A `role=` / `content=` splitter cannot tell the field separator from the prose. This is the
central case for us: the field most likely to contain the delimiter pattern is the one
carrying the book's text.

`tracing.py` calls `model_dump()` and gets a faithful dict back. **On the primary use case,
the custom code is strictly more correct than the replacement.**

This finding is not Phoenix-specific — **Langfuse uses the same
`openinference-instrumentation-dspy` package** and inherits it exactly.

## Finding 2 — page images are stored in full, 4–5× per call, and masking can't stop it

`ingestion/corrector.py` sends a rendered page PNG per page. `tracing.py` records `'<image>'`
— the bytes are reconstructable from the source PDF, and the trainable signal
(transcription → corrected) is kept in full.

OpenInference embeds the whole base64 payload. Measured with a ~600 KB image, one call:

| span | attrs | carries full image |
|---|---:|---|
| `DummyLM.__call__` | 1,202,354 chars | yes, twice |
| `ChatAdapter.__call__` | 601,027 | yes |
| `Predict(Corrector).forward` | 600,316 | yes |
| `Predict.forward` | 600,316 | yes |
| **total, one page** | **3,004,013** | **~5× the image** |

Extrapolated: **~0.9 GB of spans per 300-page book.** End-to-end against a live server it
lands at ~2.4 MB/page stored (4 copies) — so this is a storage cost, not data loss.

The documented knob does not help. `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH=1000` left every
one of those attributes at full length. The masking options are all-or-nothing at the level
we care about:

```
OPENINFERENCE_HIDE_INPUTS          # kills input.value — destroys the training data
OPENINFERENCE_HIDE_INPUT_IMAGES    # message-level attrs only
OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH  # did not apply to Predict input.value
```

There is no setting that keeps text inputs while dropping image bytes. `_plain()` in
`tracing.py` is 18 lines and does exactly that.

## Finding 3 — ChainOfThought stages lose their signature name

Span naming carries the signature for a plain `Predict` but not through `ChainOfThought`,
which rebuilds its signature into DSPy's own `StringSignature` class:

```
Predict(RoleTyper).forward        # plain Predict — stage identity present
Predict(StringSignature).forward  # ChainOfThought — stage identity GONE
ChainOfThought.forward            # also no signature name
```

Recovering the stage means joining to the sibling `ChatAdapter.__call__` span (which does
carry the signature in its `input.value`) via `parent_id`. Doable; it is extra machinery in
the direction of *more* custom code, not less.

`tracing.py` already solves this by matching on signature instructions (`_by_instructions()`),
and files per stage directly.

## Finding 4 — one logical call becomes 4–5 spans

`Predict(Sig).forward`, `Predict.forward`, `ChatAdapter.__call__`, `LM.__call__`, plus
`ChainOfThought.forward`. Building a dataset means filtering to one span per call and
deduplicating the exact-duplicate `Predict` pair. `tracing.py` already solves this — the
"only instances with `.signature`" filter exists precisely because CoT would otherwise double
every line.

## Finding 5 — weight

| | packages | size |
|---|---:|---:|
| current project venv | — | 282 MB |
| `arize-phoenix` + instrumentation + dspy | 177 | 791 MB |

Pulls in FastAPI, Starlette, uvicorn, SQLAlchemy, Alembic, strawberry-graphql, pandas,
authlib. Plus a server process to run and a SQLite store to keep. `CLAUDE.md` advertises
`uv sync` as a "light CPU core" that "installs in seconds"; this is a different category. It
also cuts against the tests running anywhere with no keys and stubbed heavy deps.

For a one-person project, the operational cost of a server is not nothing: it is one more
thing to have running before a book sweep produces data.

---

## What Phoenix is genuinely better at

Not a whitewash — the custom code has real gaps, and Phoenix fills some of them:

- **A UI.** Reading a run as a trace tree beats `jq` over JSONL. Real value when a stage
  misbehaves mid-book.
- **Latency, token counts and cost**, captured automatically. `tracing.py` records none of
  this.
- **Run-to-run comparison and eval tracking**, with LLM-as-judge evaluators and versioned
  datasets — the piece that would matter if the whack-a-mole prompt tuning described in
  `tracing.py`'s docstring gets a real held-out set.
- **Prompt/message-level capture** — the exact rendered prompt, which `tracing.py` does not
  keep.

None of these is the stated goal, and none requires deleting the custom capture.

---

## Alternatives considered

**MLflow** — DSPy's *officially documented* observability integration, and the only one its
tutorial covers. **Probed to the same depth as Phoenix (`ml_probe.py`), and it passes the
tests Phoenix failed.** It hooks DSPy's callback system — the same mechanism `tracing.py`
uses — rather than wrapping at the OTel layer, which is exactly why the fidelity is better:

- **Inputs are structured.** The adversarial `WindowNode` that defeated Phoenix round-trips
  exactly: `{'position': 0, 'content': "Let role='x' content='y' denote the map.", 'role':
  'paragraph'}`. `dspy.Example` reconstruction is lossless. **This is the blocker cleared.**
- **Images are redactable** through `mlflow.tracing.configure(span_processors=[...])`, a
  supported extension point. A 6-line processor swapping base64 for `'<image>'` took one
  page-correction call from **1.2 MB to 3,445 chars** — the same thing `_plain()` does, minus
  the maintenance. Phoenix offers no equivalent.
- **No server required.** `sqlite:///mlruns.db` as the tracking URI logs traces fine; the
  server is only needed for the UI.
- Adds latency, token counts, rendered prompts, and — uniquely — **optimiser run tracking**
  (`log_traces_from_compile=True`), the thing the traces are being collected *for*.

Where it is still behind `tracing.py`:

- **Stage identity is a heuristic.** Spans are named `Predict.forward` / `ChainOfThought.
  forward` with a `signature` attribute of `'window -> reasoning, spans'` — field names, not
  the stage. Workable while our stages have distinct field names; weaker than mapping to the
  defining module, and it does not give per-stage files for free.
- **Same duplicate-span fan-out** (`ChainOfThought.forward` and its inner `Predict.forward`
  carry identical payloads) — needs the same filter `tracing.py` already applies.
- **Async export footgun.** Traces are queued and flushed at exit; querying without
  `mlflow.flush_trace_async_logging()` silently returns zero. Cost a debugging cycle here.
- **122 packages / 719 MB**, and ~40–60 lines of glue to turn `search_traces()` output back
  into per-stage `dspy.Example`s — against 226 lines that already emit exactly that.

**Langfuse** — consumes the same `openinference-instrumentation-dspy`, so Findings 1–4 apply
identically. Hosted by default; self-hosting is a docker-compose stack. No advantage here.

**W&B Weave** — hosted account required for a solo project's local sweeps. Same category
mismatch.

**Keep `tracing.py`** — 226 lines, zero dependencies, no server, no account, one JSONL file
per stage that loads straight into `dspy.Example`. Automatic for every stage via DSPy's
callback system, so a new stage is traced the day it is written. It is not general-purpose,
and that is why it fits: it makes exactly the three domain decisions Phoenix gets wrong for us
(`model_dump()` inputs, `'<image>'` placeholders, one line per logical call, per-stage files).

---

## Recommendation

1. **Keep `core/tracing.py` as the dataset path.** Do not delete it. On the job it exists for
   it is more correct than the proposed replacement, and it is ~200 lines with no deps.
2. **The bottleneck isn't capture — it's what's downstream of it.** We already emit
   optimiser-ready JSONL and have never run an optimiser. The next real step is a metric and a
   held-out set per stage, then `BootstrapFewShot`/`MIPROv2`. No observability platform moves
   that forward.
3. **MLflow is a genuine option — unlike Phoenix — and the right one to adopt when we start
   optimising.** It ties `tracing.py` on correctness (structured inputs, redactable images)
   and beats it on everything adjacent. It is not worth adopting *today* purely to collect
   data we already collect correctly; its marginal value switches on at the moment we run a
   first optimiser, because `log_traces_from_compile=True` tracks those runs. At that point it
   can genuinely **replace** `tracing.py` rather than sit beside it — which is the outcome the
   original proposal wanted, just via a different tool and at the right time.
4. **Revisit Phoenix if the project stops being solo**, or if LLM-as-judge eval over many book
   sweeps becomes the daily activity. Its evaluator and experiment tooling is the real draw —
   and by then, Finding 1 may be fixed upstream.

Cheap hedge worth knowing: Phoenix can be pointed at an existing OTel setup and run *only*
for a debugging session, without the pipeline depending on it.

---

## Reproducing

Probes are in `docs/tracing-research/`; each is standalone and runs against
`dspy.utils.DummyLM`, so no keys are needed. They are research scratch, deliberately outside
`src/` and not covered by the test suite:

```
uv venv /tmp/pxtest --python 3.13
VIRTUAL_ENV=/tmp/pxtest uv pip install arize-phoenix \
    openinference-instrumentation-dspy "dspy-ai>=3.2.1"
/tmp/pxtest/bin/python docs/tracing-research/probe.py
```

- `probe.py` — span shapes for the three kms signature shapes; the `dspy.Example` round trip.
- `probe2.py` — span-name list, image-masking knobs, adversarial pydantic content.
- `probe3.py` — image size amplification, measured.
- `probe4.py` — end-to-end against a live `px.launch_app()` server, query back via
  `phoenix.client`. Note: in sandboxes, `register(protocol='http/protobuf')` and
  `NO_PROXY=localhost` are needed — the default gRPC exporter fails against the proxy.
- `ml_probe.py` — the same three signatures under `mlflow.dspy.autolog()`: input fidelity,
  SQLite-without-a-server, and the span-processor image redaction. Needs a separate venv
  (`uv pip install "mlflow>=3" "dspy-ai>=3.2.1"`). Call
  `mlflow.flush_trace_async_logging(terminate=True)` before querying or you get zero traces.

Versions as tested (2026-07-26): `arize-phoenix` 19.6.0, `arize-phoenix-client` 2.13.0,
`openinference-instrumentation-dspy` 0.1.37 (released 2026-05-18, declares `dspy>=2.6.22`,
Python `>=3.10,<3.15`), `dspy` 3.2.1, Python 3.13. The instrumentation is actively maintained
and compatible with our stack — the findings above are about fit, not staleness.
