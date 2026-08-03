# Entity extractor eval

Measures `kms.ingestion.entity_extractor` against the commitments it already
makes in its own prompt, on real facts from real pages. **Nothing here is
imported by the pipeline** — this directory only reads `kms.*`, it never
changes it.

## Why it is shaped this way

The pass promises specific things: names only, 1-6 words, the fact's own
wording, LaTeX kept with its delimiters, each entity at most once per fact, no
bound symbols, no derivation steps, no pronouns. Most of that is checkable
against the source fact alone — no hand-labelled answer key needed — so a
prompt edit can be measured the same day it is made.

That is deliberately a **weaker** instrument than the corrector and extractor
gold sets, and it does not replace one. It cannot tell you the pass found the
right entities; it tells you when the pass broke a rule it was told to keep.
Precision against a labelled key, and recall in any real sense, still need
labels.

## Two scripts

```
build_facts.py   gold pages -> the fixed fact corpus     (run once)
run_eval.py      corpus -> live entity run -> findings   (run per prompt edit)
```

Splitting them is the point. Building facts costs a formatter, an extractor
and an atomic-fact call per page; if the entity eval rebuilt them each time,
every number would move for two reasons at once and a prompt edit could not be
attributed. The corpus is committed at
`data/eval/entity_extractor/facts.json`, so the input is frozen and a delta
between runs is the entity prompt or the model, nothing else.

### `build_facts.py`

```
corrected.md -> formatter -> extractor -> flatten -> atomic facts
```

Input is the 38 `kind == 'real'` records of `data/gold/corrector` — body
prose, exercises and proofs across 12 books. A corrector record's
`corrected.md` **is** the first pipeline stage's output, hand-checked against
the page image, so starting there is production-exact rather than an
approximation, and it needs no page images and no vision model.

`data/gold/extractor` was tried first and is the wrong corpus: it is weighted
toward front matter, because apparatus is what it exists to measure. Run
through this chain its pages yield **zero** atomic facts — correctly — and so
cannot exercise the entity pass at all.

The formatter stays in the chain because it is the stage that wraps bare
mathematics in `$…$` and rewrites Unicode notation as LaTeX. Without it the
corpus would carry `x⁴` and the entity pass would be charged with a LaTeX
violation made upstream of it.

Node ids restart at 0 per page. Nothing in the entity pass reads them.

### `run_eval.py`

Mirrors `entity_extractor.extract_entities` rather than calling it — same
`_batch_facts` cut, same `llm.gate`, same `aforward` per batch — because the
eval needs to know **which batch** produced each mention. That is the only way
to catch a `fact_index` that names a real fact which was never in the batch
the model was looking at: the mention lands on the wrong fact, plausibly, and
nothing downstream can tell. If `extract_entities` changes shape, this
mirroring is what has to be updated with it.

Exit status is non-zero when any hard violation is found, so it can gate CI.

## The findings, and how much to trust them

**`violation` — unambiguous.** No judgement involved in calling these wrong.

| check | what it catches |
|---|---|
| `index_out_of_range` | `fact_index` outside the record's facts |
| `index_outside_batch` | a real fact index the batch was never shown — silent misattribution |
| `duplicate_in_fact` | same name twice for one fact |
| `empty_name` | blank name |
| `unbalanced_latex` | odd number of `$` |
| `alt_delimiters` | `\(…\)` / `\[…\]` instead of the dollar convention |
| `unicode_math` | `≤`, `α`, `x⁴` where LaTeX is required |
| `pronoun_name` | "it", "this", "they" |
| `ungrounded` | under half the name's tokens appear in the fact — invention |

**`review` — heuristic, a queue to read rather than a score to minimise.**
"Is this a bound symbol?" is a question about language, and these checks
answer it by looking for a three-letter run inside the LaTeX. That proxy has a
known false positive the prompt names explicitly: `$$e^{i\pi} + 1 = 0$$` is a
named formula the pass is supposed to **keep**, and it trips the same wire as
`$0 > 18$`, which the pass is supposed to drop.

| check | what it flags |
|---|---|
| `bare_symbol` | a lone symbol with no word in it — the variable tier owns these |
| `pure_relation` | a relation with no named object — a derivation step, or a named formula |
| `gloss_quoted` | a quoted phrase — a symbol's reading, not an entity |
| `too_many_words` | over the stated 1-6 |
| `not_verbatim` | not a substring of the fact, but mostly its tokens — normalisation drift |

**Coverage, not a finding.** `facts_without_mentions` is reported separately
because the prompt explicitly allows a fact to mention nothing. A bare count
cannot separate correct silence from a miss, so it is a number to watch across
runs, not a failure.

## Using it

```bash
uv run python evals/entity_extractor/build_facts.py          # once
uv run python evals/entity_extractor/run_eval.py             # per edit
uv run python evals/entity_extractor/run_eval.py \
    --baseline output/evals/entity_extractor/<earlier>.json  # to compare
```

Both take `--limit N` for a cheap partial run. Reports land in
`output/evals/entity_extractor/<timestamp>.json`, which is gitignored — keep
one by hand if you want a durable baseline.

## What this cannot tell you

The pass's output currently has **no consumer**: `entity_mentions` is written
to state by `pipeline.py`, and nothing reads it — the ingestion persister takes
nodes, statements, procedures, instructions, variables and facts, and there is
no entity label in `graph/schema.py`. So a full `pipeline.run()` would exercise
the pass and discard the result, which is why this harness drives the module
directly. Once mentions reach the graph, an end-to-end check belongs beside
this one, not inside it.
