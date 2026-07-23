# Training — data-driven stage optimization (DSPy)

We stopped hand-tuning stage prompts and drive quality from **data + a metric**
instead. Each stage is a DSPy program; a strong **teacher** (`deepseek-v4-pro`)
bootstraps high-quality traces, an **LLM judge** filters them, and the winners
become few-shot demonstrations the **flash student** (`deepseek-v4-flash`, the
workhorse doing inference) runs with. No model weights are changed — this is DSPy
*compilation* (optimized demos/instructions), which is what the DeepSeek API allows.

Pilot stage: **exercise splitter**.

## The loop

1. **Capture** — run fixtures with `KMS_TRACE_DIR` set; each stage appends its real
   per-call `(inputs, outputs)` to `$KMS_TRACE_DIR/<stage>.jsonl`
   (`core/tracing.py`). Traces come from the *actual* input a stage received, so an
   upstream failure (e.g. an OCR-dropped lead-in) is never mislabeled as this stage's
   error.

   ```
   KMS_TRACE_DIR=out/traces PYTHONPATH=src uv run --extra mistral \
     python -m training.capture_splitter            # no args => all fixtures
   ```

2. **Metric** — `training/splitter/metric.py`: a reference-free **LLM judge** on the
   teacher grades whether a window's splits + lead-in tags are both correct. Because
   it's reference-free, the trainset needs **inputs only** — no hand-labeled gold.

3. **Compile** — `training/splitter/compile.py`: `BootstrapFewShot` runs the teacher
   over the captured windows, keeps judge-approved outputs as demos for the student,
   reports dev judge pass-rate (baseline vs compiled), and saves the program.

   ```
   PYTHONPATH=src uv run python -m training.splitter.compile out/traces/splitter.jsonl
   ```

4. **Serve** — `SplitterNode` loads `training/splitter/compiled.json` if present, so
   the optimized demos ship with the pipeline. Delete the file to fall back to the
   bare student.

## Growing the data

Every stress-test run with `KMS_TRACE_DIR` set adds windows. When the judge is wrong,
tighten the judge signature (still no per-example labels) — the judge itself is the one
prompt we curate, and it's graded by the strong model, not the workhorse. As coverage
grows, graduate `BootstrapFewShot` → `MIPROv2` (also optimizes the instruction).

## Layout

```
training/
  capture_splitter.py     # truncated pipeline (front-end -> splitter) that harvests traces
  splitter/
    metric.py             # LLM-as-judge metric (teacher-graded, reference-free)
    dataset.py            # traces -> unique input windows as dspy.Example
    compile.py            # BootstrapFewShot: teacher-bootstrapped demos for the flash student
    compiled.json         # saved compiled program (committed; loaded at serve time)
```
