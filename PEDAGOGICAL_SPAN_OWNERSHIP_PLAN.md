## Context
The observed `01_CEN4020.pdf` recording supplied correct one-based local indexes but returned the exact same `(18, 18)` span twice in a single response. The desired behavior is deterministic, duplicate-free component ownership without rejecting an otherwise usable finder response; true malformed ranges and non-identical overlap remain invalid. The obsolete chain-of-thought predictor switch is removed, making `dspy.Predict` the sole predictor factory for `module.Module`.

## Approach
### 1. Add a dedicated DSPy window-readiness router
Create `src/kms/construction/pedagogical_window_readiness.py` following `module.Module` and the established router pattern in `block_corrector.py`, `instruction_finder.py`, and `splitter.py`. Define a strict boolean router with `current_nodes: list[models.NodeInput]`, `is_complete: bool`, one-based `context_window.node_input` encoding, and `TypeError('is_complete must be a bool')` for non-bool predictions. The router must not expose or depend on a chain-of-thought toggle.

### 2. Remove the obsolete chain-of-thought switch
In `src/kms/core/module.py`, delete the `use_chain_of_thought` class attribute and construct `self.predictor = dspy.Predict(self.signature)` directly. Remove all `use_chain_of_thought` subclass overrides from `block_corrector.py`, `formatter.py`, `formatter_line_edits.py`, and `procedures.py`, plus the `module.ChainOfThought` test stub in `tests/conftest.py`; no chain-of-thought switch or references remain under `src` or `tests`.
The router decides only whether the supplied node run is sufficient to determine all pedagogical-unit boundaries that start within it. It returns `False` when the final supplied node may continue a labelled/numbered unit, proof, worked solution, subpart sequence, prescribed procedure, or OCR fragment; it returns `True` when the final node closes the current unit or no unit is present. It must not return spans, classify units, or own nodes.
### 3. Gate span extraction on readiness
In `src/kms/core/walker.py`, extend `find_spans(nodes, module, readiness_module, budget, max_budget)`. Call readiness first for each window; when false before document end, grow the same cursor window without calling the finder; at `max_budget`, raise the window-readiness look-ahead-limit error. At document end, call the finder once even if readiness is false. Ready windows call the finder once, validate strictly, map positions, and advance past the final span or complete window.
### 4. Wire the readiness router through the pedagogical finder
In `src/kms/construction/pedagogical_component_finder.py`, require `readiness_module: PedagogicalWindowReadinessRouter` in the public `find_spans()` and `PedagogicalComponentFinderNode` APIs, passing it to `walker.find_spans()` with no hidden fallback. In `src/kms/construction/workflow.py`, register the router under `pedagogical_window_readiness_router` using the component-finder LM identity and inject it into the graph node.

Keep `_deduplicate_spans()` and strict non-identical overlap/reversed-order/invalid-bound validation unchanged.

### 5. Test readiness gating and strict span contracts
Add readiness tests for one-based encoding and strict boolean decoding. Update walker/finder tests with scripted readiness and finder modules covering grow-before-finder, ready invocation, max-budget failure without finder calls, and final-window fallback. Update existing direct calls and node construction to inject readiness explicitly.

### 6. Verify with the failed-book scenario
Run focused readiness, walker, finder, and module tests plus Ruff. Confirm no chain-of-thought switch references remain under `src` and `tests`. Then run non-destructive ingestion of `books/01_CEN4020.pdf` with source `01_CEN4020` and a distinct output directory; inspect readiness recordings and report final node/triplet totals without clearing Neo4j.
## Verification
From the repository root:

1. Run `.venv/bin/python -m pytest -q tests/test_module.py tests/test_pedagogical_window_readiness.py tests/test_pedagogical_component_finder.py tests/test_walker.py`.
2. Run `.venv/bin/ruff check` and `ruff format --check` for every changed Python file.
3. Confirm no `use_chain_of_thought` or `ChainOfThought` occurrences remain under `src` and `tests`.
4. Run the requested book through non-destructive ingestion with `PDF = Path('books/01_CEN4020.pdf')`, `source = '01_CEN4020'`, and a new output directory. Expected observable result: readiness recordings occur before span extraction; unready windows have no span-finder recording; ingestion reaches final projection and reports node/triplet totals.

