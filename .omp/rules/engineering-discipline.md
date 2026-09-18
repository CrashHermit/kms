---
alwaysApply: true
description: Prefer root-cause fixes, established architecture, and contract-oriented implementations over defensive patches.
---

## Root-Cause Engineering

For any non-trivial code change:

1. Diagnose before editing.
   Identify the observed failure, its execution path, and the owning invariant.
   Decide whether it is a localized defect or evidence of a broader design problem.

2. Fix the cause, not the symptom.
   Prefer correcting the violated contract, boundary, data flow, or abstraction.
   Do not add local guards, arbitrary fallbacks, hardcoded special cases, aliases,
   or broad try/catch blocks merely to suppress the observed failure.

3. Reuse the established architecture.
   Inspect neighboring implementations and existing call patterns before introducing
   a new abstraction or boundary. Do not create a second convention for the same
   responsibility.

4. Preserve invariants explicitly.
   State the relevant preconditions, postconditions, ownership rules, and
   provenance requirements. Make invalid states impossible where practical.

5. Handle trade-offs proportionally.
   For routine changes, choose the existing safe pattern and proceed.
   Present alternatives only when options have materially different architectural,
   compatibility, performance, or data-integrity consequences. Recommend one.

6. Keep the change clean.
   Update all callers, tests, documentation, and obsolete paths affected by the
   change. Do not leave compatibility shims, dead code, placeholders, or deferred
   cleanup unless explicitly requested.

7. Verify behavior, not just syntax.
   Reproduce the original failure when possible, apply the change, and run the
   narrowest meaningful behavioral verification. Report unrelated failures
   separately rather than masking them.

## Perfect-Contract Assumption

When the task explicitly states that inputs and outputs are valid and perfect:

- Treat the stated contract as an invariant, not as a condition to defend against.
- Do not invent malformed-input cases, output-repair logic, fallback values,
  exception paths, retries, or special-case validation.
- Implement the direct contract-preserving flow.
- Do not add defensive try/catch blocks or local null checks unless the task
  explicitly requires runtime handling for that boundary.
- If the contract cannot be satisfied, identify the violated design contract and
  correct it at the owning boundary instead of masking it downstream.

This assumption applies only when explicitly stated for the task; otherwise,
handle real invalid-input and failure contracts according to the surrounding
architecture.

Before editing, briefly state:

- Problem
- Root cause
- Decision
- Invariant being preserved
- Verification plan
