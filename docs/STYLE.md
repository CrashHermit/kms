# Python Style Guide

This guide describes the conventions that make the Python codebase easy to read,
change, and review. It favors the standard library and established Python
idioms over rules that depend on a particular subsystem, package layout, or
implementation phase.

The project configuration is authoritative for automatically enforced behavior.
If this guide and `pyproject.toml` disagree, update one of them; do not let the
disagreement persist.

---

## 1. General principles

- Prefer clear, direct code over clever or highly abstract code.
- Keep functions and classes focused. Put behavior near the data and boundary
  it belongs to.
- Make dependencies, state changes, and failure modes visible.
- Use names that explain domain meaning rather than implementation details.
- Preserve existing behavior unless a change is intentional and documented.
- Follow the conventions of nearby code when adding to an established module.
- Comments should explain why the code is necessary, surprising, or constrained.
  Do not narrate code that is already self-explanatory.

When two rules conflict, choose the option that is clearer at the call site and
record the reason in review.

---

## 2. Imports

Use three import groups, in this order, with a blank line between groups:

1. Standard library.
2. Third-party packages.
3. First-party project packages.

Keep imports sorted by the formatter. Prefer absolute imports for project
packages when they make the module's dependency clear. Use relative imports
only when the package structure makes them simpler and unambiguous.

Import the smallest useful public unit:

```python
from pathlib import Path

import httpx

from kms.core import models
from kms.core.state import State
```

For project modules, use module-qualified names when they improve provenance or
avoid a crowded namespace. Import a class or function directly when it is the
module's clear public API, a base class, or a standard third-party idiom.
Avoid aliases unless they resolve a real collision or make a widely used
dependency unambiguous.

Do not use wildcard imports. Keep imports at module scope unless a local import
is required to break an import cycle or defer an expensive optional dependency;
document such exceptions.

---

## 3. Naming

- Modules and packages: lowercase; use underscores for multiword module names.
- Classes and exceptions: `CapWords`.
- Functions, methods, and variables: `lower_with_under`.
- Constants: `UPPER_WITH_UNDER`.
- Type variables and generic parameters: use descriptive names unless a
  conventional single letter (`T`, `K`, `V`) is clearer.
- Boolean names should read as predicates, such as `is_ready` or `has_content`.

Prefer descriptive names over cryptic abbreviations. Common technical
abbreviations such as `id`, `url`, `uuid`, `api`, and established domain
acronyms are acceptable when their meaning is unambiguous in context. Do not
rename a clear domain term merely to expand a conventional abbreviation.

Single-character names are appropriate for short, local mathematical
expressions and trivial loop counters. They are not appropriate for values
whose meaning matters beyond that expression.

Avoid redundant type suffixes such as `_list`, `_dict`, or `_string` when the
name already communicates the value. Add a suffix when it disambiguates two
meaningfully different representations.

---

## 4. Functions, classes, and control flow

- Keep public APIs small and explicit.
- Prefer early returns for simple guard conditions.
- Avoid deeply nested conditionals; extract a named operation when that makes
  the decision easier to understand.
- Do not hide I/O, mutation, or expensive work behind names that imply a pure
  calculation.
- Use keyword arguments when positional arguments would be ambiguous.
- Use `async` only when the operation actually awaits asynchronous work.
- Use comprehensions for simple transformations and filters. Use a regular loop
  when it contains branching, side effects, or multiple meaningful steps.
- Use exceptions for exceptional failures, not ordinary control flow.

Use truthiness where the type and intent are clear:

```python
if not records:
    return []

if content:
    process(content)
```

Do not use truthiness when `None`, `0`, `False`, or an empty value have
different meanings. In those cases, test the intended state explicitly.

---

## 5. Types and data

Annotate public interfaces and non-obvious local values. Use the modern
built-in collection and union syntax supported by the project's configured
Python target:

```python
def load_labels(path: Path) -> list[str] | None:
    ...
```

Prefer precise types over `Any`. Use `Any` only at an intentional boundary,
such as untyped third-party data, and narrow the value as soon as possible.

Use dataclasses, Pydantic models, TypedDicts, enums, or plain containers
according to the data's role:

- A dataclass for structured in-process data.
- A validation model at an external or user-controlled boundary.
- A TypedDict for a dictionary-shaped interface whose keys are fixed.
- An enum for a closed set of meaningful values.
- A plain dictionary for genuinely dynamic keys.

Keep validation at boundaries. Do not duplicate validation in every consumer
unless the consumer has an independent invariant to enforce.

Use immutable values where practical. When mutating an object in place, make
that ownership and mutation visible in the function name, documentation, or
type/interface.

---

## 6. Errors, resources, and logging

Catch the narrowest exception that can be handled. Never use a bare `except:`.
Preserve useful context when translating an exception:

```python
try:
    settings = parse_settings(raw_settings)
except ValueError as error:
    raise ConfigurationError('Invalid settings') from error
```

Do not catch an exception merely to log and re-raise it. Add context at the
boundary that can act on the failure.

Use context managers for resources that need deterministic cleanup. Prefer
library-supported async context managers for asynchronous resources.

Log events at the layer that has useful context. Do not log secrets,
credentials, tokens, or unredacted user content. Use structured logging
arguments rather than eagerly formatting messages.

---

## 7. Docstrings and comments

Document public modules, classes, functions, and methods when their purpose,
contract, side effects, or failure behavior is not obvious from the signature.
Private helpers need documentation when their behavior is non-trivial.

Use triple double quotes for docstrings. Start with a concise summary sentence.
For non-trivial APIs, add sections that match the contract:

```python
def find_items(
    query: str,
    *,
    limit: int = 20,
) -> list[Item]:
    """Find items matching a query.

    Args:
        query: Text to search for.
        limit: Maximum number of items to return.

    Returns:
        Matching items in relevance order.

    Raises:
        ValueError: If `limit` is less than one.
    """
```

Document arguments, return values, and exceptions only when they add
information beyond the type signature and name. Document side effects and
mutations explicitly. Do not add empty or boilerplate docstrings just to meet a
numeric rule.

Comments and docstrings must remain true as the implementation changes. Avoid
copying volatile implementation details, model prompts, or lists of current
modules into a general guide.

---

## 8. Formatting

Use four spaces for indentation and never tabs. Do not use semicolons or place
multiple statements on one line.

Keep lines within the configured formatter limit. Let the project formatter
choose wrapping, trailing commas, and most whitespace details. For manual
wrapping, put each argument on its own line when that is clearer and keep the
closing delimiter aligned with the opening construct.

Use trailing commas in multi-line collections, calls, and definitions. They
produce stable diffs and allow formatters to reflow code safely.

Use single-quoted strings by default when that matches the formatter settings.
Use double quotes when they avoid escaping or when required by an external
format. Docstrings use triple double quotes.

Keep module-level definitions separated by two blank lines and methods inside a
class separated by one blank line, unless the formatter applies a different
project-wide convention.

---

## 9. Tests and changes

Tests should describe observable behavior and important boundaries, not mirror
implementation details. Include coverage for success, meaningful edge cases,
state transitions, and expected failures when applicable.

Use deterministic, isolated tests. Avoid network, filesystem, clock, randomness,
and model-service dependencies unless the test explicitly exercises that
integration and provides the required fixture or configuration.

Keep a change's implementation, tests, and documentation consistent. Remove
obsolete comments and examples when behavior or structure changes.

---

## 10. Tooling

Use the repository's configured tools rather than reproducing their settings in
this document. Typical commands are:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

Run the narrowest relevant checks while iterating, then run the applicable
project checks before merging. A formatter change is not a substitute for a
linter or test run.

When a rule needs an exception, prefer a narrowly scoped, tool-recognized
exception with a reason. Do not weaken a project-wide rule to accommodate one
call site.

---

## 11. Keeping this guide current

This document is policy, not a changelog or migration checklist. Do not list
current modules, individual renames, historical violations, or copied tool
configuration that will become stale as the program evolves.

When the project changes:

1. Update tooling configuration and code conventions together.
2. Change this guide only when the intended convention changes or a general
   explanation is missing.
3. Prefer stable principles and representative examples over inventories.
4. Remove examples that no longer demonstrate the current rule.
5. Verify the guide against the configured tools and nearby source code.

The guide should remain useful if packages are renamed, subsystems are added,
or implementation technologies change.
