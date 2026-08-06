# KMS Python Style Guide

Adapted from the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Where Google leaves a choice (e.g. docstring style), this document picks one and rules
out the others so the codebase is uniform.

---

## 1. Imports (largest change from current code)

### 1.1 Import only modules — never individual symbols

Google's rule: use `import package.module` or `from package import module`. Do not import
individual classes, functions, or variables into your file's namespace.

```python
# YES (Google style)
from kms.core import models, state
from kms.core import llm

node = models.ASTNode(type=models.NodeType.PARAGRAPH, content="...")
entities: list[models.Entity] = []

# NO (current code — violates Google's explicit traceability rule)
from kms.core.models import ASTNode, Entity, NodeType
from kms.core.state import State
```

**Why:** Module-qualified names (`models.ASTNode`) make every symbol's origin unambiguous at
the call site. When you see `ASTNode` you know it came from `models` — no need to scan the
import block.

### 1.2 No relative imports

Always import using the full path from the project root.

```python
# YES
from kms.entity import finders

# NO
from . import finders
from ..core import models
```

### 1.3 Import ordering

Three groups, alphabetized within each, separated by a blank line:

1. Standard library imports
2. Third-party library imports
3. Local (`kms.*`) imports

```python
import asyncio
import os
from pathlib import Path

import dspy
from langgraph.types import Send
from pydantic import BaseModel

from kms.core import llm
from kms.core import models
from kms.core import state
```

### 1.4 Exception: `typing` and `collections.abc`

Imports from `typing` for annotation-only use are exempt — import those directly, not
the module. (The `UP` pyupgrade rule converts `typing.List` → `list` anyway.)

### 1.5 Exception: name collisions inside `graph/`

`graph/`'s modules are named after the domain nouns they map — `nodes`, `entities`,
`procedures` — and those are exactly the natural names for the *parameters* that
carry them (`def act_rows(entities: list[models.Entity], ...)`). A module import therefore
shadows, and the shadowing is not always caught at import time:

```python
# NO — the parameter shadows the module; ruff flags F823 at best, UnboundLocalError at worst
from kms.graph import procedures
def persist(entities, source):
    procedures = procedures.procedure_rows(entities, source)   # boom

# YES — symbol import where the module name collides with the domain noun
from kms.graph.procedures import procedure_rows
```

So inside `graph/`, prefer a **symbol import** when the module name would collide with a
parameter or local, and keep a file internally consistent (don't mix both styles for sibling
modules in one file). Use the module import where there is no collision — `schema.py` and
`persister.py` both do, since neither has an `entities`/`nodes` parameter.

This is narrow: it applies to `graph/` because of the deliberate module-named-after-its-subject
layout. Everywhere else §1.1 stands — and `from kms.core import models` never collides, because
nothing is called `models`.

### 1.6 Standard-library and third-party conventions

`from pathlib import Path`, `from uuid import uuid5`, `from pydantic import BaseModel`, and
`from dataclasses import dataclass, field` are used directly throughout. These are universal
idioms whose origin nobody has to trace, and `BaseModel` in particular is a base class, so the
qualified form reads badly (`class Span(pydantic.BaseModel)`). §1.1 is about **our own**
modules, where module-qualification genuinely aids traceability.

---

## 2. Line length: 80 characters

Current code uses 100. Google requires 80.

**Exception:** Unsplittable items (URLs in comments, long DSPy `description=` strings inside
`dspy.InputField` / `dspy.OutputField` — these are structurally a single string literal and
breaking them would change the prompt's meaning).

**How to handle long lines:**

```python
# YES — break before binary operators (Google's rule, already our current style)
result = (
    await self.extractor.acall(segment_markdown=segment_markdown)
)

# NO — break after binary operator
result = await self.extractor.acall(
    segment_markdown=segment_markdown)

# YES — long function signature: one arg per line, closing paren on its own line
async def aforward(
    self,
    top_bottom_edge_node: SeamNodeDTO,
    bottom_top_edge_node: SeamNodeDTO,
    top_node_context: SeamNodeDTO | None = None,
    bottom_node_context: SeamNodeDTO | None = None,
) -> SeamNodeDTO | None:
```

**DSPy docstrings:** DSPy `Signature` class docstrings are structural (they are the prompt).
Keep them under 80 chars per line where possible, but a single long sentence that must stay
contiguous is exempt.

---

## 3. Naming conventions

### 3.1 Modules: `lower_with_under`

```python
# YES
instruction_finder.py
seam_merger.py

# NO
InstructionFinder.py
seamMerger.py
```

Packages are lowercase without underscores (`kms`, `entity`, `ingestion`).

### 3.2 Classes: `CapWords` (PascalCase)

```python
class ExtractorNode:        ...
class InstructionFinderNode: ...
class GovernExtent(dspy.Signature): ...
```

### 3.3 Functions and methods: `lower_with_under`

```python
def flatten_segments(segments): ...
def merge_results_into_segments(segments, results, attr): ...
async def tag_instructions(nodes, module, budget): ...
```

### 3.4 Variables: `lower_with_under`

```python
# YES
node_count = len(nodes)
segment_index = segment.index
entity_label = entity.type.value

# NO — cryptic abbreviations
n = len(nodes)           # → node_count
seg_idx = segment.index  # → segment_index
eid = entity.id          # → entity_id
```

### 3.5 Single-character names

**Banned** except for trivial loop counters (`i`, `j`, `k` in local-only loops).

```python
# NO
n = len(nodes)           # use node_count
p = min(max(s.position, 0), last_local)  # use clamped_position
r = await self.identify.acall(...)        # use result

# YES (loop counters are the one exception)
for i, node in enumerate(out):
    node.id = i
```

### 3.6 No type suffixes

```python
# YES
customer_emails = ["a@b.com"]
entity_properties = {...}

# NO
customer_email_list = ["a@b.com"]
entity_properties_dict = {...}
```

### 3.7 Abbreviation rules

| Current | Google replacement | Why |
|---|---|---|
| `seg` | `segment` | Not an accepted abbreviation |
| `idx` | `index` | Cryptic |
| `nid` | `node_id` | Cryptic |
| `lm` (parameter) | `language_model` | Not an accepted abbreviation |
| `_est_tokens` | `_estimate_tokens` | `est` is cryptic |
| `_dto` | `_to_seam_node_dto` | Cryptic |
| `n` (list length) | `count` or `node_count` | Single-char ban |

**Allowed abbreviations** (these are standard and unambiguous):
- `id` (identifier)
- `db` (database) — only in `graph/` package
- `uuid` (universally unique identifier)
- `OCR` / `LLM` (acronyms, but use uppercase: `OCR`, not `ocr`)
- `AST` (abstract syntax tree)

---

## 4. Docstrings — Google style

Every module, class, public function, and public method must have a docstring.

```python
def attribute_definition(
    entity: models.Entity,
    nodes_by_id: dict[int, models.ASTNode],
    module: Module | None = None,
) -> models.Entity:
    """Fills in the self-contained attributes on one Definition entity, in place.

    One LLM call identifies label/number/title/field; the content members are
    assembled deterministically with the label peeled off; a second LLM call
    builds the bodylist, writing each description verbatim.

    Args:
        entity: The sparse Definition entity from the finder (members only).
        nodes_by_id: The full node stream keyed by stable id.
        module: The attributor module. Created fresh if None.

    Returns:
        The same entity, with label, number, title, field, contents, and
        bodylist filled in.
    """
```

**Structure:**
1. Summary line (imperative mood, ends with period)
2. Blank line
3. Detailed description (optional, for non-trivial functions)
4. `Args:` — one per parameter: `name: Description.`
5. `Returns:` — description of return value (omit if returning None)
6. `Raises:` — only if the function explicitly raises

**When to include:**
- **Modules:** Yes, at the top. Describe what the module provides.
- **Classes:** Yes. Summarize the class's purpose.
- **Public methods/functions:** Yes.
- **Private helpers:** Yes if non-trivial; a one-liner is fine for obvious helpers.
- **`__init__`:** Yes, but put it on the class docstring, not `__init__` itself.
- **DSPy `Signature` subclasses:** The class docstring IS the LLM prompt. Write it in
  imperative prose directed at the model — the Args section is unnecessary (the
  `dspy.InputField`/`dspy.OutputField` declarations document the signature).

---

## 5. Formatting rules

### 5.1 Indentation: 4 spaces

No tabs. Already correct across the codebase.

### 5.2 Blank lines

- Two blank lines between top-level definitions (classes, module-level functions).
- One blank line between methods inside a class.
- One blank line between import groups (see §1.3).

### 5.3 Semicolons

Never terminate lines with semicolons. Never chain two statements on one line.

### 5.4 Implicit `False` evaluation

```python
# YES
if not entities:
    return []
if segment.content:
    ...

# NO
if len(entities) == 0:
    return []
if segment.content is not None and segment.content != "":
    ...
```

### 5.5 Type annotations

Use modern syntax: `list[str]` not `typing.List[str]`, `dict[int, ASTNode]` not
`typing.Dict[int, ASTNode]`, `str | None` not `typing.Optional[str]`.

Already correct across the codebase (`UP` rule enforces this).

### 5.6 Trailing commas

Use trailing commas in multi-line lists, dicts, and function calls. Ruff format
enforces this.

### 5.7 `__future__` imports

**Never** import `from __future__ import annotations`. The project targets
Python ≥3.13, where `str | None`, `list[int]`, and all other PEP 604/585
syntax is built-in — the `annotations` future is a no-op.

---

## 6. String quoting

- **Docstrings:** Triple double quotes `"""..."""`.
- **All other strings:** Single quotes `'...'` unless the string contains a single quote
  or is a docstring.

---

## 7. Automated enforcement

### 7.1 Ruff configuration changes

The current `pyproject.toml` needs these changes for Google style:

```toml
[tool.ruff]
line-length = 80                      # was 100
target-version = "py312"
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["kms"]
# Import convention: only import modules, not symbols.
# ruff's isort has no "reject from X import Y" rule, so symbol-level imports
# must be caught in review.

[tool.ruff.format]
quote-style = "single"
docstring-code-format = true
```

### 7.2 Running the formatter

```bash
uv run ruff format .
uv run ruff check .
```

---

## 8. Migration order (recommended)

1. **Ruff config** — change `line-length = 80`, add `quote-style = "single"`.
2. **Naming** — rename cryptic abbreviations across the codebase.
3. **Imports** — convert `from kms.core.models import ASTNode` → `from kms.core import models`.
4. **Docstrings** — add Google-style docstrings to every public function.
5. **Line length** — reflow long lines to 80 chars (run `ruff format`).

---

## 9. Quick reference — common patterns

| Current | Google replacement |
|---|---|
| `from kms.core.models import ASTNode, Entity` | `from kms.core import models` → `models.ASTNode` |
| `n = len(nodes)` | `node_count = len(nodes)` |
| `seg_index` | `segment_index` |
| `nid` | `node_id` |
| `lm=text_lm()` | `language_model=text_lm()` |
| `_est_tokens(node)` | `_estimate_tokens(node)` |
| `_dto(node)` | `_to_seam_node_dto(node)` |
| `def aforward(self, ...)` | Add docstring |
| Line >80 chars | Reflow (ruff format does this) |
| Triple-quoted string with `"""` | Keep; docstrings = `"""`, others = `'...'` |
