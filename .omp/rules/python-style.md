---
alwaysApply: true
description: Apply the repository's Python style conventions.
---

For Python files, follow `docs/STYLE.md`.

Before creating or modifying Python files:

- Follow the guide's naming, typing, import, formatting, and documentation rules.
- Preserve the conventions of nearby code.
- Run the relevant formatter, linter, and tests after meaningful changes.
- Keep the style guide general; do not add architecture-specific rules to it.

Use the repository's configured tools as the source of truth for automated
formatting and linting. Treat this rule as guidance, not a substitute for
Ruff, Pyright, tests, or CI enforcement.
