r"""
Rule checks over one entity-extraction run — pure functions, no LLM, no I/O.

Every check here restates a commitment the pass already makes in its own
prompt or docstring (``kms.ingestion.entity_extractor``) as something a
machine can count. Nothing is scored against a hand-labelled answer key: the
point is that most of what the prompt promises is checkable against the
source facts alone, so a prompt edit can be measured the same day it is made
rather than after someone labels a gold set.

Findings carry a severity, because the two kinds are not comparable:

* ``violation`` — an unambiguous breach. The mention names a fact that does
  not exist, repeats itself inside one fact, carries Unicode mathematics
  where the prompt demands LaTeX, or is a pronoun. No judgement is involved
  in calling these wrong.

* ``review`` — a heuristic reading of a rule whose real test is language.
  "Is this a bound symbol?" and "is this a derivation step?" are decided
  here by looking for an alphabetic run inside the LaTeX, which is a proxy
  and not the rule. ``$$e^{i\pi} + 1 = 0$$`` — a named formula the prompt
  explicitly KEEPS — trips the same wire as ``$0 > 18$``, which it drops.
  Review counts are a queue to read, never a score to minimise.

Recall is the one thing not measurable this way. ``facts_without_mentions``
is a proxy and is reported as coverage rather than as a finding: the prompt
allows a fact to mention nothing, so a bare count cannot separate correct
silence from a miss.
"""

import re
import unicodedata

# --- Severities -------------------------------------------------------------

VIOLATION = 'violation'
REVIEW = 'review'

# --- Patterns ---------------------------------------------------------------

# A LaTeX span with its delimiters: $$…$$ first so the inline form does not
# match half of a display span.
_MATH_SPAN = re.compile(r'\$\$.+?\$\$|\$.+?\$', re.DOTALL)

# A whole name that is nothing but one math span.
_ONLY_MATH = re.compile(r'^\s*(\$\$.+\$\$|\$.+\$)\s*$', re.DOTALL)

# LaTeX control sequences, stripped before looking for real words: \sqrt and
# \mathbb are notation, not vocabulary.
_LATEX_COMMAND = re.compile(r'\\[a-zA-Z]+')

# Three or more consecutive letters — the proxy for "a word, not a symbol".
_WORD_RUN = re.compile(r'[A-Za-z]{3,}')

# Relational operators, LaTeX and ASCII, that make a span a statement rather
# than a name.
_RELATION = re.compile(r'(?<![<>=!])=(?!=)|[<>]|\\leq|\\geq|\\neq|\\le|\\ge')

# The alternative delimiters the formatter is supposed to have converted.
_ALT_DELIMITERS = re.compile(r'\\\(|\\\)|\\\[|\\\]')

# An escaped dollar — a literal $, not a math delimiter.
_ESCAPED_DOLLAR = re.compile(r'\\\$')

# A name that is entirely a quoted phrase — the prompt's "gloss" case.
_QUOTED = re.compile(r'^\s*[\'"‘“].*[\'"’”]\s*$')

_PRONOUNS = frozenset(
    {
        'it',
        'this',
        'that',
        'they',
        'them',
        'these',
        'those',
        'he',
        'she',
        'we',
        'you',
        'its',
        'their',
    }
)

# Unicode blocks that mean "mathematics written as glyphs" — exactly what the
# formatter converts to LaTeX upstream and what this prompt forbids in a name.
_UNICODE_MATH_RANGES = (
    (0x00B1, 0x00B1),  # ±
    (0x00D7, 0x00D7),  # ×
    (0x00F7, 0x00F7),  # ÷
    (0x0370, 0x03FF),  # Greek
    (0x2070, 0x209F),  # super/subscripts
    (0x2100, 0x214F),  # letterlike (ℝ, ℕ, …)
    (0x2190, 0x21FF),  # arrows
    (0x2200, 0x22FF),  # mathematical operators
    (0x2A00, 0x2AFF),  # supplemental operators
)


def _is_unicode_math(char: str) -> bool:
    """Whether one character is a mathematical glyph rather than text.

    Args:
        char: The character to test.

    Returns:
        True when the character falls in a mathematical Unicode block.
    """
    point = ord(char)
    return any(low <= point <= high for low, high in _UNICODE_MATH_RANGES)


def _normalise(text: str) -> str:
    """Casefold and collapse whitespace for comparison.

    Args:
        text: The string to normalise.

    Returns:
        The normalised string.
    """
    return (
        re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text))
        .strip()
        .casefold()
    )


def _tokens(text: str) -> set[str]:
    """The alphanumeric tokens of a string, casefolded.

    Args:
        text: The string to tokenise.

    Returns:
        The token set.
    """
    return {token for token in re.findall(r'[\w\\]+', text.casefold()) if token}


# Length of the prefix two tokens must share to count as the same word. Crude
# on purpose: the alternative is a real stemmer, and the only thing this has
# to survive is morphology — "vertex" against a fact that says "vertices",
# "converges" against "convergent". Four characters separates those from
# genuine invention without pulling in a dependency.
_STEM = 4


def _grounded_tokens(name_tokens: set[str], fact_tokens: set[str]) -> int:
    """How many of a name's tokens appear in the fact, morphology allowed.

    Args:
        name_tokens: The name's tokens.
        fact_tokens: The fact's tokens.

    Returns:
        The number of name tokens with a match in the fact.
    """
    stems = {token[:_STEM] for token in fact_tokens}
    return len(
        [
            token
            for token in name_tokens
            if token in fact_tokens or token[:_STEM] in stems
        ]
    )


def word_count(name: str) -> int:
    """Count a name's words, treating each math span as one word.

    The 1-6 word rule is about natural language: ``$x = -b \\pm
    \\sqrt{b^2-4ac} / 2a$`` is one named thing however many whitespace runs
    it contains, so a math span counts once.

    Args:
        name: The entity name.

    Returns:
        The word count.
    """
    spans = _MATH_SPAN.findall(name)
    remainder = _MATH_SPAN.sub(' ', name)
    return len(spans) + len([word for word in remainder.split() if word])


def _symbolic_inner(name: str) -> str | None:
    """The inside of a name that is nothing but one math span.

    Args:
        name: The entity name.

    Returns:
        The span's contents with LaTeX commands stripped, or None when the
        name is not a lone math span.
    """
    if not _ONLY_MATH.match(name):
        return None
    inner = name.strip().strip('$')
    return _LATEX_COMMAND.sub(' ', inner)


def check_mention(
    name: str,
    fact_index: int,
    fact_text: str | None,
    fact_count: int,
    batch_indices: set[int] | None,
    seen_in_fact: set[str],
) -> list[tuple[str, str, str]]:
    """Every finding for one mention.

    Args:
        name: The emitted entity name.
        fact_index: The fact index the model attributed it to.
        fact_text: The text of that fact, or None when the index is invalid.
        fact_count: How many facts the record holds.
        batch_indices: The fact indices the mention's batch was given, or
            None to skip the batch-containment check.
        seen_in_fact: Normalised names already emitted for this fact; this
            call adds to it.

    Returns:
        One ``(check, severity, detail)`` triple per finding.
    """
    findings: list[tuple[str, str, str]] = []
    stripped = name.strip()

    if not stripped:
        findings.append(('empty_name', VIOLATION, 'name is blank'))
        return findings

    # --- Provenance ---------------------------------------------------
    if not 0 <= fact_index < fact_count:
        findings.append(
            (
                'index_out_of_range',
                VIOLATION,
                f'fact_index {fact_index} outside 0..{fact_count - 1}',
            )
        )
    elif batch_indices is not None and fact_index not in batch_indices:
        findings.append(
            (
                'index_outside_batch',
                VIOLATION,
                f'fact_index {fact_index} was not in this batch',
            )
        )

    # --- Uniqueness ---------------------------------------------------
    key = _normalise(stripped)
    if key in seen_in_fact:
        findings.append(
            (
                'duplicate_in_fact',
                VIOLATION,
                f'"{stripped}" already emitted for fact {fact_index}',
            )
        )
    seen_in_fact.add(key)

    # --- Format -------------------------------------------------------
    # `\$` is an escaped literal dollar — currency, not a delimiter. A page
    # that says "a budget surplus of \$540 million" yields a name carrying one
    # escaped dollar, which is correct and must not read as unbalanced.
    if _ESCAPED_DOLLAR.sub('', stripped).count('$') % 2:
        findings.append(
            ('unbalanced_latex', VIOLATION, 'odd number of $ delimiters')
        )
    if _ALT_DELIMITERS.search(stripped):
        findings.append(
            (
                'alt_delimiters',
                VIOLATION,
                r'uses \( \) or \[ \] instead of the dollar convention',
            )
        )
    glyphs = sorted({c for c in stripped if _is_unicode_math(c)})
    if glyphs:
        findings.append(
            (
                'unicode_math',
                VIOLATION,
                f'Unicode mathematics in the name: {"".join(glyphs)}',
            )
        )

    # --- Vocabulary ---------------------------------------------------
    if key in _PRONOUNS:
        findings.append(
            ('pronoun_name', VIOLATION, f'"{stripped}" is a pronoun')
        )
    if _QUOTED.match(stripped):
        findings.append(
            ('gloss_quoted', REVIEW, 'name is a quoted phrase — likely a gloss')
        )

    words = word_count(stripped)
    if words > 6:
        findings.append(
            ('too_many_words', REVIEW, f'{words} words, the rule says 1-6')
        )

    # --- Symbols and derivation steps ---------------------------------
    inner = _symbolic_inner(stripped)
    if inner is not None and not _WORD_RUN.search(inner):
        if _RELATION.search(inner):
            findings.append(
                (
                    'pure_relation',
                    REVIEW,
                    'a relation with no named object — derivation step, or a '
                    'named formula the prompt keeps',
                )
            )
        else:
            findings.append(
                (
                    'bare_symbol',
                    REVIEW,
                    'a lone symbol with no word in it — the variable tier owns '
                    'these',
                )
            )

    # --- Grounding ----------------------------------------------------
    if fact_text is not None:
        if _normalise(stripped) not in _normalise(fact_text):
            name_tokens = _tokens(stripped)
            overlap = (
                _grounded_tokens(name_tokens, _tokens(fact_text))
                / len(name_tokens)
                if name_tokens
                else 0.0
            )
            if overlap < 0.5:
                findings.append(
                    (
                        'ungrounded',
                        VIOLATION,
                        f"{overlap:.0%} of the name's tokens appear in the fact",
                    )
                )
            else:
                findings.append(
                    (
                        'not_verbatim',
                        REVIEW,
                        'name is not a substring of the fact — the prompt says '
                        "keep the fact's wording",
                    )
                )

    return findings
