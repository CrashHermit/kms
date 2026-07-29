r"""
Hand-annotated gold dataset for the corrector, and the tooling around it.

The corrector (``kms.ingestion.corrector``) is a generate-then-verify pass: a vision
model re-reads a page image alongside Mistral's markdown and fixes genuine transcription
errors. Optimising that module with DSPy needs a training set whose ``corrected`` labels
are *right* — an optimiser bootstrapped from a teacher model's guesses inherits the
teacher's failure modes, which is exactly what we are trying to remove. So this dataset
is annotated by hand: a human-or-strong-model annotator reads each page render, compares
it token by token against Mistral's transcription, and writes the corrected markdown
itself.

Workspace layout
----------------
Each gold sample is a directory under ``output/gold/corrector/``::

    <book>-p<NN>/
        page.png      the page render (the ground truth the annotator reads)
        raw.md        Mistral's transcription, verbatim  -> the `transcription` input
        gold.md       the annotator's corrected markdown -> the `corrected` label
        notes.json    the edit log: one entry per correction made (see `Edit`)

``raw.md`` and ``page.png`` are produced mechanically by ``stage``; ``gold.md`` and
``notes.json`` are written by the annotator. A directory with no ``gold.md`` is simply
un-annotated and is skipped by ``check`` and ``build``.

The edit log is not decoration. It is what makes the dataset reviewable — a diff alone
does not say *why* a change was made, and a gold label nobody can audit is not gold. It
also gives per-error-mode counts, so we can see which failure modes the set actually
covers before spending an optimiser run on it.

Canonical form
--------------
Every ``gold.md`` is written in one canonical markdown form, whatever Mistral happened
to emit for that page. This is the point of a gold set: if the labels vary page to page —
math in LaTeX here and in Unicode there, part-labels as ``ⓐ`` on one page and ``(a)`` on
the next — the optimiser learns the variation rather than the task, and every downstream
stage has to match both spellings.

The rules, enforced by ``check``:

1. **Math is LaTeX in dollars.** ``$ … $`` inline, ``$$ … $$`` display. No bare Unicode
   operators (``≥``, ``÷``, ``·``, ``∈``, ``√``) and no Unicode super/subscript digits
   (``²``, ``⁵``, ``ₙ``) outside a math span — those are ``\geq``, ``\div``, ``\cdot``,
   ``\in``, ``\sqrt``, ``^2``, ``^5``, ``_n`` inside one.
2. **Part-labels are** ``(a)``, ``(b)``, ``(c)``, ``(d)``. Circled forms (``ⓐ``, U+24D0)
   are a faithful reading of the page but the wrong encoding for the pipeline; the
   corpus's other books already print ``(a)``.
3. **Quotes and apostrophes are ASCII.** Mistral emits curly and straight forms
   interchangeably — sometimes on facing pages of one book — so the label picks one.
4. **Emphasis is never invented.** Where the OCR dropped a bold label or an italic
   defined term, gold leaves it dropped. Restoring it is reformatting, which the
   corrector's contract forbids, and teaching the model to add markup is how a
   proofreader turns into a rewriter.

Rules 1–3 are mechanical and live in ``canonicalize``; rule 4 is a judgement the
annotator makes and the edit log records.

Natural vs. synthetic samples
-----------------------------
Mistral transcribes most pages correctly, so an annotate-what-you-find dataset is
overwhelmingly "the page was already clean, return it unchanged". Those negatives matter
(the corrector's worst failure is rewriting correct text), but on their own they teach
nothing about the errors we built the stage to catch.

So the set has two kinds of sample, distinguished by ``meta.json``'s ``kind``:

- ``natural`` — ``raw.md`` is Mistral's real output. ``gold.md`` fixes whatever it
  actually got wrong, which is often nothing beyond delimiter normalization.
- ``synthetic`` — ``raw.md`` is a *verified-correct* transcription into which a known
  failure mode was deliberately injected (a plain ``\sqrt{x}`` given a false index, a
  subscript reattached to the wrong symbol, a Greek letter swapped). ``gold.md`` is the
  verified text. The injection is recorded in ``notes.json`` like any other edit, so a
  synthetic sample is auditable in exactly the same way.

Synthetic samples are only ever derived from a page whose gold has already been read
against the image — we corrupt known-good text, never guessed-good text.

Commands
--------
::

    python -m kms.training.gold_corrector stage tests/fixtures/books/*.pdf
    python -m kms.training.gold_corrector inject <sample-dir> <name>
    python -m kms.training.gold_corrector check
    python -m kms.training.gold_corrector build

``stage`` runs Mistral OCR and the page render; ``inject`` derives a synthetic sample
from an annotated one; ``check`` validates every annotated sample against the invariants
the corrector itself enforces at runtime; ``build`` writes the DSPy training set to
``output/examples/corrector.json`` in the format ``kms.core.recorder`` reads back.
"""

import base64
import dataclasses
import difflib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from kms.core import recorder
from kms.ingestion import corrector, ocr

# The gold set is hand-made and irreplaceable, so it lives in the tracked tree rather
# than under the gitignored `output/` scratch area. Only the text is committed: the page
# renders are a deterministic function of a fixture PDF, a page index, and
# `ocr.RENDER_SCALE`, all recorded in each sample's meta.json, so `render` recreates them
# on demand instead of putting ~11 MB of PNGs in git.
GOLD_DIR = Path('data/gold/corrector')

# Every edit recorded in a notes.json must carry one of these categories. A closed
# vocabulary keeps the coverage report meaningful — free-text categories drift into
# synonyms and stop being countable.
CATEGORIES = frozenset(
    {
        'root_index',  # \sqrt{x} <-> \sqrt[n]{x}: an index invented, dropped, or misread
        'sub_superscript',  # attached to the wrong symbol, or dropped: f(x)_1 vs f(x_1)
        'operator',  # wrong operator or relation: + / \pm, \le / <, \to / \mapsto
        'symbol',  # wrong letter or glyph: Greek, \ell vs l, a digit misread
        'delimiter',  # wrong or unbalanced bracket/paren/brace in the math itself
        'math_delimiters',  # \( \) or \[ \] -> $ $$, or undelimited display math wrapped
        'structure',  # markdown structure: heading level, list nesting, table shape
        'dropped_text',  # words or a whole line the OCR lost
        'spurious_text',  # text the OCR invented, or page chrome it left inline
        'spelling',  # a plain misspelling of a word that is legible in the image
        'presentation_glyph',  # a typographic glyph normalized to its canonical form:
        # circled part-labels (ⓐ -> (a)), Unicode super/subscript digits, and the like.
        # The glyph is a faithful reading of the page; it is the *encoding* that is wrong
        # for the pipeline, since downstream stages match on `(a)` and on LaTeX, not on
        # U+24D0. Distinct from `symbol`, where the OCR read the wrong character.
    }
)


@dataclasses.dataclass(frozen=True)
class Edit:
    """One correction the annotator made, as recorded in ``notes.json``.

    Attributes:
        category: One of ``CATEGORIES``, naming the failure mode.
        image_shows: What the page render actually shows, quoted.
        ocr_had: What ``raw.md`` said instead, quoted.
        note: Why the change is right — the reasoning a reviewer needs to check it.
    """

    category: str
    image_shows: str
    ocr_had: str
    note: str = ''


# --- canonical form (rules 1-3 of the module docstring) ----------------------------

# Circled part-labels -> the corpus's parenthesized convention.
_PART_LABELS = (('ⓐ', '(a)'), ('ⓑ', '(b)'), ('ⓒ', '(c)'), ('ⓓ', '(d)'))

# Curly punctuation -> ASCII.
_QUOTES = (('“', '"'), ('”', '"'), ('‘', "'"), ('’', "'"))

# Characters that belong inside a math span, keyed to the LaTeX they should be written
# as. Used only to *report* violations — rewriting them needs the surrounding expression
# to be delimited too, which is the annotator's judgement, not a substitution.
_MATH_ONLY = {
    '×': r'\times',
    '÷': r'\div',
    '±': r'\pm',
    '∓': r'\mp',
    '≥': r'\geq',
    '≤': r'\leq',
    '≠': r'\neq',
    '≈': r'\approx',
    '∈': r'\in',
    '∉': r'\notin',
    '⊂': r'\subset',
    '⊆': r'\subseteq',
    '∪': r'\cup',
    '∩': r'\cap',
    '∅': r'\emptyset',
    '∞': r'\infty',
    '√': r'\sqrt',
    '∑': r'\sum',
    '∏': r'\prod',
    '∫': r'\int',
    '→': r'\to',
    '↔': r'\leftrightarrow',
    '∀': r'\forall',
    '∃': r'\exists',
    '−': '-',
    '·': r'\cdot',
    '⁰': '^0',
    '¹': '^1',
    '²': '^2',
    '³': '^3',
    '⁴': '^4',
    '⁵': '^5',
    '⁶': '^6',
    '⁷': '^7',
    '⁸': '^8',
    '⁹': '^9',
    'ⁿ': '^n',
    '₀': '_0',
    '₁': '_1',
    '₂': '_2',
    '₃': '_3',
    '₄': '_4',
    '₅': '_5',
    '₆': '_6',
    '₇': '_7',
    '₈': '_8',
    '₉': '_9',
    'ₙ': '_n',
}

# A `$$ … $$` or `$ … $` span. Display is matched first so a display pair is never read
# as two empty inline ones.
_MATH_SPAN = re.compile(r'\$\$.*?\$\$|\$.*?\$', re.DOTALL)


def canonicalize(text: str) -> str:
    """Apply the mechanical canonical-form rules (part-labels and quotes) to a label.

    Math normalization (rule 1) is deliberately not done here: wrapping a bare Unicode
    expression in dollars means deciding where the expression starts and ends, which
    needs the page image. ``check`` reports what is left instead.

    Args:
        text: A gold transcription.

    Returns:
        The same text with circled part-labels parenthesized and curly quotes flattened.
    """
    for glyph, canonical in _PART_LABELS + _QUOTES:
        text = text.replace(glyph, canonical)
    return text


def _math_glyphs_outside_math(text: str) -> list[str]:
    """The math-only characters appearing in ``text`` outside any ``$``-delimited span."""
    prose = _MATH_SPAN.sub('', text)
    return sorted({char for char in prose if char in _MATH_ONLY})


def _slug(pdf_path: Path, page: int) -> str:
    """The workspace directory name for one page of one book."""
    return f'{pdf_path.stem}-p{page:02d}'


# --- render: recreate the page images the committed text refers to -----------------


def render(force: bool = False) -> list[Path]:
    """Recreate every sample's ``page.png`` from its recorded source PDF and page.

    The renders are not committed (see ``GOLD_DIR``), so this is what a fresh checkout
    runs before ``build`` — or before annotating, since the image is the ground truth the
    annotator reads. Rendering goes through the same ``ocr.RENDER_SCALE`` the corrector
    was validated on, so a regenerated image is byte-comparable to the original.

    Args:
        force: Re-render samples that already have a ``page.png``.

    Returns:
        The sample directories whose image was written.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            'pypdfium2 is required to render the gold page images. '
            'Install the Mistral front-end deps:  uv sync --extra mistral'
        ) from exc

    written: list[Path] = []
    documents: dict[str, object] = {}
    try:
        for sample_dir in sorted(GOLD_DIR.iterdir()):
            meta_path = sample_dir / 'meta.json'
            if not meta_path.exists():
                continue
            image_path = sample_dir / 'page.png'
            if image_path.exists() and not force:
                continue
            meta = json.loads(meta_path.read_text())
            source = meta['source_pdf']
            if source not in documents:
                documents[source] = pdfium.PdfDocument(source)
            page = documents[source][meta['source_page']]
            page.render(scale=ocr.RENDER_SCALE).to_pil().save(image_path)
            written.append(sample_dir)
    finally:
        for document in documents.values():
            document.close()
    return written


# --- stage: PDF -> page.png + raw.md ---------------------------------------------


def stage(pdf_path: str | Path, pages: list[int] | None = None) -> list[Path]:
    """OCR a fixture PDF and lay out one un-annotated sample directory per page.

    The OCR runs into a throwaway directory so the pipeline's own ``output/Segments``
    working area is never disturbed; only the page render and the markdown are kept.
    Existing ``gold.md``/``notes.json`` are left alone, so re-staging a book refreshes
    the transcription without destroying annotation work.

    Args:
        pdf_path: The source PDF.
        pages: 0-based page numbers to stage. None stages the whole document.

    Returns:
        The sample directories written, in page order.
    """
    pdf_path = Path(pdf_path)
    staged: list[Path] = []
    with tempfile.TemporaryDirectory() as scratch:
        segments = ocr.extract(pdf_path, output_dir=scratch, pages=pages)
        for position, segment in enumerate(segments):
            page = pages[position] if pages is not None else position
            sample_dir = GOLD_DIR / _slug(pdf_path, page)
            sample_dir.mkdir(parents=True, exist_ok=True)

            shutil.copyfile(segment.image_path, sample_dir / 'page.png')
            (sample_dir / 'raw.md').write_text(segment.content or '')
            (sample_dir / 'meta.json').write_text(
                json.dumps(
                    {
                        'kind': 'natural',
                        'source_pdf': str(pdf_path),
                        'source_page': page,
                    },
                    indent=2,
                )
                + '\n'
            )
            staged.append(sample_dir)
    return staged


# --- inject: annotated sample -> synthetic sample --------------------------------


def inject(sample_dir: str | Path, name: str) -> Path:
    """Create an empty synthetic sample derived from an annotated one.

    The new directory shares the source page image and starts with ``raw.md`` set to
    the *parent's gold* — verified-correct text. The annotator then edits that ``raw.md``
    to inject a failure mode and records the injection in ``notes.json``; ``gold.md`` is
    the untouched verified text, so the label is correct by construction.

    Args:
        sample_dir: An annotated sample directory to derive from.
        name: Suffix for the synthetic sample (e.g. ``root_index``).

    Returns:
        The synthetic sample directory.

    Raises:
        FileNotFoundError: If the parent sample has no ``gold.md`` to derive from.
    """
    sample_dir = Path(sample_dir)
    gold_path = sample_dir / 'gold.md'
    if not gold_path.exists():
        raise FileNotFoundError(
            f'{sample_dir} has no gold.md; a synthetic sample may only be derived '
            'from text that has already been verified against the page image.'
        )

    target = sample_dir.parent / f'{sample_dir.name}--{name}'
    target.mkdir(parents=True, exist_ok=True)
    verified = gold_path.read_text()
    shutil.copyfile(sample_dir / 'page.png', target / 'page.png')
    # raw.md starts as the verified text; the annotator corrupts it in place.
    (target / 'raw.md').write_text(verified)
    (target / 'gold.md').write_text(verified)
    parent_meta = json.loads((sample_dir / 'meta.json').read_text())
    (target / 'meta.json').write_text(
        json.dumps(
            {
                **parent_meta,
                'kind': 'synthetic',
                'derived_from': sample_dir.name,
            },
            indent=2,
        )
        + '\n'
    )
    return target


# --- check: validate the annotated samples ---------------------------------------


def _unbalanced_dollars(text: str) -> bool:
    """True when the page's ``$``/``$$`` math delimiters do not pair up.

    ``$$`` is counted first so a display pair is not miscounted as two inline ones.
    A page whose delimiters are unbalanced would hand malformed math to the extractor,
    so it is never acceptable in a gold label.
    """
    stripped = text.replace(r'\$', '')
    display = stripped.count('$$')
    inline = stripped.replace('$$', '').count('$')
    return display % 2 != 0 or inline % 2 != 0


def _read_edits(sample_dir: Path) -> list[Edit]:
    """Load a sample's edit log, or an empty log when it has none."""
    path = sample_dir / 'notes.json'
    if not path.exists():
        return []
    return [Edit(**entry) for entry in json.loads(path.read_text())]


def annotated_samples() -> list[Path]:
    """Every sample directory that has a ``gold.md``, in name order."""
    if not GOLD_DIR.exists():
        return []
    return sorted(
        path for path in GOLD_DIR.iterdir() if (path / 'gold.md').exists()
    )


def check() -> list[str]:
    """Validate every annotated sample and return the problems found.

    The checks are the invariants the corrector enforces at runtime plus the ones the
    dataset itself needs to be auditable:

    - the gold survives the runtime divergence guard (``_within_tolerance``) — a label
      the corrector would reject at inference is worse than useless as a target;
    - the gold is already delimiter-normalized, so the node's deterministic
      ``_normalize_math_delimiters`` is a no-op on it and training target and runtime
      output agree;
    - ``$``/``$$`` pair up;
    - a gold that differs from its raw has an edit log, every edit uses a known
      category, and a gold identical to its raw claims no edits.

    Returns:
        One human-readable message per problem; empty when the dataset is clean.
    """
    problems: list[str] = []
    for sample_dir in annotated_samples():
        raw = (sample_dir / 'raw.md').read_text()
        gold = (sample_dir / 'gold.md').read_text()
        edits = _read_edits(sample_dir)
        name = sample_dir.name

        if not corrector._within_tolerance(raw, gold):
            problems.append(
                f"{name}: gold is outside the corrector's ±30% length guard "
                f'({len(raw)} chars raw, {len(gold)} gold) — the runtime would '
                'reject this correction.'
            )
        if corrector._normalize_math_delimiters(gold) != gold:
            problems.append(
                f'{name}: gold still contains \\( \\) or \\[ \\] math delimiters; '
                'the label must already be normalized to $ / $$.'
            )
        if _unbalanced_dollars(gold):
            problems.append(f'{name}: unbalanced $ / $$ in gold.')
        if canonicalize(gold) != gold:
            problems.append(
                f'{name}: gold is not in canonical form — circled part-labels or '
                'curly quotes remain (see `canonicalize`).'
            )
        stray = _math_glyphs_outside_math(gold)
        if stray:
            problems.append(
                f'{name}: math-only glyph(s) outside a math span in gold: '
                + ', '.join(
                    f'{glyph!r} (write as {_MATH_ONLY[glyph]})'
                    for glyph in stray
                )
            )
        if gold != raw and not edits:
            problems.append(
                f'{name}: gold differs from raw but notes.json records no edits.'
            )
        if gold == raw and edits:
            problems.append(
                f'{name}: notes.json records {len(edits)} edit(s) but gold is '
                'byte-identical to raw.'
            )
        for edit in edits:
            if edit.category not in CATEGORIES:
                problems.append(
                    f'{name}: unknown edit category {edit.category!r}.'
                )
    return problems


def coverage() -> dict[str, int]:
    """Count the recorded edits per failure-mode category across the dataset."""
    counts: dict[str, int] = {}
    for sample_dir in annotated_samples():
        for edit in _read_edits(sample_dir):
            counts[edit.category] = counts.get(edit.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def diff(sample_dir: str | Path) -> str:
    """A unified diff of one sample's raw -> gold, for reviewing an annotation."""
    sample_dir = Path(sample_dir)
    return ''.join(
        difflib.unified_diff(
            (sample_dir / 'raw.md').read_text().splitlines(keepends=True),
            (sample_dir / 'gold.md').read_text().splitlines(keepends=True),
            fromfile=f'{sample_dir.name}/raw.md',
            tofile=f'{sample_dir.name}/gold.md',
        )
    )


# --- build: samples -> output/examples/corrector.json ------------------------------


def build(output_dir: str = 'output/examples') -> int:
    """Write the annotated samples out as a DSPy training set.

    The file is rewritten from scratch (not appended to) so the gold set on disk is
    always exactly what the workspace holds — ``recorder.record_example`` appends, which
    would duplicate every sample on a second run.

    Args:
        output_dir: Directory to write ``corrector.json`` into.

    Returns:
        The number of examples written.

    Raises:
        RuntimeError: If ``check`` finds any problem — a dataset that fails its own
            invariants must not reach an optimiser.
    """
    problems = check()
    if problems:
        raise RuntimeError(
            'refusing to build a gold set that fails validation:\n  '
            + '\n  '.join(problems)
        )

    path = Path(output_dir) / 'corrector.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[]')

    samples = annotated_samples()
    for sample_dir in samples:
        encoded = base64.b64encode((sample_dir / 'page.png').read_bytes())
        recorder.record_example(
            'corrector',
            {
                'page_image': f'data:image/png;base64,{encoded.decode("utf-8")}',
                'transcription': (sample_dir / 'raw.md').read_text(),
            },
            {'corrected': (sample_dir / 'gold.md').read_text()},
            output_dir=output_dir,
        )
    return len(samples)


# --- CLI ---------------------------------------------------------------------------


def main() -> None:
    """Dispatch the ``stage`` / ``inject`` / ``check`` / ``build`` subcommands."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command, args = sys.argv[1], sys.argv[2:]

    if command == 'stage':
        for pdf_path in args:
            for sample_dir in stage(pdf_path):
                print(f'staged {sample_dir}')
    elif command == 'inject':
        print(f'created {inject(args[0], args[1])}')
    elif command == 'check':
        problems = check()
        for problem in problems:
            print(f'FAIL {problem}')
        print(
            f'{len(annotated_samples())} annotated sample(s), '
            f'{len(problems)} problem(s)'
        )
        for category, count in coverage().items():
            print(f'  {category}: {count}')
        sys.exit(1 if problems else 0)
    elif command == 'render':
        written = render(force='--force' in args)
        print(f'rendered {len(written)} page image(s)')
    elif command == 'diff':
        print(diff(args[0]))
    elif command == 'build':
        print(f'wrote {build()} example(s) to output/examples/corrector.json')
    else:
        print(f'unknown command {command!r}')
        sys.exit(1)


if __name__ == '__main__':
    main()
