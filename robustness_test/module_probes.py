"""Live contract tests for each DSPy module, in isolation.

Every probe drives ONE module directly against hand-built input and asserts the contract its
Signature/docstring claims — including the hazards docs/HANDOFF.md records as known. This is
the per-module counterpart to the end-to-end sweep in
robustness_test/ENTITY-REBUILD-VALIDATION.md.

DSPy's caches are disabled, so every probe is a real call (see the HANDOFF gotcha: a cached
re-run returns byte-identical output in milliseconds and looks like determinism).
"""

import asyncio
import sys

import dspy

from kms.core import models
from kms.entity import (
    block_typer,
    group_finder,
    instruction_distributor,
    instruction_finder,
    procedure_extractor,
    role_typer,
    splitter,
    statement_extractor,
)
from kms.ingestion import corrector, extractor, seam_merger

RESULTS: list[tuple[str, str, bool, str]] = []


def check(module: str, name: str, ok: bool, detail: str) -> None:
    RESULTS.append((module, name, bool(ok), detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')


def nodes(*specs) -> list[models.ASTNode]:
    return [
        models.ASTNode(type=models.NodeType(t), content=c, id=i)
        for i, (t, c) in enumerate(specs)
    ]


def window(ns) -> list:
    return [
        splitter.WindowNode(
            position=i, type=(n.type.value if n.type else ''), content=n.content
        )
        for i, n in enumerate(ns)
    ]


# --- 1. extractor: purely structural, no semantic typing -----------------------------


async def probe_extractor() -> None:
    print('\n=== extractor ===')
    markdown = (
        '## 3.2 Continuity\n\n'
        '**Theorem 3.4.** If $f$ is continuous on $[a,b]$ then $f$ is bounded.\n\n'
        '*Proof.* Suppose not. Then for each $n$ there is $x_n$ with $|f(x_n)| > n$.\n\n'
        '$$\\lim_{n\\to\\infty} x_n = x^*$$\n\n'
        '1. Show $f(x)=x^2$ is continuous.\n'
        '2. Show $f(x)=1/x$ is not continuous at $0$.\n'
    )
    out = await extractor.Module().aforward(segment_markdown=markdown)
    kinds = {str(n.type) for n in out}
    structural = {t.value for t in models.NodeType}
    check(
        'extractor',
        'emits only structural node types',
        kinds <= structural,
        f'{len(out)} nodes, types={sorted(kinds)}',
    )
    joined = ' '.join((n.content or '') for n in out)
    check(
        'extractor',
        'preserves the display-math block verbatim',
        '\\lim_{n\\to\\infty} x_n = x^*' in joined,
        'LaTeX survived' if '\\lim' in joined else 'LaTeX LOST',
    )


# --- 2. seam_merger: merge a true split, leave a true boundary alone -----------------


async def probe_seam() -> None:
    print('\n=== seam_merger ===')
    module = seam_merger.Module()

    split = await module.aforward(
        top_bottom_edge_node=seam_merger.SeamNodeDTO(
            content='The sequence is bounded, and therefore has a convergent',
            types=['paragraph'],
        ),
        bottom_top_edge_node=seam_merger.SeamNodeDTO(
            content='subsequence by the Bolzano-Weierstrass theorem.',
            types=['paragraph'],
        ),
    )
    merged = split is not None and bool(getattr(split, 'content', None))
    check(
        'seam_merger',
        'merges a sentence split across the page break',
        merged,
        (getattr(split, 'content', '') or 'no merge')[:90],
    )

    apart = await module.aforward(
        top_bottom_edge_node=seam_merger.SeamNodeDTO(
            content='This completes the proof. $\\square$',
            types=['paragraph'],
        ),
        bottom_top_edge_node=seam_merger.SeamNodeDTO(
            content='**Definition 3.7.** A set is *compact* if every open cover has a finite subcover.',
            types=['paragraph'],
        ),
    )
    left = apart is None or not getattr(apart, 'content', None)
    check(
        'seam_merger',
        'leaves a genuine boundary unmerged',
        left,
        'left split' if left else f'WRONGLY merged: {apart.content[:70]}',
    )


# --- 3. splitter: split a packed run, leave a lone exercise alone --------------------


async def probe_splitter() -> None:
    print('\n=== splitter ===')
    module = splitter.Module()

    packed = nodes(
        (
            'list',
            '1.23 Find the derivative of $f(x)=x^3$.\n'
            '1.24 Find the derivative of $g(x)=\\sin x$.\n'
            '1.25 Find the derivative of $h(x)=e^{2x}$.',
        )
    )
    out = await module.aforward(window(packed))
    pieces = out[0].exercises if out else []
    check(
        'splitter',
        'splits a node packing three exercises',
        len(out) == 1 and len(pieces) == 3,
        f'{len(out)} split(s), {len(pieces)} piece(s): {[p.number for p in pieces]}',
    )

    lone = nodes(
        (
            'paragraph',
            '**Example 4.2.** Compute $\\int_0^1 x^2 dx$. We apply the power rule.',
        )
    )
    out2 = await module.aforward(window(lone))
    check(
        'splitter',
        'leaves a single worked example unsplit',
        not out2,
        'no split'
        if not out2
        else f'WRONGLY split into {len(out2[0].exercises)}',
    )

    embedded = nodes(
        (
            'list',
            '7. Graph $y=x^2$.\n'
            'In Exercises 8-9, find the domain and range.\n'
            '8. $y=\\sqrt{x}$.\n'
            '9. $y=1/x$.',
        )
    )
    out3 = await module.aforward(window(embedded))
    got = out3[0].exercises if out3 else []
    lead = [p for p in got if not (p.number or '').strip()]
    check(
        'splitter',
        'breaks an embedded lead-in onto its own piece',
        len(lead) >= 1,
        f'{len(got)} piece(s), {len(lead)} numberless (lead-in)',
    )


# --- 4. instruction_finder: tag a real lead-in, not a plain exercise -----------------


async def probe_instruction_finder() -> None:
    print('\n=== instruction_finder ===')
    module = instruction_finder.Module()
    stream = nodes(
        ('header', '2.4 Exercises'),
        ('paragraph', 'In Exercises 3-8, graph the given relation.'),
        ('paragraph', '3. $y = 2x + 1$'),
        ('paragraph', '4. $y = x^2 - 3$'),
        ('paragraph', '9. Prove that the sum of two even integers is even.'),
    )
    positions = await module.aforward(
        [
            instruction_finder.WindowNode(
                position=i,
                type=(n.type.value if n.type else ''),
                content=n.content,
            )
            for i, n in enumerate(stream)
        ]
    )
    check(
        'instruction_finder',
        'tags the lead-in and nothing else',
        positions == [1],
        f'tagged positions {positions} (expected [1])',
    )


# --- 5. group_finder: cuts the statement/derivation boundary, marked or not ------------


async def probe_group_finder() -> None:
    """The finder emits UNTYPED spans now: assert it CUTS in the right place."""
    print('\n=== group_finder ===')
    module = group_finder.Module()

    marked = nodes(
        (
            'header',
            '**Theorem 2.1.** Every bounded monotone sequence converges.',
        ),
        (
            'paragraph',
            '*Proof.* Let $S$ be the set of terms. By completeness $S$ has a supremum $L$.',
        ),
        (
            'paragraph',
            'Given $\\epsilon>0$ there is $N$ with $a_N > L-\\epsilon$. Hence $a_n \\to L$. $\\square$',
        ),
    )
    spans = await group_finder.find_spans(marked, module=module)
    check(
        'group_finder',
        'MARKED derivation -> cut into two spans',
        len(spans) == 2,
        f'{len(spans)} span(s): {spans}',
    )

    unmarked = nodes(
        ('header', "**Example 3.2.** Solve $y' = y^2$, $y(0)=A$."),
        (
            'paragraph',
            'We know how to solve this. Assume $A \\neq 0$, so $x = -1/y + C$, giving $y = 1/(C-x)$.',
        ),
        (
            'paragraph',
            'If $A = 0$ then $y = 0$ is a solution. The solution blows up at $x=1/A$.',
        ),
    )
    spans2 = await group_finder.find_spans(unmarked, module=module)
    check(
        'group_finder',
        'UNMARKED derivation -> cut into two spans',
        len(spans2) >= 2,
        f'{len(spans2)} span(s): {spans2}',
    )

    # Regression: a bare labelled definition with nothing worked out after it got NO span
    # at all, which deleted it from the document.
    bare = nodes(
        (
            'paragraph',
            'Definition 2.5.1 (Primitive Root). A primitive root modulo $n$ is an '
            'element of $(\\mathbf{Z}/n\\mathbf{Z})^*$ of maximal order.',
        ),
        ('paragraph', 'We now turn to the question of existence.'),
    )
    spans3 = await group_finder.find_spans(bare, module=module)
    check(
        'group_finder',
        'a bare labelled definition still gets its own span',
        any(0 in span for span in spans3),
        f'{len(spans3)} span(s): {spans3}',
    )


# --- 5b. role_typer: the closed block/derivation call ---------------------------------


async def probe_role_typer() -> None:
    print('\n=== role_typer ===')
    module = role_typer.Module()

    cases = [
        (
            'a stated theorem',
            'entity',
            '**Theorem 2.1.** Every bounded monotone sequence converges.',
        ),
        (
            'a marked proof',
            'procedure',
            '*Proof.* Let $S$ be the set of terms. By completeness $S$ has a supremum $L$. '
            'Hence $a_n \\to L$. $\\square$',
        ),
        (
            'an UNMARKED worked solution',
            'procedure',
            'We know how to solve this. Assume $A \\neq 0$, so $x = -1/y + C$, giving '
            '$y = 1/(C-x)$. If $A = 0$ then $y = 0$ is a solution.',
        ),
        (
            'a posed exercise',
            'entity',
            "12. Sketch the slope field for $y' = e^{x-y}$.",
        ),
        (
            'a bare definition',
            'entity',
            '**Definition 3.7.** A set is *compact* if every open cover has a finite subcover.',
        ),
        # Regression: a LABELLED example containing a worked session was read as a
        # derivation, demoting Stein's SAGE examples into procedures of the block above.
        (
            'a LABELLED example holding a worked session',
            'entity',
            '*SAGE Example 2.5.4.* We use Sage to find the roots of a polynomial.\n\n'
            'sage: f = x^15 + 1\nsage: f.roots()\n[(12, 1), (10, 1), (4, 1)]',
        ),
        # Regression, the other direction: over-correcting the above made an UNLABELLED
        # session an entity. A session with no label of its own is the block's working.
        (
            'an UNLABELLED computation session',
            'procedure',
            'sage: R.<x> = PolynomialRing(Integers(13))\nsage: f = x^15 + 1\n'
            'sage: f.roots()\n[(12, 1), (10, 1), (4, 1)]\n\n'
            'The output above lists each root with its multiplicity.',
        ),
        # Working is not only algebra: text that EXHIBITS the answer (Lebl 1.2.2) or
        # ANALYSES the posed figures (Levin 2.1.7) is still the block's derivation.
        (
            'text EXHIBITING the answer with no algebra',
            'procedure',
            'See Figure 1.7 on the next page. Note that $y = 0$ is a solution. But '
            'another solution is the function $y(x) = x^2$ for $x \\geq 0$.',
        ),
        (
            'text ANALYSING the posed example',
            'procedure',
            'Here both $G_2$ and $G_3$ are subgraphs of $G_1$. But only $G_2$ is an '
            '*induced* subgraph. The graph $G_4$ is NOT a subgraph of $G_1$.',
        ),
    ]
    for name, expected, text in cases:
        got = await module.role(text)
        check(
            'role_typer',
            f'{name} -> {expected}',
            got == expected,
            f'got {got!r}',
        )


# --- 6. block_typer + statement_extractor: typing and the number hazard ---------------


async def probe_block_typer() -> None:
    print('\n=== block_typer ===')
    module = block_typer.Module()

    law = nodes(
        ('header', "Law 4.1 (Ohm's Law)."),
        (
            'paragraph',
            'The current through a conductor is proportional to the voltage across it, $V = IR$.',
        ),
    )
    induced = await module.block_type(law)
    check(
        'block_typer',
        'induces an open non-math type',
        induced not in ('', None, 'theorem', 'definition', 'example'),
        f'type={induced!r}',
    )

    bare = nodes(('paragraph', '$P \\vee (Q \\Rightarrow R)$'))
    induced2 = await module.block_type(bare)
    check(
        'block_typer',
        'types a bare formula exercise as an exercise',
        induced2 in ('exercise', 'problem'),
        f'type={induced2!r}',
    )

    assertion = nodes(
        (
            'paragraph',
            'For matrix $A$ to be invertible, it is necessary and sufficient that '
            '$\\det(A) \\neq 0$.',
        ),
    )
    induced3 = await module.block_type(assertion)
    check(
        'block_typer',
        'types an exercise whose body is an assertion as an exercise',
        induced3 in ('exercise', 'problem'),
        f'type={induced3!r} (subject matter is a theorem; the block is an exercise)',
    )


# --- 6. statement_extractor: the documented number hazard ---------------------------


async def probe_statement_extractor() -> None:
    print('\n=== statement_extractor ===')
    module = statement_extractor.Module()

    cross_ref = nodes(
        (
            'paragraph',
            '2.1.12 Prove Proposition 2.1.13 using the result of Theorem 2.1.9.',
        )
    )
    identity = await module.identity(cross_ref)
    check(
        'statement_extractor',
        "takes the block's OWN number, not an in-text cross-reference",
        identity.number == '2.1.12',
        f'number={identity.number!r} (expected 2.1.12), label={identity.label!r}',
    )


# --- 7. procedure_extractor: the verbatim partition contract ------------------------


async def probe_procedure_extractor() -> None:
    print('\n=== procedure_extractor ===')
    module = procedure_extractor.Module()
    contents = (
        '*Proof.* We induct on $n$. For $n=1$ the claim is immediate. '
        'Assume it holds for $n=k$. Then $\\sum_{i=1}^{k+1} i = \\frac{k(k+1)}{2} + (k+1)$. '
        'Simplifying gives $\\frac{(k+1)(k+2)}{2}$, which is the claim for $n=k+1$. $\\square$'
    )
    steps = await module.steps(contents)
    joined = ''.join(steps)

    def norm(text: str) -> str:
        """Whitespace-insensitive comparison key for the partition check."""
        return ''.join(text.split())

    exact = norm(joined) == norm(contents)
    check(
        'procedure_extractor',
        'steps are a verbatim partition of the content',
        exact,
        f'{len(steps)} step(s); '
        + (
            'exact'
            if exact
            else f'{len(norm(joined))} vs {len(norm(contents))} chars'
        ),
    )
    check(
        'procedure_extractor',
        'decomposes into more than one step',
        len(steps) > 1,
        f'{len(steps)} step(s)',
    )

    # A worked session plus a TRAILING remark: the closing commentary is the part that gets
    # silently dropped, which breaks the partition (seen live on Stein's SAGE Example 2.5.4).
    session = (
        'sage: R.<x> = PolynomialRing(Integers(13))\n'
        'sage: f = x^15 + 1\n'
        'sage: f.roots()\n'
        '[(12, 1), (10, 1), (4, 1)]\n\n'
        'The output of the roots command above lists each root along with its '
        'multiplicity (which is 1 in each case above).'
    )
    session_steps = await module.steps(session)
    kept = norm(''.join(session_steps)) == norm(session)
    check(
        'procedure_extractor',
        'keeps a trailing prose remark in the partition',
        kept,
        f'{len(session_steps)} step(s); '
        + (
            'exact'
            if kept
            else f'{len(norm("".join(session_steps)))} vs {len(norm(session))} chars'
        ),
    )


# --- 8. instruction_distributor: extent a range parser cannot get --------------------


async def probe_distributor() -> None:
    print('\n=== instruction_distributor ===')
    module = instruction_distributor.Module()
    following = [
        instruction_distributor.WindowProblem(
            position=0,
            number='12',
            text='Prove that $\\sqrt{2}$ is irrational.',
        ),
        instruction_distributor.WindowProblem(
            position=1,
            number='13',
            text='Prove that the sum of two odd integers is even.',
        ),
        instruction_distributor.WindowProblem(
            position=2, number='14', text='Compute $\\int_0^1 x^3 dx$.'
        ),
    ]
    instruction, governed = await module.govern(
        lead_in='Prove each of the following.', following=following
    )
    check(
        'instruction_distributor',
        'governs the two prove-problems and excludes the compute one',
        sorted(governed) == [0, 1],
        f'governed={sorted(governed)} (expected [0, 1]), instruction={instruction[:50]!r}',
    )


# --- 9. corrector: repair injected math errors against the real page image ----------

# Subtle, semantically wrong single-token edits — the error class the corrector exists for
# (see core.llm: "fixed subtle math errors, e.g. a misread root index").
INJECTIONS = [
    ('f^{-1}', 'f^{-2}'),
    ('\\bigcup', '\\bigcap'),
    ('\\emptyset', '\\infty'),
]


async def probe_corrector(pdf: str, out_dir: str) -> None:
    """OCR one page, corrupt its transcription, and check the correction pass repairs it."""
    print('\n=== corrector ===')
    from kms.ingestion import ocr

    segment = ocr.extract(pdf, output_dir=out_dir, pages=[1])[0]
    original = segment.content or ''
    corrupted, applied = original, []
    for good, bad in INJECTIONS:
        if good in corrupted:
            corrupted = corrupted.replace(good, bad, 1)
            applied.append((good, bad))
    if not applied:
        print('  [SKIP] no injection target on this page')
        return

    out = await corrector.Module().aforward(
        page_image=corrector._load_dspy_image(segment.image_path),
        transcription=corrupted,
    )
    check(
        'corrector',
        'stays within the tolerance guard (no runaway rewrite)',
        corrector._within_tolerance(corrupted, out),
        f'{len(corrupted)} chars in, {len(out)} out',
    )
    for good, bad in applied:
        check(
            'corrector',
            f'repairs the injected {bad!r} against the page image',
            bad not in out and good in out,
            'repaired' if bad not in out else f'{bad} SURVIVED',
        )


async def main() -> None:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    await probe_extractor()
    await probe_seam()
    await probe_splitter()
    await probe_instruction_finder()
    await probe_group_finder()
    await probe_role_typer()
    await probe_block_typer()
    await probe_statement_extractor()
    await probe_procedure_extractor()
    await probe_distributor()

    if (
        len(sys.argv) > 2
    ):  # optional: <book.pdf> <out_dir> adds the vision probe
        await probe_corrector(sys.argv[1], sys.argv[2])

    print('\n===== SUMMARY =====')
    passed = sum(1 for *_, ok, _ in RESULTS if ok)
    for module, name, ok, detail in RESULTS:
        if not ok:
            print(f'  FAIL  {module}: {name} — {detail}')
    print(f'{passed}/{len(RESULTS)} probes passed')
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == '__main__':
    asyncio.run(main())
