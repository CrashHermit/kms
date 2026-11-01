"""Explicit runtime demonstrations for adaptive external research."""

from io import BytesIO

import dspy
from PIL import Image

from kms2.core.model.research.inference import (
    FetchWebAction,
    FinishResearchAction,
    InspectedResearchEvidence,
    ListEvidenceAction,
    OpenPdfAction,
    ReadEvidenceAction,
    ReadPdfPagesAction,
    ResearchAction,
    ResearchContext,
    ResearchDocumentDescriptor,
    ResearchEvidenceDescriptor,
    ResearchFailure,
    ResearchLocalFinding,
    ResearchTask,
    ResearchToolObservation,
    SearchPdfAction,
    WebSearchAction,
)


def _demo_image(color: tuple[int, int, int]) -> dspy.Image:
    """Build a valid tiny in-memory visual for a runtime demonstration."""
    output = BytesIO()
    Image.new('RGB', (1, 1), color).save(output, format='PNG')
    return dspy.Image(output.getvalue())


def _context(
    *,
    question: str,
    urls: list[str] | None = None,
    documents: list[ResearchDocumentDescriptor] | None = None,
    evidence: list[ResearchEvidenceDescriptor] | None = None,
    observations: list[ResearchToolObservation] | None = None,
    inspected: list[InspectedResearchEvidence] | None = None,
    local_findings: list[ResearchLocalFinding] | None = None,
    tools_remaining: int = 8,
) -> ResearchContext:
    """Construct a fresh bounded demonstration context."""
    return ResearchContext(
        task=ResearchTask(
            question=question,
            context='',
            urls=urls or [],
            local_findings=local_findings or [],
        ),
        documents=documents or [],
        evidence=evidence or [],
        observations=observations or [],
        inspected_evidence=inspected or [],
        tools_remaining=tools_remaining,
    )


def _example(
    context: ResearchContext,
    action: ResearchAction,
    *,
    images: list[dspy.Image] | None = None,
) -> dspy.Example:
    """Build one runtime demonstration with explicit visual input bytes."""
    return dspy.Example(
        context=context,
        evidence_images=images or [],
        action=action,
    ).with_inputs('context', 'evidence_images')


def _web_document() -> ResearchDocumentDescriptor:
    return ResearchDocumentDescriptor(
        position=0,
        url='https://example.edu/linear-algebra',
        title='Linear Algebra Notes',
        media_type='text/markdown',
        page_count=None,
    )


def _pdf_document() -> ResearchDocumentDescriptor:
    return ResearchDocumentDescriptor(
        position=0,
        url='https://example.edu/notes.pdf',
        title='Course Notes',
        media_type='application/pdf',
        page_count=12,
    )


def researcher_examples() -> list[dspy.Example]:
    """Return explicit examples spanning text, images, PDFs, and gaps."""
    question = 'Explain what an eigenvector is.'
    base = _context(question=question)
    searched = _context(
        question=question,
        observations=[
            ResearchToolObservation(
                kind='web_search',
                summary=(
                    'Discovery only: a university linear algebra reference '
                    'matches the topic.'
                ),
            )
        ],
        local_findings=[
            ResearchLocalFinding(
                kind='block',
                text='A nonzero vector v satisfies Av = λv.',
                inspected=False,
            )
        ],
    )
    fetched = _context(
        question=question,
        documents=[_web_document()],
        evidence=[
            ResearchEvidenceDescriptor(
                position=0,
                document_position=0,
                kind='web_passage',
                title='Linear Algebra Notes',
                url='https://example.edu/linear-algebra',
                page_index=None,
                line_start=12,
                line_end=18,
                token_count=24,
                heading_path=['Eigenvectors'],
                preview='A nonzero vector v is an eigenvector when Av = λv.',
            )
        ],
        observations=[
            ResearchToolObservation(
                kind='fetch_web',
                summary='Extracted a page and registered passage descriptors.',
                document_positions=[0],
                evidence_positions=[0],
            )
        ],
    )
    text_read = _context(
        question=question,
        documents=[_web_document()],
        observations=[
            ResearchToolObservation(
                kind='read_evidence',
                summary=(
                    'Complete passage supplied for inspection: A nonzero vector '
                    'v is an eigenvector when Av = λv.'
                ),
                evidence_positions=[0],
            )
        ],
        inspected=[
            InspectedResearchEvidence(
                position=0,
                kind='web_passage',
                url='https://example.edu/linear-algebra',
                title='Linear Algebra Notes',
                retrieved_at='2026-10-31T00:00:00Z',
                page_index=None,
                line_start=12,
                line_end=18,
                raw_content='A nonzero vector v is an eigenvector when Av = λv.',
                content='A nonzero vector v is an eigenvector when Av = λv.',
                native_image_position=None,
            )
        ],
        local_findings=[
            ResearchLocalFinding(
                kind='block',
                text='A nonzero vector v is an eigenvector when Av = λv.',
                inspected=True,
            )
        ],
    )
    image_discovered = _context(
        question='What does the linked diagram show?',
        documents=[_web_document()],
        evidence=[
            ResearchEvidenceDescriptor(
                position=1,
                document_position=0,
                kind='image',
                title='Linear Algebra Notes',
                url='https://example.edu/eigenvector-diagram.png',
                page_index=None,
                line_start=None,
                line_end=None,
                token_count=0,
                preview='https://example.edu/eigenvector-diagram.png',
            )
        ],
        observations=[
            ResearchToolObservation(
                kind='fetch_web',
                summary='Image URL discovered, not visually inspected.',
                evidence_positions=[1],
            )
        ],
    )
    image_read = _context(
        question='What does the linked diagram show?',
        documents=[_web_document()],
        observations=[
            ResearchToolObservation(
                kind='read_evidence',
                summary='Downloaded original image and included its PNG bytes.',
                evidence_positions=[1],
            )
        ],
        inspected=[
            InspectedResearchEvidence(
                position=1,
                kind='image',
                url='https://example.edu/eigenvector-diagram.png',
                title='Linear Algebra Notes',
                retrieved_at='2026-10-31T00:00:00Z',
                page_index=None,
                line_start=None,
                line_end=None,
                raw_content=None,
                content=None,
                native_image_position=0,
            )
        ],
    )
    pdf = _pdf_document()
    pdf_context = _context(
        question='Find the projection example in the PDF.',
        documents=[pdf],
    )
    pdf_opened = _context(
        question='Find the projection example in the PDF.',
        documents=[pdf],
        observations=[
            ResearchToolObservation(
                kind='open_pdf',
                summary='Opened 12-page PDF metadata; no page was inspected.',
                document_positions=[0],
            )
        ],
    )
    pdf_located = _context(
        question='Find the projection example in the PDF.',
        documents=[pdf],
        observations=[
            ResearchToolObservation(
                kind='search_pdf',
                summary=(
                    'Native text locator found a projection match on zero-based '
                    'page index 3; locator text is not inspected evidence.'
                ),
                document_positions=[0],
            )
        ],
    )
    pdf_read = _context(
        question='Find the projection example in the PDF.',
        documents=[pdf],
        observations=[
            ResearchToolObservation(
                kind='read_pdf_pages',
                summary='OCRed and rendered selected page index 3.',
                document_positions=[0],
                evidence_positions=[0],
            ),
            ResearchToolObservation(
                kind='read_evidence',
                summary='OCR text and actual selected page image were included.',
                evidence_positions=[0],
            ),
        ],
        inspected=[
            InspectedResearchEvidence(
                position=0,
                kind='pdf_page',
                url='https://example.edu/notes.pdf',
                title='Course Notes',
                retrieved_at='2026-10-31T00:00:00Z',
                page_index=3,
                line_start=0,
                line_end=3,
                raw_content='Projection onto the eigenspace leaves its vectors fixed.',
                content='Projection onto the eigenspace leaves its vectors fixed.',
                native_image_position=0,
            )
        ],
    )
    scanned_pdf_context = _context(
        question='Inspect the scanned chapter pages around the diagram.',
        documents=[pdf],
        observations=[
            ResearchToolObservation(
                kind='open_pdf',
                summary='Opened a scanned 12-page PDF; native text is unavailable.',
                document_positions=[0],
            )
        ],
        tools_remaining=2,
    )
    partial_failure = _context(
        question='Explain the claim from the requested page.',
        observations=[
            ResearchToolObservation(
                kind='fetch_web',
                summary='One requested URL failed; another source remains available.',
                failures=[
                    ResearchFailure(
                        operation='fetch_web',
                        url='https://example.edu/unavailable',
                        message='The page could not be extracted.',
                    )
                ],
            )
        ],
    )
    empty = _context(
        question='Find current evidence about an unavailable topic.',
        observations=[
            ResearchToolObservation(
                kind='web_search',
                summary='No search results were returned.',
            )
        ],
        tools_remaining=0,
    )

    return [
        _example(
            base,
            WebSearchAction(
                kind='web_search', query='eigenvector definition university'
            ),
        ),
        _example(
            _context(
                question='Explain this assigned page.',
                urls=['https://example.edu/linear-algebra'],
            ),
            FetchWebAction(
                kind='fetch_web', urls=['https://example.edu/linear-algebra']
            ),
        ),
        _example(
            searched,
            FetchWebAction(
                kind='fetch_web', urls=['https://example.edu/linear-algebra']
            ),
        ),
        _example(
            fetched,
            ListEvidenceAction(
                kind='list_evidence', document_position=0, start_position=0
            ),
        ),
        _example(
            fetched,
            ReadEvidenceAction(kind='read_evidence', evidence_positions=[0]),
        ),
        _example(
            text_read,
            FinishResearchAction(
                kind='finish',
                brief=(
                    'A nonzero vector v is an eigenvector when Av = λv '
                    '(Linear Algebra Notes).'
                ),
                evidence_positions=[0],
            ),
        ),
        _example(
            image_discovered,
            ReadEvidenceAction(kind='read_evidence', evidence_positions=[1]),
        ),
        _example(
            image_read,
            FinishResearchAction(
                kind='finish',
                brief='The image shows the vector direction preserved by the transformation.',
                evidence_positions=[1],
            ),
            images=[_demo_image((200, 30, 30))],
        ),
        _example(
            pdf_context,
            OpenPdfAction(kind='open_pdf', url='https://example.edu/notes.pdf'),
        ),
        _example(
            pdf_opened,
            SearchPdfAction(
                kind='search_pdf', document_position=0, query='projection'
            ),
        ),
        _example(
            pdf_located,
            ReadPdfPagesAction(
                kind='read_pdf_pages', document_position=0, page_indexes=[3]
            ),
        ),
        _example(
            pdf_read,
            FinishResearchAction(
                kind='finish',
                brief=(
                    'The selected PDF page shows projection onto the fixed '
                    'eigenspace (Course Notes, p. 4).'
                ),
                evidence_positions=[0],
            ),
            images=[_demo_image((20, 30, 200))],
        ),
        _example(
            scanned_pdf_context,
            ReadPdfPagesAction(
                kind='read_pdf_pages', document_position=0, page_indexes=[5, 6]
            ),
        ),
        _example(
            partial_failure,
            FetchWebAction(
                kind='fetch_web', urls=['https://example.edu/another-source']
            ),
        ),
        _example(
            empty,
            FinishResearchAction(
                kind='finish',
                brief='No source was available to verify this claim.',
                evidence_positions=[],
            ),
        ),
    ]


__all__ = ['researcher_examples']
