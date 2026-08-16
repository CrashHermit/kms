"""Structural block extraction: page markdown into typed AST nodes."""

import logging
import re

import dspy
from langgraph.types import Send
from pydantic import BaseModel, Field

from kms.core import logs, models, module, recording, state

logger = logging.getLogger(__name__)

_VALID_TYPES = frozenset(
    {
        'paragraph',
        'math',
        'code',
        'list',
        'table',
        'image',
        'caption',
        'header',
        'bibliographic',
        'note',
    }
)
_BLOCK_FURNITURE = 'furniture'


def _node_for(node_type: str, content: str | None) -> models.ASTNode:
    """Converts a validated block type into an AST node.

    Args:
        node_type: The block type as emitted by the model.
        content: The source content covered by the block.

    Returns:
        The AST node.

    Raises:
        ValueError: If the block type is not a known node type.
    """
    node_type = node_type.strip().lower()
    if node_type not in _VALID_TYPES:
        raise ValueError(f'Unknown block type: {node_type!r}')
    if node_type == 'image':
        content = _image_placeholder(content)
    return models.ASTNode(type=node_type, content=content)


def _image_placeholder(content: str | None) -> str | None:
    """Removes a source-list marker from a standalone image placeholder."""
    if not content:
        return content
    match = re.fullmatch(
        r'\s*(?:[*-]\s*)?(?:[A-Za-z]\)\s*)?'
        r'(!\[[^\]]+\]\(\))\s*',
        content,
    )
    return match.group(1) if match else content


class DSPyModel(BaseModel):
    """One reconstructed structural block used by the extractor worker."""

    type: str = Field(
        description=(
            'The block type: paragraph, math, code, list, table, image, '
            'caption, header, bibliographic, note, or furniture.'
        )
    )
    content: str | None = Field(
        default=None, description='The verbatim source content of the block.'
    )


class LineSpan(BaseModel):
    """One inclusive, 1-based source-line span and its structural type."""

    start: int = Field(description='The first 1-based source line.')
    end: int = Field(description='The last 1-based source line.')
    type: str = Field(
        description=(
            'The block type: paragraph, math, code, list, table, image, '
            'caption, header, bibliographic, note, or furniture.'
        )
    )


def _validate_spans(spans: list[LineSpan], source_lines: list[str]) -> None:
    """Checks that spans cover every nonblank source line exactly once."""
    if not spans:
        raise ValueError('Extractor returned no line spans')

    next_line = 1
    line_count = len(source_lines)
    for span in spans:
        if span.start < 1 or span.end < span.start:
            raise ValueError(f'Invalid line span: {span!r}')
        if span.end > line_count:
            raise ValueError(
                f'Line span {span!r} exceeds source line count {line_count}'
            )
        if span.start < next_line:
            raise ValueError(
                f'Line span {span!r} overlaps before source line {next_line}'
            )
        skipped = source_lines[next_line - 1 : span.start - 1]
        if any(line.strip() for line in skipped):
            raise ValueError(
                f'Line span {span!r} leaves nonblank source lines '
                f'before source line {span.start}'
            )
        if not any(
            line.strip() for line in source_lines[span.start - 1 : span.end]
        ):
            raise ValueError(f'Line span {span!r} contains only blank lines')
        next_line = span.end + 1

    trailing = source_lines[next_line - 1 : line_count]
    if any(line.strip() for line in trailing):
        raise ValueError(
            f'Line spans leave nonblank source lines after source line '
            f'{next_line - 1}'
        )


def _partition(
    blocks: list[DSPyModel],
) -> tuple[list[DSPyModel], list[DSPyModel]]:
    """Splits extracted blocks into kept nodes and discarded furniture."""
    kept: list[DSPyModel] = []
    discarded: list[DSPyModel] = []
    for block in blocks:
        target = (
            discarded
            if (block.type or '').strip().lower() == _BLOCK_FURNITURE
            else kept
        )
        target.append(block)
    return kept, discarded


class Signature(dspy.Signature):
    r"""
    Partition one textbook segment into ordered structural blocks.

    The input is the original markdown as a list of lines, each with a number.
    Return only inclusive line spans and block types. The caller copies all
    block text from the original source.

    COVERAGE IS MANDATORY

    - Read every numbered line from top to bottom before answering.
    - Every nonblank line belongs to exactly one span.
    - Do not create a span containing only blank lines.
    - Never skip ordinary prose, continuation text, notes, or uncertain text.
    - Spans are ordered, non-overlapping, in range, and source-preserving.
    - Blank separator lines may be omitted.
    - A span may contain several nonblank lines and internal blank lines when
      they belong to one structure.
    - Never copy, rewrite, summarize, repair, normalize, duplicate, or reorder
      source text.

    SCAN THEN CLASSIFY

    First mark every nonblank line as owned. Group owned lines into contiguous
    blocks. Then assign each block one type. Before answering, verify that no
    nonblank line was skipped or assigned twice.

    TYPES

    - `paragraph`: ordinary prose, definitions, proofs, examples, solutions,
      exercises, fragments, or uncertain content.
    - `caption`: a figure caption, table title, or asset label in its own
      block.
    - `header`: a Markdown heading or standalone section label.
    - `math`: standalone display mathematics.
    - `code`: a fenced or clearly delimited code block, including its internal
      blank lines.
    - `list`: a bullet list, numbered list, or consecutive numbered exercises.
    - `table`: a Markdown table, including its header and rows.
    - `image`: a standalone image placeholder such as `![1]()`, including a
      bullet or part marker immediately before it.
    - `note`: a marked footnote, endnote, or marginal note.
    - `bibliographic`: a clearly separate citation or reference.
    - `furniture`: repeated page apparatus such as folios, running heads,
      licence text, access URLs, or colophons.

    When uncertain between content and furniture, use `paragraph`. Keep
    fragments at segment boundaries as content. Return the ordered list of
    spans and nothing else. Return the ordered list of spans and nothing else.
    """

    lines: str = dspy.InputField(
        description=(
            'The original markdown as a list of source lines. DSPy displays '
            '1-based line numbers; return only span boundaries and types.'
        )
    )
    spans: list[LineSpan] = dspy.OutputField(
        description=(
            'The ordered line spans and structural types for the source '
            'blocks. Do not return block text.'
        )
    )


class Extractor(module.Module):
    """Splits one page of markdown into typed structural blocks."""

    signature = Signature
    record_name = 'extractor'

    def __init__(
        self,
        language_model: dspy.LM,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__(language_model, recorder)
        self.predictor.demos = [
            dspy.Example(
                lines=[
                    '**Exercise 2.4:** Match each equation to its slope field.',
                    '',
                    'a) ![1]()',
                    '',
                    'b) ![2]()',
                ],
                spans=[
                    LineSpan(start=1, end=1, type='paragraph'),
                    LineSpan(start=3, end=3, type='image'),
                    LineSpan(start=5, end=5, type='image'),
                ],
            ).with_inputs('lines'),
            dspy.Example(
                lines=[
                    '#### Definition 2.1.6 Subgraphs.',
                    '',
                    'We say that $G$ is a subgraph of $H$.',
                    '',
                    '$$G \\subseteq H.$$',
                ],
                spans=[
                    LineSpan(start=1, end=1, type='header'),
                    LineSpan(start=3, end=3, type='paragraph'),
                    LineSpan(start=5, end=5, type='math'),
                ],
            ).with_inputs('lines'),
            dspy.Example(
                lines=[
                    'First paragraph.',
                    '',
                    'Second paragraph.',
                    '',
                    'Third paragraph.',
                    '',
                    'Fourth paragraph.',
                    '',
                    '#### Lemma 2.1.8.',
                    '',
                    'The sum of the degrees is even.',
                ],
                spans=[
                    LineSpan(start=1, end=1, type='paragraph'),
                    LineSpan(start=3, end=3, type='paragraph'),
                    LineSpan(start=5, end=5, type='paragraph'),
                    LineSpan(start=7, end=7, type='paragraph'),
                    LineSpan(start=9, end=9, type='header'),
                    LineSpan(start=11, end=11, type='paragraph'),
                ],
            ).with_inputs('lines'),
            dspy.Example(
                lines=[
                    'This paragraph continues',
                    'across two source lines.',
                    '',
                    '$$x + y',
                    '= 4$$',
                ],
                spans=[
                    LineSpan(start=1, end=2, type='paragraph'),
                    LineSpan(start=4, end=5, type='math'),
                ],
            ).with_inputs('lines'),
        ]

    def encode(self, segment_markdown: str) -> dict:
        """Builds the numbered-line input for one page."""
        return {'lines': segment_markdown.split('\n')}

    def decode(self, prediction, **inputs) -> list[DSPyModel]:
        """Reconstructs verbatim blocks from the returned line spans."""
        source_lines = inputs['segment_markdown'].split('\n')
        spans = module.as_list(prediction.spans)
        _validate_spans(spans, source_lines)
        return [
            DSPyModel(
                type=span.type,
                content='\n'.join(source_lines[span.start - 1 : span.end]),
            )
            for span in spans
        ]


class ExtractorNode:
    """Langgraph node dispatching one extraction worker per segment."""

    def __init__(self, module: Extractor) -> None:
        self.module = module

    def dispatch(self, state: state.State) -> list[Send] | str:
        """Sends one worker per segment with content, else the collector."""
        segments = state.get('segments', [])
        sends = [
            Send('extractor_worker', {'segment': segment})
            for segment in segments
            if segment.content
        ]
        return sends or 'extractor_collect'

    async def worker(self, state: dict) -> dict:
        """Extracts and partitions one segment into AST nodes."""
        segment: models.Segment = state['segment']
        extracted = await self.module.aforward(segment_markdown=segment.content)
        kept, discarded = _partition(extracted)
        for block in discarded:
            logger.debug(
                'page %d: discarded %s block %r',
                segment.index,
                block.type,
                logs.elide(block.content),
            )
        if discarded:
            logger.info(
                'extractor: page %d dropped %d furniture block(s)',
                segment.index,
                len(discarded),
            )
        nodes = [_node_for(block.type, block.content) for block in kept]
        return {'extract_results': [(segment.index, nodes)]}

    def collect(self, state: state.State) -> dict:
        """Merges per-page extraction results back onto the segments."""
        results = state.get('extract_results', [])
        segments = models.merge_results_into_segments(
            state['segments'], results, 'nodes'
        )
        logger.info(
            'extractor: %d page(s) -> %d node(s)',
            len(results),
            sum(len(nodes) for _, nodes in results),
        )
        return {'segments': segments}
