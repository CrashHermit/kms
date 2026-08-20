"""Mistral OCR front-end: source documents with pages and assets."""

import base64
import json
import re
from pathlib import Path
from typing import Any, Literal

import httpx
import pypdfium2 as pdfium
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)

from kms import config
from kms.core import models, state

_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


class MistralOCRError(RuntimeError):
    """Raised when the Mistral OCR API cannot be used or fails."""

    pass


class OCRBlock(BaseModel):
    """Represents one typed Mistral OCR block and its page coordinates."""

    model_config = ConfigDict(extra='allow')

    type: str
    content: str | None = None
    top_left_x: float | None = None
    top_left_y: float | None = None
    bottom_right_x: float | None = None
    bottom_right_y: float | None = None
    confidence: float | None = None
    confidence_scores: dict[str, Any] | list[Any] | None = None


class OCRImageAnnotation(BaseModel):
    """Describes an extracted image for correction routing."""

    model_config = ConfigDict(extra='allow')

    category: Literal[
        'diagram',
        'graph',
        'table',
        'photograph',
        'illustration',
        'scanned_text',
        'unknown',
    ] = 'unknown'
    description: str | None = None
    contains_text: bool | None = None
    contains_math: bool | None = None
    contains_code: bool | None = None


class OCRImage(BaseModel):
    """Represents one image returned by Mistral OCR."""

    model_config = ConfigDict(extra='allow')

    id: str | None = None
    image_base64: str | None = None
    top_left_x: float | None = None
    top_left_y: float | None = None
    bottom_right_x: float | None = None
    bottom_right_y: float | None = None
    image_annotation: OCRImageAnnotation | None = None

    @field_validator('image_annotation', mode='before')
    @classmethod
    def parse_image_annotation(cls, value: Any) -> Any:
        """Parses Mistral's JSON-string image annotations."""
        if isinstance(value, str):
            return json.loads(value)
        return value


class OCRTable(BaseModel):
    """Represents one table returned by Mistral OCR."""

    model_config = ConfigDict(extra='allow')

    id: str | None = None
    content: str | None = None


class OCRPage(BaseModel):
    """Represents one page in a Mistral OCR response."""

    model_config = ConfigDict(extra='allow')

    index: int | None = None
    markdown: str = ''
    dimensions: dict[str, Any] | None = None
    blocks: list[OCRBlock] = Field(default_factory=list)
    images: list[OCRImage] = Field(default_factory=list)
    tables: list[OCRTable] = Field(default_factory=list)
    header: str | None = None
    footer: str | None = None


class OCRReference(BaseModel):
    """Represents one document reference found by Mistral."""

    model_config = ConfigDict(extra='allow')

    kind: str
    label: str
    context: str | None = None
    target_description: str | None = None
    page_index: int | None = None


class OCRDocumentAnnotation(BaseModel):
    """Represents document-level metadata returned by Mistral."""

    model_config = ConfigDict(extra='allow')

    references: list[OCRReference] = Field(default_factory=list)


class OCRResponse(BaseModel):
    """Represents the validated Mistral OCR response."""

    model_config = ConfigDict(extra='allow')

    pages: list[OCRPage] = Field(default_factory=list)
    document_annotation: OCRDocumentAnnotation | None = None
    _raw_response: dict[str, Any] = PrivateAttr(default_factory=dict)

    @field_validator('document_annotation', mode='before')
    @classmethod
    def parse_document_annotation(cls, value: Any) -> Any:
        """Parses Mistral's JSON-string document annotation."""
        if isinstance(value, str):
            return json.loads(value)
        return value

    @classmethod
    def from_raw(cls, raw_response: dict[str, Any]) -> 'OCRResponse':
        """Validates and retains one raw Mistral response."""
        response = cls.model_validate(raw_response)
        response._raw_response = raw_response
        return response

    @property
    def raw_response(self) -> dict[str, Any]:
        """Returns the original response body for diagnostics and replay."""
        return self._raw_response or self.model_dump()


REFERENCE_ANNOTATION_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'document_references',
        'schema': {
            'type': 'object',
            'properties': {
                'references': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'kind': {
                                'type': 'string',
                                'enum': [
                                    'figure',
                                    'equation',
                                    'table',
                                    'section',
                                    'other',
                                ],
                            },
                            'label': {'type': 'string'},
                            'context': {'type': 'string'},
                            'target_description': {'type': 'string'},
                            'page_index': {'type': 'integer'},
                        },
                        'required': ['kind', 'label', 'context'],
                        'additionalProperties': False,
                    },
                }
            },
            'required': ['references'],
            'additionalProperties': False,
        },
    },
}

REFERENCE_ANNOTATION_PROMPT = (
    'Scan the entire document for explicit references to figures, equations, '
    'tables, sections, theorems, definitions, examples, or other numbered '
    'items. Return each reference exactly as printed, with its surrounding '
    'sentence and a short description of the referenced target when visible. '
    'Do not infer references that are not explicitly present.'
)


BBOX_ANNOTATION_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'image_metadata',
        'schema': {
            'type': 'object',
            'properties': {
                'category': {
                    'type': 'string',
                    'enum': [
                        'diagram',
                        'graph',
                        'table',
                        'photograph',
                        'illustration',
                        'scanned_text',
                        'unknown',
                    ],
                },
                'description': {'type': 'string'},
                'contains_text': {'type': 'boolean'},
                'contains_math': {'type': 'boolean'},
                'contains_code': {'type': 'boolean'},
            },
            'required': [
                'category',
                'description',
                'contains_text',
                'contains_math',
                'contains_code',
            ],
            'additionalProperties': False,
        },
    },
}


class OCRRequestOptions(BaseModel):
    """Represents optional Mistral OCR request parameters."""

    include_image_base64: bool = True
    include_blocks: bool = True
    extract_header: bool = True
    extract_footer: bool = True
    pages: list[int] | None = None
    table_format: Literal['markdown', 'html'] | None = None
    document_annotation_format: dict | None = None
    document_annotation_prompt: str | None = None
    bbox_annotation_format: dict | None = None

    @classmethod
    def with_references(cls) -> 'OCRRequestOptions':
        """Returns options enabling document reference extraction."""
        return cls(
            document_annotation_format=REFERENCE_ANNOTATION_FORMAT,
            document_annotation_prompt=REFERENCE_ANNOTATION_PROMPT,
        )

    @classmethod
    def with_image_metadata(cls) -> 'OCRRequestOptions':
        """Returns options enabling image classification metadata."""
        return cls(bbox_annotation_format=BBOX_ANNOTATION_FORMAT)


class OCRRequest(BaseModel):
    """Represents a validated request sent to Mistral OCR."""

    model: str
    document_url: str
    options: OCRRequestOptions = Field(default_factory=OCRRequestOptions)

    def payload(self) -> dict[str, Any]:
        """Builds the Mistral API request payload."""
        return {
            'model': self.model,
            'document': {
                'type': 'document_url',
                'document_url': self.document_url,
            },
            **self.options.model_dump(exclude_none=True),
        }


_MISTRAL_NODE_TYPES = {
    'text': models.NodeType.PARAGRAPH,
    'paragraph': models.NodeType.PARAGRAPH,
    'heading': models.NodeType.HEADER,
    'header': models.NodeType.HEADER,
    'title': models.NodeType.HEADER,
    'footer': models.NodeType.FOOTER,
    'equation': models.NodeType.MATH,
    'math': models.NodeType.MATH,
    'code': models.NodeType.CODE,
    'list': models.NodeType.LIST,
    'table': models.NodeType.TABLE,
    'image': models.NodeType.IMAGE,
    'caption': models.NodeType.CAPTION,
    # Mistral may emit a page-level references block when document
    # annotations are enabled. Preserve it as bibliographic source content;
    # it is distinct from OCRResponse.document_annotation.references.
    'reference': models.NodeType.BIBLIOGRAPHIC,
    'references': models.NodeType.BIBLIOGRAPHIC,
}


def _canonical_node_type(provider_type: str) -> models.NodeType:
    """Maps one Mistral block type to a canonical KMS node type."""
    normalized_type = provider_type.strip().lower()
    try:
        return _MISTRAL_NODE_TYPES[normalized_type]
    except KeyError as exc:
        raise ValueError(
            f'Unknown Mistral OCR block type: {provider_type!r}'
        ) from exc


class OCRBlockRegion(BaseModel):
    """Represents one Mistral block prepared for visual correction."""

    block_index: int
    block: OCRBlock
    crop_path: str | None = None
    crop_bbox: tuple[int, int, int, int] | None = None

    @property
    def canonical_type(self) -> models.NodeType:
        """Returns the canonical type for this provider block."""
        return _canonical_node_type(self.block.type)


class OCRPageArtifact(BaseModel):
    """Represents one materialized OCR page and its correction inputs."""

    index: int
    markdown: str
    image_path: str
    footer: str | None = None
    pictures: list[models.Picture] = Field(default_factory=list)
    blocks: list[OCRBlockRegion] = Field(default_factory=list)

    def to_document(self) -> models.Document:
        """Converts this provider page into the canonical document model."""
        nodes: list[models.Node] = []
        picture_cursor = 0
        for region in self.blocks:
            block = region.block
            node = models.Node(
                type=region.canonical_type,
                content=block.content or '',
                index=region.block_index,
                provenance={
                    'provider': 'mistral',
                    'provider_type': block.type,
                    'provider_index': region.block_index,
                    'bbox': (
                        block.top_left_x,
                        block.top_left_y,
                        block.bottom_right_x,
                        block.bottom_right_y,
                    ),
                    'confidence': block.confidence,
                },
            )
            if region.crop_bbox is not None:
                node.provenance['crop_bbox'] = region.crop_bbox
            if region.crop_path is not None:
                node.provenance['crop_path'] = region.crop_path
            if block.type == 'image' and picture_cursor < len(self.pictures):
                node.image_path = self.pictures[picture_cursor].image_path
                picture_cursor += 1
            nodes.append(node)
        if not nodes and self.markdown.strip():
            nodes.append(
                models.Node(
                    type=models.NodeType.MARKDOWN,
                    content=self.markdown,
                    index=0,
                    provenance={
                        'provider': 'mistral',
                        'provider_type': 'markdown',
                    },
                )
            )
        footer = (self.footer or '').strip()
        if footer:
            nodes.append(
                models.Node(
                    type=models.NodeType.FOOTER,
                    content=footer,
                    index=len(nodes),
                    provenance={
                        'provider': 'mistral',
                        'provider_type': 'footer',
                    },
                )
            )
        return models.Document(
            index=self.index,
            image_path=self.image_path,
            content=(
                self.markdown
                if not footer
                else f'{self.markdown.rstrip()}\n\n{footer}'
            ),
            pictures=self.pictures,
            nodes=nodes,
            metadata={
                'provider': 'mistral',
                'markdown': self.markdown,
                'footer': footer or None,
            },
        )


def _require_key() -> str:
    """Returns the configured Mistral API key.

    Raises:
        MistralOCRError: If no key is configured.
    """
    key = config.get_settings().ocr.api_key
    if not key:
        raise MistralOCRError(
            'KMS_OCR__API_KEY is not set. Export your Mistral API key '
            '(e.g. `export KMS_OCR__API_KEY=...`) before running the '
            'Mistral front-end.'
        )
    return key


def _request_ocr(request: OCRRequest) -> OCRResponse:
    """Sends one validated request to the Mistral OCR endpoint."""
    payload = request.payload()
    headers = {
        'Authorization': f'Bearer {_require_key()}',
        'Content-Type': 'application/json',
    }
    try:
        response = httpx.post(
            config.get_settings().ocr.url,
            json=payload,
            headers=headers,
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500]
        raise MistralOCRError(
            f'Mistral OCR returned HTTP {exc.response.status_code}: {body}'
        ) from exc
    except httpx.HTTPError as exc:
        raise MistralOCRError(f'Mistral OCR request failed: {exc}') from exc
    return OCRResponse.from_raw(response.json())


def ocr_pdf(
    pdf_bytes: bytes,
    pages: list[int] | None = None,
    include_blocks: bool | None = None,
    options: OCRRequestOptions | None = None,
) -> OCRResponse:
    """Runs Mistral OCR on a PDF and returns its JSON response.

    Args:
        pdf_bytes: The PDF file bytes.
        pages: Optional 0-based page indexes to OCR.
        include_blocks: Optional compatibility override for ``options``.
        options: Additional typed Mistral OCR request parameters.

    Returns:
        The parsed OCR response body.

    Raises:
        MistralOCRError: If the request fails or the API errors.
    """
    data_url = 'data:application/pdf;base64,' + base64.b64encode(
        pdf_bytes
    ).decode('ascii')
    request_options = options or OCRRequestOptions()
    updates = {}
    if pages is not None:
        updates['pages'] = pages
    if include_blocks is not None:
        updates['include_blocks'] = include_blocks
    if updates:
        request_options = request_options.model_copy(update=updates)
    request = OCRRequest(
        model=config.get_settings().ocr.model,
        document_url=data_url,
        options=request_options,
    )
    return _request_ocr(request)


_IMG_REF = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')


def _write_image(data: str, path: Path) -> None:
    """Decodes base64 image data into a file, tolerating bad data."""
    if not data:
        return
    if data.startswith('data:'):
        data = data.split(',', 1)[-1]
    try:
        path.write_bytes(base64.b64decode(data))
    except (ValueError, OSError):
        pass


def _rewrite_page(
    markdown: str, images: list[OCRImage], segment_dir: Path
) -> tuple[str, list[models.Picture]]:
    """Rewrites image placeholders to local indexes and saves pictures.

    Args:
        markdown: The page's OCR markdown.
        images: The page's image records from the OCR response.
        segment_dir: The segment's output directory.

    Returns:
        ``(markdown, pictures)`` with placeholders rewritten to
        ``![N]()`` and pictures saved under ``Images/``.
    """
    images_by_id = {image.id: image for image in images if image.id}
    pictures_dir = segment_dir / 'Images'
    pictures_dir.mkdir(parents=True, exist_ok=True)
    order: list[str] = []

    def index_of(image_id: str) -> int:
        """Returns the 1-based display index, assigning it on first use."""
        if image_id not in order:
            order.append(image_id)
        return order.index(image_id) + 1

    def replace(match: re.Match) -> str:
        """Rewrites one placeholder to its local ``![N]()`` form."""
        target = match.group(1)
        if target not in images_by_id:
            return match.group(0)
        return f'![{index_of(target)}]()'

    rewritten = _IMG_REF.sub(replace, markdown)
    for image_id in images_by_id:
        index_of(image_id)

    pictures: list[models.Picture] = []
    for position, image_id in enumerate(order, start=1):
        path = pictures_dir / f'Image_{position - 1:03d}.png'
        _write_image(images_by_id[image_id].image_base64 or '', path)
        pictures.append(models.Picture(index=position, image_path=str(path)))
    return rewritten, pictures


def _with_footer(markdown: str, footer: str | None) -> str:
    """Appends the extracted page footer to the markdown, if any."""
    footer = (footer or '').strip()
    if not footer:
        return markdown
    return f'{markdown.rstrip()}\n\n{footer}'


def _block_bbox(
    block: OCRBlock,
    page: OCRPage,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    values = (
        block.top_left_x,
        block.top_left_y,
        block.bottom_right_x,
        block.bottom_right_y,
    )
    if any(value is None for value in values):
        return None
    width, height = image_size
    dimensions = page.dimensions or {}
    source_width = dimensions.get('width')
    source_height = dimensions.get('height')
    if not source_width or not source_height:
        source_width, source_height = width, height
    left, top, right, bottom = values
    scale_x = width / source_width
    scale_y = height / source_height
    margin = max(8, round(max(width, height) * 0.006))
    left = round(left * scale_x) - margin
    top = round(top * scale_y) - margin
    right = round(right * scale_x) + margin
    bottom = round(bottom * scale_y) + margin
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def _materialize_block_crops(
    artifacts: list[OCRPageArtifact], response: OCRResponse
) -> None:
    for page_artifact, page in zip(artifacts, response.pages, strict=True):
        image_path = Path(page_artifact.image_path)
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            for region in page_artifact.blocks:
                bbox = _block_bbox(region.block, page, image.size)
                if bbox is None:
                    continue
                crop_path = (
                    image_path.parent
                    / 'Blocks'
                    / f'Block_{region.block_index:04d}.png'
                )
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop = image.crop(bbox)
                scale = config.get_settings().ocr.block_crop_scale
                if scale != 1.0:
                    crop = crop.resize(
                        (
                            round(crop.width * scale),
                            round(crop.height * scale),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                crop.save(crop_path)
                region.crop_bbox = bbox
                region.crop_path = str(crop_path)


def materialize_document(
    response: OCRResponse,
    output_dir: str | Path,
    pdf_path: str | Path | None = None,
    pages: list[int] | None = None,
    render_pages: bool = True,
) -> models.Source:
    """Materializes validated OCR pages and canonical document nodes.

    Args:
        response: The validated Mistral OCR response.
        output_dir: The root directory for materialized artifacts.
        pdf_path: Optional source PDF used to render page images.
        pages: Optional source PDF page indexes corresponding to response pages.
        render_pages: Whether to render page images when ``pdf_path`` is given.

    Returns:
        The OCR document with materialized page artifacts.
    """
    output_dir = Path(output_dir)
    artifacts: list[OCRPageArtifact] = []
    for order_index, page in enumerate(response.pages):
        document_dir = output_dir / 'Documents' / f'Document_{order_index:04d}'
        markdown, pictures = _rewrite_page(
            page.markdown,
            page.images,
            document_dir,
        )
        artifacts.append(
            OCRPageArtifact(
                index=order_index,
                image_path=str(document_dir / 'Document.png'),
                pictures=pictures,
                markdown=markdown,
                footer=page.footer,
                blocks=[
                    OCRBlockRegion(block_index=index, block=block)
                    for index, block in enumerate(page.blocks)
                ],
            )
        )
    raw_response_path = output_dir / 'ocr_response.json'
    raw_response_path.parent.mkdir(parents=True, exist_ok=True)
    raw_response_path.write_text(
        json.dumps(response.raw_response, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    if render_pages and pdf_path is not None:
        _render_page_images(
            pdf_path,
            [artifact.to_document() for artifact in artifacts],
            pages,
        )
    _materialize_block_crops(artifacts, response)
    source = models.Source(
        ocr_response=response,
        metadata={
            'provider': 'mistral',
            'raw_response_path': str(raw_response_path),
        },
        documents=[artifact.to_document() for artifact in artifacts],
    )
    for document, artifact in zip(source.documents, artifacts, strict=True):
        for node, region in zip(document.nodes, artifact.blocks, strict=False):
            node.provenance['crop_bbox'] = region.crop_bbox
            node.provenance['crop_path'] = region.crop_path
    return source


def build_source(
    response: OCRResponse, output_dir: str | Path
) -> models.Source:
    """Builds canonical source documents from a validated OCR response."""
    return materialize_document(response, output_dir)


def _render_page_images(
    pdf_path: str | Path,
    documents: list[models.Document],
    pages: list[int] | None,
) -> None:
    """Renders each segment's page to PNG at its configured image path."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i, document in enumerate(documents):
            source_page = pages[i] if pages is not None else i
            image = (
                pdf[source_page]
                .render(scale=config.get_settings().ocr.render_scale)
                .to_pil()
            )
            Path(document.image_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(document.image_path)
    finally:
        pdf.close()


def extract(
    pdf_path: str | Path,
    output_dir: str | Path = 'output',
    pages: list[int] | None = None,
    render_pages: bool = True,
) -> models.Source:
    """OCR's a PDF into canonical source documents with ordered nodes.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Root directory for segment output.
        pages: Optional 0-based page indexes to OCR.
        render_pages: When True, render each page to PNG.

    Returns:
        The canonical source containing one document per OCR page.
    """
    pdf_bytes = Path(pdf_path).read_bytes()
    response = ocr_pdf(pdf_bytes, pages=pages)
    return materialize_document(
        response,
        output_dir=output_dir,
        pdf_path=pdf_path,
        pages=pages,
        render_pages=render_pages,
    )


class OCRNode:
    def run(self, current_state: state.State) -> dict:
        source = extract(
            current_state['pdf_path'],
            output_dir=current_state['output_dir'],
            pages=current_state.get('pages'),
        )
        if isinstance(source, models.Source):
            source.key = current_state['source_key']
            source.metadata.update(current_state.get('source_metadata', {}))
        else:
            raise TypeError('OCR extract did not return a canonical Source')
        return {
            'source': source,
            'documents': source.documents,
        }
