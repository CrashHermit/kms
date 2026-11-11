"""Mistral OCR transport and KMS2 artifact materialization."""

import base64
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
import pypdfium2 as pdfium
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from kms.config.services import OCRSettings
from kms.ocr.pdfium import PDFIUM_LOCK
from kms.source.models.source import BlockType
from kms.source.processing.ocr_models import OCRArtifact, OCRPageArtifact

_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


class MistralOCRError(RuntimeError):
    """Raised when the Mistral OCR API cannot be used or parsed."""


class PageDimensions(BaseModel):
    """Required source-page dimensions returned by Mistral."""

    model_config = ConfigDict(extra='allow')

    width: float = Field(gt=0)
    height: float = Field(gt=0)


class OCRLocatedItem(BaseModel):
    """Mistral response item with a complete page-space bounding box."""

    model_config = ConfigDict(extra='allow')

    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return the item's page-space bounding box."""
        return (
            self.top_left_x,
            self.top_left_y,
            self.bottom_right_x,
            self.bottom_right_y,
        )


class OCRBlock(OCRLocatedItem):
    """One typed Mistral OCR block."""

    type: str
    content: str
    table_id: str | None = None
    image_id: str | None = None
    confidence_scores: dict[str, object] | None = None


class OCRImage(OCRLocatedItem):
    """One image returned by Mistral OCR."""

    id: str | None = None
    image_base64: str


class OCRPage(BaseModel):
    """One page returned by Mistral OCR."""

    model_config = ConfigDict(extra='allow')

    index: int
    markdown: str
    images: list[OCRImage] = Field(default_factory=list)
    tables: list[dict[str, object]] = Field(default_factory=list)
    hyperlinks: list[dict[str, object] | str] = Field(default_factory=list)
    header: str | None = None
    footer: str | None = None
    dimensions: PageDimensions
    confidence_scores: dict[str, object] | None = None
    blocks: list[OCRBlock] = Field(default_factory=list)


class OCRResponse(BaseModel):
    """The validated Mistral OCR response."""

    model_config = ConfigDict(extra='allow')

    pages: list[OCRPage] = Field(default_factory=list)
    model: str | None = None
    document_annotation: dict[str, object] | None = None
    usage_info: dict[str, object] = Field(default_factory=dict)
    _raw_response: dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_raw(cls, raw_response: dict[str, Any]) -> OCRResponse:
        """Validates and retains one raw Mistral response."""
        response = cls.model_validate(raw_response)
        response._raw_response = raw_response
        return response

    @property
    def raw_response(self) -> dict[str, Any]:
        """Return the original unmodified provider response."""
        return self._raw_response or self.model_dump()


class OCRRequestOptions(BaseModel):
    """Optional parameters for one Mistral OCR request."""

    include_image_base64: bool = True
    include_blocks: bool = True
    extract_header: bool = True
    extract_footer: bool = True
    confidence_scores_granularity: Literal['page', 'block', 'word'] | None = (
        None
    )
    pages: list[int] | None = None
    table_format: Literal['markdown', 'html'] | None = None


class OCRRequest(BaseModel):
    """One validated request sent to the Mistral OCR endpoint."""

    model: str
    document_url: str
    options: OCRRequestOptions = Field(default_factory=OCRRequestOptions)

    def payload(self) -> dict[str, Any]:
        """Build the Mistral API request payload."""
        return {
            'model': self.model,
            'document': {
                'type': 'document_url',
                'document_url': self.document_url,
            },
            **self.options.model_dump(exclude_none=True),
        }


def _require_key(settings: OCRSettings) -> str:
    """Return the configured Mistral API key or raise."""
    key = settings.api_key
    if not key:
        raise MistralOCRError(
            'KMS_OCR__API_KEY is not set. Export your Mistral API key '
            'before running the Mistral OCR client.'
        )
    return key


def _request_ocr(request: OCRRequest, settings: OCRSettings) -> OCRResponse:
    """Send one validated request to the Mistral OCR endpoint."""
    try:
        response = httpx.post(
            settings.url,
            json=request.payload(),
            headers={
                'Authorization': f'Bearer {_require_key(settings)}',
                'Content-Type': 'application/json',
            },
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

    try:
        return OCRResponse.from_raw(response.json())
    except (TypeError, ValueError) as exc:
        raise MistralOCRError(
            'Mistral OCR returned an invalid response'
        ) from exc


def ocr_pdf(
    pdf_bytes: bytes,
    settings: OCRSettings,
    pages: list[int] | None = None,
    options: OCRRequestOptions | None = None,
) -> OCRResponse:
    """Run Mistral OCR on PDF bytes and return the validated response."""
    request_options = options or OCRRequestOptions()
    if pages is not None:
        request_options = request_options.model_copy(update={'pages': pages})
    request = OCRRequest(
        model=settings.model,
        document_url=(
            'data:application/pdf;base64,'
            + base64.b64encode(pdf_bytes).decode('ascii')
        ),
        options=request_options,
    )
    return _request_ocr(request, settings)


def _render_page(
    pdf: Any, page_index: int, path: Path, scale: float, pdf_path: Path
) -> tuple[int, int]:
    """Render one source page and return its pixel dimensions."""
    try:
        with PDFIUM_LOCK:
            page = None
            bitmap = None
            try:
                page = pdf[page_index]
                bitmap = page.render(scale=scale)
                rendered = bitmap.to_pil().copy()
            finally:
                try:
                    if bitmap is not None:
                        bitmap.close()
                finally:
                    bitmap = None
                    try:
                        if page is not None:
                            page.close()
                    finally:
                        page = None
        size = rendered.size
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(path)
        return size
    except Exception as exc:
        raise MistralOCRError(
            f'Failed to render page {page_index} from PDF {pdf_path}'
        ) from exc


def _block_bbox(
    block: OCRBlock,
    page: OCRPage,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return a clamped block crop box in rendered-image pixels."""
    width, height = image_size
    source_width = page.dimensions.width
    source_height = page.dimensions.height
    left, top, right, bottom = block.bbox
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


def _block_type(value: str) -> BlockType:
    """Map Mistral structural labels to canonical block types."""
    if value in {'text', 'paragraph'}:
        return BlockType.PARAGRAPH
    if value == 'title':
        return BlockType.HEADER
    if value == 'references':
        return BlockType.BIBLIOGRAPHIC
    return BlockType(value)


def _materialize_page(
    page: OCRPage,
    document_dir: Path,
    pdf: Any,
    render_scale: float,
    block_crop_scale: float,
    pdf_path: Path,
    rendered_page: bytes | None = None,
) -> OCRPageArtifact:
    """Materialize rendered pages and canonical block crops."""
    page_path = document_dir / 'Document.png'
    if rendered_page is None:
        image_size = _render_page(
            pdf, page.index, page_path, render_scale, pdf_path
        )
    else:
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_bytes(rendered_page)
        with Image.open(page_path) as rendered:
            image_size = rendered.size
    artifacts: list[OCRArtifact] = []
    with Image.open(page_path) as opened:
        page_image = opened.convert('RGB')
        for block_index, block in enumerate(page.blocks):
            block_bbox = _block_bbox(block, page, image_size)
            path = document_dir / 'Blocks' / f'Block_{block_index:04d}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            crop = page_image.crop(block_bbox)
            if block_crop_scale != 1.0:
                crop = crop.resize(
                    (
                        max(1, round(crop.width * block_crop_scale)),
                        max(1, round(crop.height * block_crop_scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            crop.save(path)
            artifacts.append(
                OCRArtifact(
                    page_index=page.index,
                    block_index=block_index,
                    block_type=_block_type(block.type),
                    content=block.content,
                    crop_path=str(path),
                    crop_bbox=block_bbox,
                )
            )
    page_image.close()
    return OCRPageArtifact(
        page_index=page.index,
        page_image_path=str(page_path),
        markdown=page.markdown,
        blocks=artifacts,
    )


def materialize_ocr_artifacts(
    response: OCRResponse,
    pdf_path: str | Path,
    output_dir: str | Path,
    render_scale: float,
    block_crop_scale: float,
    *,
    rendered_pages: dict[int, bytes] | None = None,
) -> list[OCRPageArtifact]:
    """Materialize OCR response pages as provider-neutral artifacts."""
    source_path = Path(pdf_path)
    output_root = Path(output_dir)
    pdf = None
    if rendered_pages is None:
        with PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(str(source_path))
    try:
        pages: list[OCRPageArtifact] = []
        for page in response.pages:
            document_dir = (
                output_root / 'Documents' / f'Document_{page.index:04d}'
            )
            pages.append(
                _materialize_page(
                    page,
                    document_dir,
                    pdf,
                    render_scale,
                    block_crop_scale,
                    source_path,
                    rendered_page=(
                        rendered_pages.get(page.index)
                        if rendered_pages is not None
                        else None
                    ),
                )
            )
        return pages
    finally:
        if pdf is not None:
            with PDFIUM_LOCK:
                pdf.close()
                pdf = None


class MistralOCRProvider:
    """Adapt Mistral OCR responses to KMS2 correction artifacts."""

    def __init__(self, settings: OCRSettings) -> None:
        self._settings = settings

    def extract(
        self,
        pdf_path: str | Path,
        *,
        source_uuid: str,
        pages: list[int] | None = None,
    ) -> list[OCRPageArtifact]:
        """OCR a PDF and materialize its ordered page artifacts."""
        source_path = Path(pdf_path)
        response = ocr_pdf(
            source_path.read_bytes(),
            self._settings,
            pages=pages,
        )
        return materialize_ocr_artifacts(
            response,
            source_path,
            Path(self._settings.output_dir)
            / 'sources'
            / source_uuid
            / uuid4().hex,
            self._settings.render_scale,
            self._settings.block_crop_scale,
        )
