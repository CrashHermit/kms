"""Mistral OCR transport and KMS2 artifact materialization."""

import base64
import binascii
from pathlib import Path
from typing import Any, Literal

import httpx
import pypdfium2 as pdfium
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from kms2 import config
from kms2.core.model import OCRArtifact, OCRImageArtifact

_TIMEOUT = httpx.Timeout(300.0, connect=30.0)


class MistralOCRError(RuntimeError):
    """Raised when the Mistral OCR API cannot be used or parsed."""


class OCRBlock(BaseModel):
    """One typed Mistral OCR block and its optional page coordinates."""

    model_config = ConfigDict(extra='allow')

    type: str
    content: str | None = None
    top_left_x: float | None = None
    top_left_y: float | None = None
    bottom_right_x: float | None = None
    bottom_right_y: float | None = None
    confidence: float | None = None


class OCRImage(BaseModel):
    """One image returned by Mistral OCR."""

    model_config = ConfigDict(extra='allow')

    id: str | None = None
    image_base64: str | None = None
    top_left_x: float | None = None
    top_left_y: float | None = None
    bottom_right_x: float | None = None
    bottom_right_y: float | None = None


class OCRPage(BaseModel):
    """One page returned by Mistral OCR."""

    model_config = ConfigDict(extra='allow')

    index: int | None = None
    markdown: str = ''
    dimensions: dict[str, Any] | None = None
    blocks: list[OCRBlock] = Field(default_factory=list)
    images: list[OCRImage] = Field(default_factory=list)
    footer: str | None = None


class OCRResponse(BaseModel):
    """The validated Mistral OCR response."""

    model_config = ConfigDict(extra='allow')

    pages: list[OCRPage] = Field(default_factory=list)
    _raw_response: dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_raw(cls, raw_response: dict[str, Any]) -> 'OCRResponse':
        """Validates and retains one raw Mistral response."""
        response = cls.model_validate(raw_response)
        response._raw_response = raw_response
        return response

    @property
    def raw_response(self) -> dict[str, Any]:
        """Returns the original response body for replay and diagnostics."""
        return self._raw_response or self.model_dump()


class OCRRequestOptions(BaseModel):
    """Optional parameters for one Mistral OCR request."""

    include_image_base64: bool = True
    include_blocks: bool = True
    extract_header: bool = True
    extract_footer: bool = True
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


def _require_key() -> str:
    """Return the configured Mistral API key or raise."""
    key = config.get_settings().mistral_ocr.api_key
    if not key:
        raise MistralOCRError(
            'KMS2_MISTRAL_OCR__API_KEY is not set. Export your Mistral API key '
            'before running the Mistral OCR client.'
        )
    return key


def _request_ocr(request: OCRRequest) -> OCRResponse:
    """Send one validated request to the Mistral OCR endpoint."""
    try:
        response = httpx.post(
            config.get_settings().mistral_ocr.url,
            json=request.payload(),
            headers={
                'Authorization': f'Bearer {_require_key()}',
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
    pages: list[int] | None = None,
    options: OCRRequestOptions | None = None,
) -> OCRResponse:
    """Run Mistral OCR on PDF bytes and return the validated response."""
    request_options = options or OCRRequestOptions()
    if pages is not None:
        request_options = request_options.model_copy(update={'pages': pages})
    request = OCRRequest(
        model=config.get_settings().mistral_ocr.model,
        document_url=(
            'data:application/pdf;base64,'
            + base64.b64encode(pdf_bytes).decode('ascii')
        ),
        options=request_options,
    )
    return _request_ocr(request)


def _coordinates(
    value: OCRBlock | OCRImage,
) -> tuple[float, float, float, float] | None:
    """Return one complete Mistral page-space bounding box."""
    values = (
        value.top_left_x,
        value.top_left_y,
        value.bottom_right_x,
        value.bottom_right_y,
    )
    if any(item is None for item in values):
        return None
    return values  # type: ignore[return-value]


def _page_dimensions(page: OCRPage) -> tuple[float, float] | None:
    """Return positive source-page dimensions when Mistral supplied them."""
    dimensions = page.dimensions or {}
    width = dimensions.get('width')
    height = dimensions.get('height')
    if not isinstance(width, (int, float)) or not isinstance(
        height, (int, float)
    ):
        return None
    if width <= 0 or height <= 0:
        return None
    return float(width), float(height)


def _normalized_bbox(
    image: OCRImage, page: OCRPage
) -> tuple[float, float, float, float] | None:
    """Normalize an image's page-space box to source-page fractions."""
    bbox = _coordinates(image)
    dimensions = _page_dimensions(page)
    if bbox is None or dimensions is None:
        return None
    width, height = dimensions
    left, top, right, bottom = bbox
    return left / width, top / height, right / width, bottom / height


def _overlaps(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    """Return whether two page-space boxes have positive-area overlap."""
    first_left, first_top, first_right, first_bottom = first
    second_left, second_top, second_right, second_bottom = second
    return max(first_left, second_left) < min(
        first_right, second_right
    ) and max(first_top, second_top) < min(first_bottom, second_bottom)


def _write_image(data: str, path: Path) -> bool:
    """Decode one embedded OCR image, returning whether it was written."""
    if not data:
        return False
    if data.startswith('data:'):
        data = data.split(',', 1)[-1]
    try:
        decoded = base64.b64decode(data, validate=True)
        path.write_bytes(decoded)
    except (binascii.Error, ValueError, OSError):
        return False
    return True


def _resolve_page_indices(
    response: OCRResponse, pages: list[int] | None
) -> list[int]:
    """Resolve and validate source page identities before materialization."""
    selected = set(pages or ())
    resolved: list[int] = []
    seen: set[int] = set()
    for order, page in enumerate(response.pages):
        page_index = page.index
        if page_index is None:
            if pages is not None:
                if order >= len(pages):
                    raise ValueError(
                        'Mistral OCR returned more pages than the requested set'
                    )
                page_index = pages[order]
            else:
                page_index = order
        if pages is not None and page_index not in selected:
            raise ValueError(
                f'Mistral OCR returned page {page_index} outside requested pages'
            )
        if page_index in seen:
            raise ValueError(
                f'Mistral OCR returned duplicate page {page_index}'
            )
        seen.add(page_index)
        resolved.append(page_index)
    return resolved


def _render_page(
    pdf: Any, page_index: int, path: Path, scale: float, pdf_path: Path
) -> tuple[int, int]:
    """Render one source page and return its pixel dimensions."""
    try:
        rendered = pdf[page_index].render(scale=scale).to_pil()
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(path)
        return rendered.size
    except Exception as exc:
        raise MistralOCRError(
            f'Failed to render page {page_index} from PDF {pdf_path}'
        ) from exc


def _block_bbox(
    block: OCRBlock,
    page: OCRPage,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return a clamped block crop box in rendered-image pixels."""
    values = _coordinates(block)
    if values is None:
        return None
    width, height = image_size
    dimensions = _page_dimensions(page)
    source_width, source_height = dimensions or (width, height)
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


def _materialize_page(
    page: OCRPage,
    page_index: int,
    document_dir: Path,
    pdf: Any,
    render_scale: float,
    block_crop_scale: float,
    pdf_path: Path,
) -> list[OCRArtifact]:
    """Render one page and build its ordered block artifacts."""
    page_path = document_dir / 'Document.png'
    image_size = _render_page(
        pdf, page_index, page_path, render_scale, pdf_path
    )
    materialized_images: list[tuple[OCRImage, Path]] = []
    images_dir = document_dir / 'Images'
    images_dir.mkdir(parents=True, exist_ok=True)
    for image_index, image in enumerate(page.images):
        image_path = images_dir / f'Image_{image_index:03d}.png'
        if _write_image(image.image_base64 or '', image_path):
            materialized_images.append((image, image_path))

    artifacts: list[OCRArtifact] = []
    with Image.open(page_path) as opened:
        page_image = opened.convert('RGB')
        for block_index, block in enumerate(page.blocks):
            block_bbox = _block_bbox(block, page, image_size)
            block_coordinates = _coordinates(block)
            attached_images = [
                OCRImageArtifact(
                    path=str(image_path),
                    image_id=image.id,
                    bbox=_normalized_bbox(image, page),
                )
                for image, image_path in materialized_images
                if block_coordinates is not None
                and _coordinates(image) is not None
                and _overlaps(block_coordinates, _coordinates(image))
            ]
            crop_path: str | None = None
            if block_bbox is not None:
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
                crop_path = str(path)
            artifacts.append(
                OCRArtifact(
                    page_index=page_index,
                    block_index=block_index,
                    block_type=block.type,
                    content=block.content,
                    images=attached_images,
                    crop_path=crop_path,
                    crop_bbox=block_bbox,
                )
            )
    page_image.close()
    return artifacts


def _materialize_artifacts(
    response: OCRResponse,
    pdf_path: Path,
    pages: list[int] | None,
    output_dir: str | Path,
    render_scale: float,
    block_crop_scale: float,
) -> list[OCRArtifact]:
    """Materialize response images and crops as provider-neutral artifacts."""
    page_indices = _resolve_page_indices(response, pages)
    output_root = Path(output_dir)
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        artifacts: list[OCRArtifact] = []
        for page, page_index in zip(response.pages, page_indices, strict=True):
            document_dir = (
                output_root / 'Documents' / f'Document_{page_index:04d}'
            )
            artifacts.extend(
                _materialize_page(
                    page,
                    page_index,
                    document_dir,
                    pdf,
                    render_scale,
                    block_crop_scale,
                    pdf_path,
                )
            )
        return artifacts
    finally:
        pdf.close()


class MistralOCRProvider:
    """Adapt Mistral OCR responses to KMS2 correction artifacts."""

    def extract(
        self,
        pdf_path: str | Path,
        *,
        pages: list[int] | None = None,
    ) -> list[OCRArtifact]:
        """OCR a PDF and materialize its ordered KMS2 artifacts."""
        source_path = Path(pdf_path)
        response = ocr_pdf(source_path.read_bytes(), pages=pages)
        settings = config.get_settings()
        return _materialize_artifacts(
            response,
            source_path,
            pages,
            settings.ocr.output_dir,
            settings.ocr.render_scale,
            settings.ocr.block_crop_scale,
        )
