"""Basic Mistral OCR request and response models for KMS2."""

import base64
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from kms import config

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
        """Builds the Mistral API request payload."""
        return {
            'model': self.model,
            'document': {
                'type': 'document_url',
                'document_url': self.document_url,
            },
            **self.options.model_dump(exclude_none=True),
        }


def _require_key() -> str:
    """Returns the configured Mistral API key or raises."""
    key = config.get_settings().ocr.api_key
    if not key:
        raise MistralOCRError(
            'KMS_OCR__API_KEY is not set. Export your Mistral API key before '
            'running the Mistral OCR client.'
        )
    return key


def _request_ocr(request: OCRRequest) -> OCRResponse:
    """Sends one validated request to the Mistral OCR endpoint."""
    try:
        response = httpx.post(
            config.get_settings().ocr.url,
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
    """Runs Mistral OCR on PDF bytes and returns the validated response."""
    request_options = options or OCRRequestOptions()
    if pages is not None:
        request_options = request_options.model_copy(update={'pages': pages})
    request = OCRRequest(
        model=config.get_settings().ocr.model,
        document_url=(
            'data:application/pdf;base64,'
            + base64.b64encode(pdf_bytes).decode('ascii')
        ),
        options=request_options,
    )
    return _request_ocr(request)
