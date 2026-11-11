import base64
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from kms.config.services import OCRSettings
from kms.ocr import mistral
from kms.ocr.provider import source_pages_from_ocr


class _Response:
    def __init__(self, payload, status_error=None):
        self._payload = payload
        self._status_error = status_error
        self.text = 'error body'

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def json(self):
        return self._payload


class _RenderedPage:
    def to_pil(self):
        return Image.new('RGB', (100, 200), 'white')

    def close(self):
        assert mistral.PDFIUM_LOCK._is_owned()


class _PdfPage:
    def render(self, scale):
        assert scale == 1.0
        return _RenderedPage()

    def close(self):
        assert mistral.PDFIUM_LOCK._is_owned()


class _PdfDocument:
    def __getitem__(self, page_index):
        assert page_index in {0, 2}
        return _PdfPage()

    def close(self):
        pass


def _block_payload(content='text', **overrides):
    block = {
        'type': 'text',
        'content': content,
        'top_left_x': 10,
        'top_left_y': 20,
        'bottom_right_x': 50,
        'bottom_right_y': 100,
    }
    block.update(overrides)
    return block


def _image_payload(image_base64, **overrides):
    image = {
        'id': 'image-1',
        'image_base64': image_base64,
        'top_left_x': 20,
        'top_left_y': 30,
        'bottom_right_x': 40,
        'bottom_right_y': 60,
    }
    image.update(overrides)
    return image


def _page_payload(index=0, *, blocks=None, images=None, markdown=''):
    return {
        'index': index,
        'markdown': markdown,
        'dimensions': {'width': 100, 'height': 200},
        'blocks': [_block_payload()] if blocks is None else blocks,
        'images': [] if images is None else images,
    }


def _ocr_settings(**values) -> OCRSettings:
    return OCRSettings(**values)


def _configure_materialization(monkeypatch, tmp_path):
    ocr_settings = _ocr_settings(
        api_key='test-key',
        output_dir=str(tmp_path / 'output'),
    )
    monkeypatch.setattr(
        mistral.pdfium,
        'PdfDocument',
        lambda path: _PdfDocument(),
    )
    return ocr_settings


def test_ocr_request_builds_document_payload():
    request = mistral.OCRRequest(
        model='mistral-ocr-latest',
        document_url='data:application/pdf;base64,JVBERg==',
        options=mistral.OCRRequestOptions(pages=[2], table_format='markdown'),
    )

    assert request.payload() == {
        'model': 'mistral-ocr-latest',
        'document': {
            'type': 'document_url',
            'document_url': 'data:application/pdf;base64,JVBERg==',
        },
        'include_image_base64': True,
        'include_blocks': True,
        'extract_header': True,
        'extract_footer': True,
        'pages': [2],
        'table_format': 'markdown',
    }


def test_ocr_response_retains_raw_response():
    raw = {'pages': [_page_payload(blocks=[_block_payload('hi')])]}

    response = mistral.OCRResponse.from_raw(raw)

    assert response.pages[0].blocks[0].content == 'hi'
    assert response.raw_response is raw


def test_ocr_response_accepts_string_hyperlinks():
    page = _page_payload()
    page['hyperlinks'] = ['https://example.com/reference']
    raw = {'pages': [page]}

    response = mistral.OCRResponse.from_raw(raw)

    assert response.pages[0].hyperlinks == ['https://example.com/reference']


def test_ocr_pdf_sends_pdf_bytes_and_applies_pages(monkeypatch):
    calls = []
    ocr_settings = _ocr_settings(api_key='test-key')

    def fake_post(url, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return _Response({'pages': []})

    monkeypatch.setattr(mistral.httpx, 'post', fake_post)

    response = mistral.ocr_pdf(b'%PDF', ocr_settings, pages=[2])

    assert response.pages == []
    assert calls[0][1]['document']['document_url'] == (
        'data:application/pdf;base64,JVBERg=='
    )
    assert calls[0][1]['pages'] == [2]
    assert calls[0][2]['Authorization'] == 'Bearer test-key'


def test_ocr_pdf_translates_http_failures(monkeypatch):
    request = mistral.OCRRequest(model='model', document_url='url')
    error = mistral.httpx.HTTPError('offline')
    ocr_settings = _ocr_settings(api_key='test-key')

    monkeypatch.setattr(
        mistral.httpx,
        'post',
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(mistral.MistralOCRError, match='request failed'):
        mistral._request_ocr(request, ocr_settings)


def test_mistral_structural_labels_map_to_canonical_block_types():
    assert mistral._block_type('title') is mistral.BlockType.HEADER
    assert mistral._block_type('references') is mistral.BlockType.BIBLIOGRAPHIC


def test_concurrent_pdf_materialization_serializes_pdfium_operations(
    monkeypatch, tmp_path
):
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    class ConcurrentDocument:
        def __getitem__(self, page_index):
            assert page_index == 0
            assert mistral.PDFIUM_LOCK._is_owned()
            return self

        def render(self, scale):
            nonlocal active, maximum_active
            assert scale == 1.0
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.01)
            with state_lock:
                active -= 1
            return self

        def to_pil(self):
            assert mistral.PDFIUM_LOCK._is_owned()
            return Image.new('RGB', (100, 200), 'white')

        def close(self):
            assert mistral.PDFIUM_LOCK._is_owned()

    monkeypatch.setattr(
        mistral.pdfium, 'PdfDocument', lambda path: ConcurrentDocument()
    )
    response = mistral.OCRResponse.from_raw(
        {'pages': [_page_payload(blocks=[])]}
    )

    def materialize(index):
        return mistral.materialize_ocr_artifacts(
            response,
            tmp_path / 'book.pdf',
            tmp_path / f'output-{index}',
            1.0,
            1.0,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(materialize, range(2)))

    assert maximum_active == 1


def test_rendered_pages_materialize_without_pdfium_lock(monkeypatch, tmp_path):
    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError('rendered pages must not acquire PDFium lock')

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(mistral, 'PDFIUM_LOCK', ForbiddenLock())
    monkeypatch.setattr(
        mistral.pdfium,
        'PdfDocument',
        lambda path: (_ for _ in ()).throw(
            AssertionError('rendered pages must not open a PDFium document')
        ),
    )
    image_bytes = io.BytesIO()
    Image.new('RGB', (100, 200), 'white').save(image_bytes, format='PNG')
    response = mistral.OCRResponse.from_raw(
        {'pages': [_page_payload(blocks=[])]}
    )

    artifacts = mistral.materialize_ocr_artifacts(
        response,
        tmp_path / 'book.pdf',
        tmp_path / 'output',
        1.0,
        1.0,
        rendered_pages={0: image_bytes.getvalue()},
    )

    assert len(artifacts) == 1


def test_mistral_provider_materializes_ordered_artifacts(monkeypatch, tmp_path):
    ocr_settings = _configure_materialization(monkeypatch, tmp_path)
    pdf_path = tmp_path / 'book.pdf'
    pdf_path.write_bytes(b'%PDF')
    first_embedded = base64.b64encode(b'first-image').decode('ascii')
    second_embedded = 'data:image/png;base64,' + base64.b64encode(
        b'second-image'
    ).decode('ascii')
    response = mistral.OCRResponse.from_raw(
        {
            'pages': [
                _page_payload(
                    index=2,
                    markdown='## Second',
                    blocks=[
                        _block_payload(
                            'first',
                            type='aside_text',
                            top_left_x=10,
                            top_left_y=20,
                            bottom_right_x=50,
                            bottom_right_y=100,
                        ),
                        _block_payload(
                            'second',
                            top_left_x=60,
                            top_left_y=120,
                            bottom_right_x=90,
                            bottom_right_y=180,
                        ),
                    ],
                    images=[
                        _image_payload(first_embedded),
                        _image_payload(
                            second_embedded,
                            id='image-2',
                            top_left_x=70,
                            top_left_y=150,
                            bottom_right_x=80,
                            bottom_right_y=170,
                        ),
                    ],
                ),
                _page_payload(
                    index=0,
                    markdown='## First',
                    blocks=[_block_payload('third')],
                ),
            ]
        }
    )
    monkeypatch.setattr(
        mistral,
        'ocr_pdf',
        lambda data, settings, pages=None: response,
    )

    artifacts = mistral.MistralOCRProvider(ocr_settings).extract(
        pdf_path, source_uuid='source-1'
    )

    assert [page.page_index for page in artifacts] == [2, 0]
    assert [page.markdown for page in artifacts] == ['## Second', '## First']
    assert [block.content for page in artifacts for block in page.blocks] == [
        'first',
        'second',
        'third',
    ]
    assert artifacts[0].blocks[0].block_type.value == 'aside_text'
    assert [
        block.page_index for page in artifacts for block in page.blocks
    ] == [
        2,
        2,
        0,
    ]
    assert artifacts[0].blocks[0].crop_bbox == (2, 12, 58, 108)
    assert artifacts[0].blocks[1].crop_bbox == (52, 112, 98, 188)
    assert artifacts[1].blocks[0].crop_bbox == (2, 12, 58, 108)
    assert all(
        not hasattr(block, 'images')
        for page in artifacts
        for block in page.blocks
    )
    assert not (
        tmp_path / 'output' / 'Documents' / 'Document_0002' / 'Images'
    ).exists()


def test_ocr_runs_preserve_crop_artifacts_without_fragment_files(
    monkeypatch, tmp_path
):
    settings = _configure_materialization(monkeypatch, tmp_path)
    pdf_path = tmp_path / 'book.pdf'
    pdf_path.write_bytes(b'%PDF')
    payloads = iter([b'first image', b'second image', b'reprocessed image'])
    colors = iter(['red', 'green', 'blue'])

    def ocr_pdf(_data, _settings, pages=None):
        image = base64.b64encode(next(payloads)).decode('ascii')
        return mistral.OCRResponse.from_raw(
            {'pages': [_page_payload(images=[_image_payload(image)])]}
        )

    monkeypatch.setattr(mistral, 'ocr_pdf', ocr_pdf)
    monkeypatch.setattr(
        _RenderedPage,
        'to_pil',
        lambda self: Image.new('RGB', (100, 200), next(colors)),
    )
    provider = mistral.MistralOCRProvider(settings)
    blocks = [
        provider.extract(pdf_path, source_uuid=source_uuid)[0].blocks[0]
        for source_uuid in ('source-1', 'source-2', 'source-1')
    ]

    crop_paths = [Path(block.crop_path) for block in blocks]
    assert len(set(crop_paths)) == 3
    assert all(path.is_file() for path in crop_paths)
    assert not list((tmp_path / 'output').rglob('Images'))
    for block, expected in zip(
        blocks, [(255, 0, 0), (0, 128, 0), (0, 0, 255)], strict=True
    ):
        with Image.open(block.crop_path) as crop:
            assert crop.getpixel((0, 0)) == expected


@pytest.mark.parametrize('fragment_count', [0, 1, 3])
def test_image_ocr_block_uses_one_rendered_block_crop(
    monkeypatch, tmp_path, fragment_count
):
    settings = _configure_materialization(monkeypatch, tmp_path)
    pdf_path = tmp_path / 'book.pdf'
    pdf_path.write_bytes(b'%PDF')
    response = mistral.OCRResponse.from_raw(
        {
            'pages': [
                _page_payload(
                    index=2,
                    blocks=[_block_payload('diagram', type='image')],
                    images=[
                        _image_payload('not-base64', id=f'image-{index}')
                        for index in range(fragment_count)
                    ],
                )
            ]
        }
    )
    monkeypatch.setattr(mistral, 'ocr_pdf', lambda *args, **kwargs: response)

    artifacts = mistral.MistralOCRProvider(settings).extract(
        pdf_path, source_uuid='source-run'
    )
    pages = source_pages_from_ocr(
        artifacts, source_uuid='00000000-0000-0000-0000-000000000001'
    )
    image_block = pages[0].blocks[0]
    repeated = source_pages_from_ocr(
        artifacts, source_uuid='00000000-0000-0000-0000-000000000001'
    )

    assert image_block.image_path == artifacts[0].blocks[0].crop_path
    assert image_block.image_page_indexes == [2]
    assert image_block.crop_bbox == artifacts[0].blocks[0].crop_bbox
    assert Path(image_block.image_path).is_file()
    assert not list((tmp_path / 'output').rglob('Images'))
    assert image_block.uuid == repeated[0].blocks[0].uuid


@pytest.mark.parametrize(
    'missing_field',
    [
        'page_index',
        'page_dimensions',
        'block_coordinate',
        'block_content',
        'image_coordinate',
        'image_base64',
    ],
)
def test_mistral_response_requires_guaranteed_fields(missing_field):
    raw = {
        'pages': [
            _page_payload(
                blocks=[_block_payload()],
                images=[_image_payload('aGVsbG8=')],
            )
        ]
    }
    page = raw['pages'][0]
    block = page['blocks'][0]
    image = page['images'][0]

    if missing_field == 'page_index':
        page.pop('index')
    elif missing_field == 'page_dimensions':
        page.pop('dimensions')
    elif missing_field == 'block_coordinate':
        block.pop('top_left_x')
    elif missing_field == 'block_content':
        block.pop('content')
    elif missing_field == 'image_coordinate':
        image.pop('top_left_x')
    else:
        image.pop('image_base64')

    with pytest.raises(ValidationError):
        mistral.OCRResponse.from_raw(raw)
