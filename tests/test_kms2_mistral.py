import base64

import pytest
from PIL import Image

from kms2 import config
from kms2.ocr import mistral


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


class _PdfPage:
    def render(self, scale):
        assert scale == 1.0
        return _RenderedPage()


class _PdfDocument:
    def __getitem__(self, page_index):
        assert page_index in {0, 2}
        return _PdfPage()

    def close(self):
        pass


def _configure_materialization(monkeypatch, tmp_path):
    settings = config.Settings(
        ocr={'output_dir': str(tmp_path / 'output')},
        mistral_ocr={'api_key': 'test-key'},
    )
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    monkeypatch.setattr(
        mistral.pdfium,
        'PdfDocument',
        lambda path: _PdfDocument(),
    )


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
    raw = {
        'pages': [{'index': 0, 'blocks': [{'type': 'text', 'content': 'hi'}]}]
    }

    response = mistral.OCRResponse.from_raw(raw)

    assert response.pages[0].blocks[0].content == 'hi'
    assert response.raw_response is raw


def test_ocr_pdf_sends_pdf_bytes_and_applies_pages(monkeypatch):
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return _Response({'pages': []})

    monkeypatch.setattr(mistral, '_require_key', lambda: 'test-key')
    monkeypatch.setattr(mistral.httpx, 'post', fake_post)

    response = mistral.ocr_pdf(b'%PDF', pages=[2])

    assert response.pages == []
    assert calls[0][1]['document']['document_url'] == (
        'data:application/pdf;base64,JVBERg=='
    )
    assert calls[0][1]['pages'] == [2]
    assert calls[0][2]['Authorization'] == 'Bearer test-key'


def test_ocr_pdf_translates_http_failures(monkeypatch):
    request = mistral.OCRRequest(model='model', document_url='url')
    error = mistral.httpx.HTTPError('offline')

    monkeypatch.setattr(mistral, '_require_key', lambda: 'test-key')
    monkeypatch.setattr(
        mistral.httpx,
        'post',
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(mistral.MistralOCRError, match='request failed'):
        mistral._request_ocr(request)


def test_mistral_provider_materializes_ordered_artifacts(monkeypatch, tmp_path):
    _configure_materialization(monkeypatch, tmp_path)
    pdf_path = tmp_path / 'book.pdf'
    pdf_path.write_bytes(b'%PDF')
    embedded = base64.b64encode(b'embedded-image').decode('ascii')
    response = mistral.OCRResponse.from_raw(
        {
            'pages': [
                {
                    'index': 0,
                    'dimensions': {'width': 100, 'height': 200},
                    'images': [
                        {
                            'id': 'image-1',
                            'image_base64': embedded,
                            'top_left_x': 20,
                            'top_left_y': 30,
                            'bottom_right_x': 40,
                            'bottom_right_y': 60,
                        }
                    ],
                    'blocks': [
                        {
                            'type': 'text',
                            'content': 'first',
                            'top_left_x': 10,
                            'top_left_y': 20,
                            'bottom_right_x': 50,
                            'bottom_right_y': 100,
                        },
                        {'type': 'text', 'content': 'second'},
                    ],
                }
            ]
        }
    )
    monkeypatch.setattr(mistral, 'ocr_pdf', lambda data, pages=None: response)

    artifacts = mistral.MistralOCRProvider().extract(pdf_path)

    assert [artifact.content for artifact in artifacts] == ['first', 'second']
    assert artifacts[0].block_type == 'text'
    assert artifacts[0].page_index == 0
    assert artifacts[0].crop_bbox == (2, 12, 58, 108)
    assert artifacts[0].crop_path == str(
        tmp_path / 'output/Documents/Document_0000/Blocks/Block_0000.png'
    )
    assert artifacts[0].images[0].image_id == 'image-1'
    assert artifacts[0].images[0].bbox == (0.2, 0.15, 0.4, 0.3)
    assert artifacts[1].crop_path is None
    assert artifacts[1].crop_bbox is None
    assert (
        tmp_path / 'output/Documents/Document_0000/Images/Image_000.png'
    ).read_bytes() == b'embedded-image'


def test_mistral_provider_resolves_requested_page_without_index(
    monkeypatch, tmp_path
):
    _configure_materialization(monkeypatch, tmp_path)
    pdf_path = tmp_path / 'book.pdf'
    pdf_path.write_bytes(b'%PDF')
    response = mistral.OCRResponse.from_raw(
        {
            'pages': [
                {
                    'blocks': [
                        {
                            'type': 'text',
                            'content': 'page',
                            'top_left_x': 10,
                            'top_left_y': 20,
                            'bottom_right_x': 50,
                            'bottom_right_y': 100,
                        }
                    ]
                }
            ]
        }
    )
    monkeypatch.setattr(mistral, 'ocr_pdf', lambda data, pages=None: response)

    artifacts = mistral.MistralOCRProvider().extract(pdf_path, pages=[2])

    assert artifacts[0].page_index == 2
    assert artifacts[0].crop_path == str(
        tmp_path / 'output/Documents/Document_0002/Blocks/Block_0000.png'
    )


def test_mistral_provider_rejects_invalid_page_identity_before_writing(
    monkeypatch, tmp_path
):
    _configure_materialization(monkeypatch, tmp_path)
    pdf_path = tmp_path / 'book.pdf'
    pdf_path.write_bytes(b'%PDF')
    response = mistral.OCRResponse.from_raw(
        {'pages': [{'index': 1, 'blocks': [{'type': 'text'}]}]}
    )
    monkeypatch.setattr(mistral, 'ocr_pdf', lambda data, pages=None: response)

    with pytest.raises(ValueError, match='outside requested pages'):
        mistral.MistralOCRProvider().extract(pdf_path, pages=[2])

    assert not (tmp_path / 'output').exists()


def test_mistral_provider_rejects_duplicate_page_identity_before_writing(
    monkeypatch, tmp_path
):
    _configure_materialization(monkeypatch, tmp_path)
    pdf_path = tmp_path / 'book.pdf'
    pdf_path.write_bytes(b'%PDF')
    response = mistral.OCRResponse.from_raw(
        {
            'pages': [
                {'index': 2, 'blocks': []},
                {'index': 2, 'blocks': []},
            ]
        }
    )
    monkeypatch.setattr(mistral, 'ocr_pdf', lambda data, pages=None: response)

    with pytest.raises(ValueError, match='duplicate page'):
        mistral.MistralOCRProvider().extract(pdf_path, pages=[2, 3])

    assert not (tmp_path / 'output').exists()


def test_mistral_provider_omits_malformed_embedded_images(
    monkeypatch, tmp_path
):
    _configure_materialization(monkeypatch, tmp_path)
    pdf_path = tmp_path / 'book.pdf'
    pdf_path.write_bytes(b'%PDF')
    response = mistral.OCRResponse.from_raw(
        {
            'pages': [
                {
                    'images': [{'image_base64': 'not-valid-base64'}],
                    'blocks': [{'type': 'image', 'content': 'diagram'}],
                }
            ]
        }
    )
    monkeypatch.setattr(mistral, 'ocr_pdf', lambda data, pages=None: response)

    artifacts = mistral.MistralOCRProvider().extract(pdf_path)

    assert artifacts[0].images == []
    assert artifacts[0].page_index == 0
    assert not list((tmp_path / 'output').rglob('Image_*.png'))
