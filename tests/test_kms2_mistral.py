import base64

import pytest
from PIL import Image
from pydantic import ValidationError

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


def _page_payload(index=0, *, blocks=None, images=None):
    return {
        'index': index,
        'dimensions': {'width': 100, 'height': 200},
        'blocks': [_block_payload()] if blocks is None else blocks,
        'images': [] if images is None else images,
    }


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
    raw = {'pages': [_page_payload(blocks=[_block_payload('hi')])]}

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
    first_embedded = base64.b64encode(b'first-image').decode('ascii')
    second_embedded = base64.b64encode(b'second-image').decode('ascii')
    response = mistral.OCRResponse.from_raw(
        {
            'pages': [
                _page_payload(
                    index=2,
                    blocks=[
                        _block_payload(
                            'first',
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
                _page_payload(index=0, blocks=[_block_payload('third')]),
            ]
        }
    )
    monkeypatch.setattr(mistral, 'ocr_pdf', lambda data, pages=None: response)

    artifacts = mistral.MistralOCRProvider().extract(pdf_path)

    assert [artifact.content for artifact in artifacts] == [
        'first',
        'second',
        'third',
    ]
    assert [artifact.page_index for artifact in artifacts] == [2, 2, 0]
    assert artifacts[0].crop_bbox == (2, 12, 58, 108)
    assert artifacts[1].crop_bbox == (52, 112, 98, 188)
    assert artifacts[2].crop_bbox == (2, 12, 58, 108)
    assert artifacts[0].crop_path == str(
        tmp_path / 'output/Documents/Document_0002/Blocks/Block_0000.png'
    )
    assert artifacts[1].crop_path == str(
        tmp_path / 'output/Documents/Document_0002/Blocks/Block_0001.png'
    )
    assert artifacts[2].crop_path == str(
        tmp_path / 'output/Documents/Document_0000/Blocks/Block_0000.png'
    )
    assert [image.image_id for image in artifacts[0].images] == ['image-1']
    assert [image.image_id for image in artifacts[1].images] == ['image-2']
    assert artifacts[2].images == []
    assert artifacts[0].images[0].bbox == (0.2, 0.15, 0.4, 0.3)
    assert (
        tmp_path / 'output/Documents/Document_0002/Images/Image_000.png'
    ).read_bytes() == b'first-image'
    assert (
        tmp_path / 'output/Documents/Document_0002/Images/Image_001.png'
    ).read_bytes() == b'second-image'


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
