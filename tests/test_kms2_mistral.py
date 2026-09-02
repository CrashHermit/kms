import pytest

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
