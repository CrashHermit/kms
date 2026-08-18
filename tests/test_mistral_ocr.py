from pathlib import Path

from kms.construction import ocr
from kms.core import models

_PNG_B64 = (
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4'
    '2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
)


def _img(image_id: str, data_url: bool = False) -> dict:
    b64 = f'data:image/png;base64,{_PNG_B64}' if data_url else _PNG_B64
    return {
        'id': image_id,
        'image_base64': b64,
        'top_left_x': 0,
        'top_left_y': 0,
        'bottom_right_x': 1,
        'bottom_right_y': 1,
    }


def test_ocr_request_options_include_prompts():
    options = ocr.OCRRequestOptions(
        table_format='html',
        document_annotation_format={'type': 'json_schema'},
        document_annotation_prompt='Extract document metadata.',
        bbox_annotation_format={'type': 'json_schema'},
    )

    metadata_options = ocr.OCRRequestOptions.with_image_metadata()
    assert metadata_options.bbox_annotation_format is not None
    assert options.model_dump(exclude_none=True) == {
        'include_image_base64': True,
        'include_blocks': False,
        'extract_header': True,
        'extract_footer': True,
        'table_format': 'html',
        'document_annotation_format': {'type': 'json_schema'},
        'document_annotation_prompt': 'Extract document metadata.',
        'bbox_annotation_format': {'type': 'json_schema'},
    }


def test_ocr_pdf_requests_blocks_when_enabled(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'pages': []}

    def fake_post(url, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return Response()

    monkeypatch.setattr(ocr, '_require_key', lambda: 'test-key')
    monkeypatch.setattr(ocr.httpx, 'post', fake_post)

    result = ocr.ocr_pdf(b'%PDF', pages=[2], include_blocks=True)

    assert isinstance(result, ocr.OCRResponse)
    assert result.pages == []
    assert calls[0][1]['pages'] == [2]
    assert calls[0][1]['include_blocks'] is True


def test_build_segments_rewrites_refs_and_saves_pictures(tmp_path):
    resp = {
        'pages': [
            {
                'index': 0,
                'markdown': (
                    '# Title\n\n![alt](img-0.jpeg)\n\n'
                    'prose $x^2$\n\n![alt2](img-1.jpeg)\n'
                ),
                'images': [
                    _img('img-0.jpeg'),
                    _img('img-1.jpeg', data_url=True),
                ],
            }
        ]
    }
    resp = ocr.OCRResponse.model_validate(resp)
    segs = ocr.build_segments(resp, tmp_path)
    assert len(segs) == 1
    segment = segs[0]
    assert segment.index == 0
    assert '![1]()' in segment.content and '![2]()' in segment.content
    assert (
        'img-0.jpeg' not in segment.content
        and 'img-1.jpeg' not in segment.content
    )
    assert [picture.index for picture in segment.pictures] == [1, 2]
    for picture in segment.pictures:
        assert (
            Path(picture.image_path).exists()
            and Path(picture.image_path).stat().st_size > 0
        )


def test_materialize_document_crops_blocks(tmp_path):
    resp = {
        'pages': [
            {
                'index': 0,
                'dimensions': {'width': 100, 'height': 100},
                'markdown': 'text',
                'blocks': [
                    {
                        'type': 'text',
                        'content': 'text',
                        'top_left_x': 10,
                        'top_left_y': 20,
                        'bottom_right_x': 50,
                        'bottom_right_y': 60,
                    }
                ],
            }
        ]
    }
    response = ocr.OCRResponse.model_validate(resp)
    image_path = tmp_path / 'page.png'
    from PIL import Image

    Image.new('RGB', (100, 100), 'white').save(image_path)
    document = models.Document(
        response=response,
        pages=[
            ocr.OCRPageArtifact(
                index=0,
                markdown='text',
                image_path=str(image_path),
                blocks=[
                    ocr.OCRBlockRegion(
                        block_index=0,
                        block=response.pages[0].blocks[0],
                    )
                ],
            )
        ],
    )
    ocr._materialize_block_crops(document)
    region = document.pages[0].blocks[0]
    assert region.crop_bbox == (2, 12, 58, 68)
    assert region.crop_path is not None
    assert Path(region.crop_path).exists()
    with Image.open(region.crop_path) as crop:
        assert crop.size == (56, 56)


def test_unreferenced_figure_is_still_saved(tmp_path):
    resp = {
        'pages': [
            {
                'index': 0,
                'markdown': 'prose only, no refs',
                'images': [_img('img-0.jpeg')],
            }
        ]
    }
    resp = ocr.OCRResponse.model_validate(resp)
    segs = ocr.build_segments(resp, tmp_path)
    assert len(segs[0].pictures) == 1
    assert Path(segs[0].pictures[0].image_path).exists()


def test_non_figure_link_left_untouched(tmp_path):
    md = 'see ![diagram](https://example.com/x.png) here'
    rewritten, pics = ocr._rewrite_page(
        md, [], tmp_path / 'Segments' / 'Segment_0000'
    )
    assert rewritten == md
    assert pics == []


def test_footer_is_appended_to_the_page_markdown(tmp_path):
    resp = {
        'pages': [
            {
                'index': 0,
                'markdown': 'body text',
                'header': '42\nCHAPTER 1. TOPOLOGICAL SPACES',
                'footer': '$^1$G. Polya, "Two Incidents," 1970.',
                'images': [],
            }
        ]
    }
    resp = ocr.OCRResponse.model_validate(resp)
    content = ocr.build_segments(resp, tmp_path)[0].content
    assert content == 'body text\n\n$^1$G. Polya, "Two Incidents," 1970.'
    assert 'TOPOLOGICAL SPACES' not in content


def test_page_without_a_footer_is_unchanged(tmp_path):
    resp = {
        'pages': [
            {'index': 0, 'markdown': 'body text', 'footer': None, 'images': []}
        ]
    }
    resp = ocr.OCRResponse.model_validate(resp)
    assert ocr.build_segments(resp, tmp_path)[0].content == 'body text'
    assert ocr._with_footer('body text', '   ') == 'body text'


def test_ocr_node_reads_graph_input_and_emits_segments(monkeypatch, tmp_path):
    segments = [object()]
    calls = []

    class Document:
        def to_segments(self):
            return segments

    fake_document = Document()

    def fake_extract(pdf_path, output_dir, pages):
        calls.append((pdf_path, output_dir, pages))
        return fake_document

    monkeypatch.setattr(ocr, 'extract', fake_extract)
    result = ocr.OCRNode().run(
        {
            'pdf_path': 'book.pdf',
            'output_dir': str(tmp_path),
            'pages': [2],
        }
    )

    assert calls == [('book.pdf', str(tmp_path), [2])]
    assert result == {'document': fake_document, 'segments': segments}


def test_pages_are_indexed_densely(tmp_path):
    resp = {
        'pages': [
            {
                'index': 5,
                'markdown': '![a](img-0.jpeg)',
                'images': [_img('img-0.jpeg')],
            },
            {
                'index': 9,
                'markdown': '![b](img-0.jpeg)',
                'images': [_img('img-0.jpeg')],
            },
        ]
    }
    resp = ocr.OCRResponse.model_validate(resp)
    segs = ocr.build_segments(resp, tmp_path)
    assert [s.index for s in segs] == [0, 1]
    assert all(len(s.pictures) == 1 for s in segs)
    assert all('![1]()' in s.content for s in segs)
