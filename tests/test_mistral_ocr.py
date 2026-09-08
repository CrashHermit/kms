from pathlib import Path

import pytest

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
        'include_blocks': True,
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

    result = ocr.ocr_pdf(
        b'%PDF',
        pages=[2],
        options=ocr.OCRRequestOptions(include_blocks=True),
    )

    assert isinstance(result, ocr.OCRResponse)
    assert result.pages == []
    assert calls[0][1]['pages'] == [2]
    assert calls[0][1]['include_blocks'] is True


@pytest.mark.parametrize(
    ('provider_type', 'canonical_type'),
    [
        ('text', models.NodeType.PARAGRAPH),
        ('paragraph', models.NodeType.PARAGRAPH),
        ('aside_text', models.NodeType.PARAGRAPH),
        ('heading', models.NodeType.HEADER),
        ('equation', models.NodeType.MATH),
        ('code', models.NodeType.CODE),
        ('list', models.NodeType.LIST),
        ('table', models.NodeType.TABLE),
        ('image', models.NodeType.IMAGE),
    ],
)
def test_mistral_block_types_map_to_canonical_types(
    provider_type, canonical_type
):
    block = ocr.OCRBlock(type=provider_type)
    region = ocr.OCRBlockRegion(block_index=0, block=block)
    assert region.canonical_type is canonical_type


def test_unknown_mistral_block_type_is_rejected():
    region = ocr.OCRBlockRegion(
        block_index=0,
        block=ocr.OCRBlock(type='unknown'),
    )
    with pytest.raises(ValueError, match='Unknown Mistral OCR block type'):
        ocr._canonical_node_type(region.block.type)


def test_references_mistral_block_is_bibliographic_content():
    assert (
        ocr._canonical_node_type('references') == models.NodeType.BIBLIOGRAPHIC
    )


def test_materialize_document_preserves_references_block(tmp_path):
    response = ocr.OCRResponse.model_validate(
        {
            'pages': [
                {
                    'index': 0,
                    'dimensions': {'width': 100, 'height': 100},
                    'markdown': 'body\n\nReferences',
                    'blocks': [
                        {'type': 'text', 'content': 'body'},
                        {
                            'type': 'references',
                            'content': 'References\n[1] Example',
                            'top_left_x': 10,
                            'top_left_y': 10,
                            'bottom_right_x': 90,
                            'bottom_right_y': 90,
                        },
                    ],
                }
            ]
        }
    )
    from PIL import Image

    image_path = tmp_path / 'page.png'
    Image.new('RGB', (100, 100), 'white').save(image_path)
    source = ocr.materialize_document(
        response, tmp_path / 'materialized', pdf_path=None, render_pages=False
    )
    document = source.documents[0]
    assert [node.type for node in document.nodes] == [
        models.NodeType.PARAGRAPH,
        models.NodeType.BIBLIOGRAPHIC,
    ]
    assert document.nodes[1].provenance['provider_type'] == 'references'
    assert document.nodes[1].content == 'References\n[1] Example'


def test_build_source_materializes_blockless_page_images(tmp_path):
    resp = {
        'pages': [
            {
                'index': 0,
                'markdown': 'ignored source page markdown',
                'images': [
                    _img('img-0.jpeg'),
                    _img('img-1.jpeg', data_url=True),
                ],
            }
        ]
    }
    source = ocr.build_source(ocr.OCRResponse.model_validate(resp), tmp_path)

    document = source.documents[0]
    assert document.index == 0
    assert [node.type for node in document.nodes] == ['image', 'image']
    assert [asset.path for node in document.nodes for asset in node.assets] == [
        str(
            tmp_path
            / 'Documents'
            / 'Document_0000'
            / 'Images'
            / 'Image_000.png'
        ),
        str(
            tmp_path
            / 'Documents'
            / 'Document_0000'
            / 'Images'
            / 'Image_001.png'
        ),
    ]


def test_image_block_discards_provider_placeholder_content(tmp_path):
    artifact = ocr.OCRPageArtifact(
        index=0,
        image_path=str(tmp_path / 'page.png'),
        picture_paths=[str(tmp_path / 'Image_000.png')],
        blocks=[
            ocr.OCRBlockRegion(
                block_index=0,
                block=ocr.OCRBlock(
                    type='image',
                    content='![img-0.jpeg](img-0.jpeg)',
                ),
            )
        ],
    )

    node = artifact.to_document().nodes[0]

    assert node.type is models.NodeType.IMAGE
    assert node.content is None
    assert [asset.path for asset in node.assets] == [
        str(tmp_path / 'Image_000.png')
    ]


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
    artifact = ocr.OCRPageArtifact(
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
    ocr._materialize_block_crops([artifact], response)
    region = artifact.blocks[0]
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
    source = ocr.build_source(resp, tmp_path)
    assert len(source.documents[0].nodes[0].assets) == 1
    assert Path(source.documents[0].nodes[0].assets[0].path).exists()


def test_materialize_images_preserves_provider_order(tmp_path):
    paths = ocr._materialize_images(
        [
            ocr.OCRImage.model_validate(_img('first')),
            ocr.OCRImage.model_validate(_img('second')),
        ],
        tmp_path / 'Document_0000',
    )

    assert [Path(path).name for path in paths] == [
        'Image_000.png',
        'Image_001.png',
    ]


def test_footer_becomes_a_footer_node_without_page_markdown(tmp_path):
    resp = {
        'pages': [
            {
                'index': 0,
                'markdown': 'ignored source page markdown',
                'footer': '$^1$G. Polya, "Two Incidents," 1970.',
                'images': [],
            }
        ]
    }
    document = ocr.build_source(
        ocr.OCRResponse.model_validate(resp), tmp_path
    ).documents[0]

    assert [node.type for node in document.nodes] == ['footer']
    assert document.nodes[0].content == '$^1$G. Polya, "Two Incidents," 1970.'


def test_ocr_node_reads_graph_input_and_emits_documents(monkeypatch, tmp_path):
    document_list = [object()]
    calls = []

    class Source:
        key = None
        metadata = {}

    fake_source = models.Source(documents=[])
    fake_source.documents = document_list

    def fake_extract(pdf_path, output_dir, pages):
        calls.append((pdf_path, output_dir, pages))
        return fake_source

    monkeypatch.setattr(ocr, 'extract', fake_extract)
    result = ocr.OCRNode().run(
        {
            'pdf_path': 'book.pdf',
            'output_dir': str(tmp_path),
            'pages': [2],
        }
    )

    assert calls == [('book.pdf', str(tmp_path), [2])]
    assert result == {'source': fake_source, 'documents': document_list}


def test_pages_are_indexed_densely(tmp_path):
    resp = {
        'pages': [
            {'index': 5, 'markdown': 'ignored', 'images': [_img('img-0.jpeg')]},
            {'index': 9, 'markdown': 'ignored', 'images': [_img('img-0.jpeg')]},
        ]
    }
    source = ocr.build_source(ocr.OCRResponse.model_validate(resp), tmp_path)

    assert [document.index for document in source.documents] == [0, 1]
    assert all(
        [node.type for node in document.nodes] == ['image']
        and len(document.nodes[0].assets) == 1
        for document in source.documents
    )
