import dspy
from PIL import Image

from kms.core import content


def test_from_parts_tags_text_and_images():
    image = dspy.Image(url='data:image/png;base64,AAAA')
    content_value = content.Content.from_parts(['What is this?', image])
    assert content_value.parts == [
        content.TextPart(text='What is this?'),
        content.ImagePart(image=image),
    ]


def test_from_text_makes_single_text_part():
    assert content.Content.from_text('hello').parts == [
        content.TextPart(text='hello')
    ]


def test_from_text_and_pictures_loads_image_paths(tmp_path):
    image_file = tmp_path / 'fig.png'
    Image.new('RGB', (10, 10), (255, 0, 0)).save(image_file)
    content_value = content.Content.from_text_and_pictures(
        'see the figure',
        [{'index': 0, 'document_index': 1, 'image_path': str(image_file)}],
    )
    assert content_value.parts[0] == content.TextPart(text='see the figure')
    assert isinstance(content_value.parts[1], content.ImagePart)


def test_from_text_and_pictures_skips_missing_images():
    content_value = content.Content.from_text_and_pictures(
        'text only', [{'index': 0, 'image_path': None}]
    )
    assert content_value.parts == [content.TextPart(text='text only')]


def test_render_marks_images():
    image = dspy.Image(url='data:image/png;base64,AAAA')
    content_value = content.Content.from_parts(
        ['What is this?', image, 'And this?']
    )
    assert content_value.render() == 'What is this? [image] And this?'


def test_openai_blocks_interleave_text_and_images():
    image = dspy.Image(url='data:image/png;base64,AAAA')
    content_value = content.Content.from_parts(['What is this?', image])
    assert content_value.openai_blocks() == [
        {'type': 'text', 'text': 'What is this?'},
        {
            'type': 'image_url',
            'image_url': {'url': 'data:image/png;base64,AAAA'},
        },
    ]


def test_embedding_blocks_use_flat_image_format():
    image = dspy.Image(url='data:image/png;base64,AAAA')
    content_value = content.Content.from_parts(['What is this?', image])
    assert content_value.embedding_blocks() == [
        {'type': 'text', 'text': 'What is this?'},
        {'type': 'image_base64', 'image_base64': 'data:image/png;base64,AAAA'},
    ]


def test_image_url_converts_local_path(tmp_path):
    path = tmp_path / 'img.png'
    path.write_bytes(b'\x89PNG\r\n\x1a\n')
    url = content.image_url(dspy.Image(url=str(path)))
    assert url.startswith('data:image/png;base64,')


def test_labeled_content_text_only_formats_one_text_block():
    nodes = [
        type(
            'NodeView',
            (),
            {
                'position': 0,
                'type': 'paragraph',
                'content': 'A statement.',
                'image_path': None,
                'marker': None,
            },
        )()
    ]

    assert content.labeled_content_parts(nodes).format() == [
        {'type': 'text', 'text': '[0] (paragraph): A statement.'}
    ]


def test_labeled_content_image_has_label_then_image_block(tmp_path):
    image_path = tmp_path / 'figure.png'
    Image.new('RGB', (4, 2), (0, 0, 255)).save(image_path)
    node = type(
        'NodeView',
        (),
        {
            'position': 1,
            'type': 'image',
            'content': '',
            'image_path': str(image_path),
            'marker': None,
        },
    )()

    blocks = content.labeled_content_parts([node]).format()

    assert blocks[0] == {'type': 'text', 'text': '[1] (image):'}
    assert blocks[1]['type'] == 'image_url'
    assert blocks[1]['image_url']['url'].startswith('data:image/png;base64,')


def test_labeled_content_preserves_text_image_text_block_order(tmp_path):
    image_path = tmp_path / 'figure.png'
    Image.new('RGB', (4, 2), (0, 0, 255)).save(image_path)
    nodes = [
        type(
            'NodeView',
            (),
            {
                'position': 0,
                'type': 'paragraph',
                'content': 'Before.',
                'image_path': None,
                'marker': None,
            },
        )(),
        type(
            'NodeView',
            (),
            {
                'position': 1,
                'type': 'image',
                'content': '',
                'image_path': str(image_path),
                'marker': None,
            },
        )(),
        type(
            'NodeView',
            (),
            {
                'position': 2,
                'type': 'paragraph',
                'content': 'After.',
                'image_path': None,
                'marker': None,
            },
        )(),
    ]

    blocks = content.labeled_content_parts(nodes).format()

    assert [block['type'] for block in blocks] == [
        'text',
        'text',
        'image_url',
        'text',
    ]
    assert blocks[0]['text'] == '[0] (paragraph): Before.'
    assert blocks[1]['text'] == '[1] (image):'
    assert blocks[3]['text'] == '[2] (paragraph): After.'


def test_labeled_content_empty_nodes_formats_empty_blocks():
    assert content.labeled_content_parts([]).format() == []


def test_load_image_returns_none_without_path():
    assert content.load_image(None) is None
    assert content.load_image('') is None
