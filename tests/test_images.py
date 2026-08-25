import base64

import dspy
import pytest
from PIL import Image

from kms.core import images


def test_load_image_returns_data_uri(tmp_path):
    path = tmp_path / 'source.png'
    Image.new('RGB', (10, 4), (255, 0, 0)).save(path)

    result = images.load_image(str(path))

    assert result is not None
    assert result.url.startswith('data:image/png;base64,')


def test_load_image_resizes_longest_side(tmp_path):
    path = tmp_path / 'source.png'
    Image.new('RGB', (100, 20), (255, 0, 0)).save(path)

    result = images.load_image(str(path), max_dim=10)

    assert result is not None
    payload = base64.b64decode(result.url.split(',', 1)[1])
    with Image.open(__import__('io').BytesIO(payload)) as resized:
        assert resized.size == (10, 2)


def test_load_image_missing_path_returns_none():
    assert images.load_image(None) is None


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        images.load_image(str(tmp_path / 'missing.png'))


def test_image_url_preserves_data_uri_and_http_url():
    data = dspy.Image(url='data:image/png;base64,AAAA')
    remote = dspy.Image(url='https://example.test/image.png')

    assert images.image_url(data) == data.url
    assert images.image_url(remote) == remote.url
