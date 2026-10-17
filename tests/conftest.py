import pytest
from PIL import Image


@pytest.fixture
def uploads_dir(tmp_path):
    return tmp_path / "uploads"


@pytest.fixture
def make_image(tmp_path):
    def _make(name="leaf.png", size=(128, 128), color=(30, 160, 60)):
        path = tmp_path / name
        Image.new("RGB", size, color).save(path)
        return path

    return _make