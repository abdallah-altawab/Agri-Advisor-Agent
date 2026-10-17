import re

import pytest
from PIL import Image

from agri_advisor.services.errors import ServiceError
from agri_advisor.vision.images import MAX_BYTES, resolve_image, store_image


def test_store_valid_image_returns_id_and_clean_jpeg(make_image, uploads_dir):
    image_id = store_image(make_image("leaf.png"), uploads_dir)
    assert re.fullmatch(r"[0-9a-f]{32}\.jpg", image_id)
    with Image.open(uploads_dir / image_id) as stored:
        assert stored.format == "JPEG"


def test_store_strips_exif_metadata(tmp_path, uploads_dir):
    exif = Image.Exif()
    exif[0x010F] = "TestCam"  # the 'Make' tag
    source = tmp_path / "photo.jpg"
    Image.new("RGB", (128, 128)).save(source, exif=exif)
    with Image.open(source) as original:
        assert len(original.getexif()) > 0  # precondition: the source really has EXIF

    image_id = store_image(source, uploads_dir)

    with Image.open(uploads_dir / image_id) as stored:
        assert len(stored.getexif()) == 0


def test_rejects_non_image_file(tmp_path, uploads_dir):
    fake = tmp_path / "notes.jpg"
    fake.write_text("hello")
    with pytest.raises(ServiceError, match="not a readable image"):
        store_image(fake, uploads_dir)


def test_rejects_too_small_image(make_image, uploads_dir):
    with pytest.raises(ServiceError, match="too small"):
        store_image(make_image(size=(10, 10)), uploads_dir)


def test_rejects_oversized_file(tmp_path, uploads_dir):
    big = tmp_path / "big.jpg"
    big.write_bytes(b"0" * (MAX_BYTES + 1))
    with pytest.raises(ServiceError, match="too large"):
        store_image(big, uploads_dir)


def test_rejects_missing_file(tmp_path, uploads_dir):
    with pytest.raises(ServiceError, match="not found"):
        store_image(tmp_path / "nope.png", uploads_dir)


def test_resolve_image_returns_stored_file(make_image, uploads_dir):
    image_id = store_image(make_image(), uploads_dir)
    assert resolve_image(image_id, uploads_dir) == uploads_dir / image_id


@pytest.mark.parametrize("bad_id", ["../secret.txt", "abc.jpg", "0" * 32 + ".png"])
def test_resolve_image_rejects_unsafe_ids(bad_id, uploads_dir):
    with pytest.raises(ServiceError, match="Invalid image id"):
        resolve_image(bad_id, uploads_dir)


def test_resolve_image_reports_missing_file(uploads_dir):
    with pytest.raises(ServiceError, match="not found"):
        resolve_image("0" * 32 + ".jpg", uploads_dir)