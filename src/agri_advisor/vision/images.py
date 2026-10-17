"""Safe intake of user-supplied images: validate, strip metadata, store under an opaque id."""

import re
import uuid
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from agri_advisor.services.errors import ServiceError

MAX_BYTES = 8 * 1024 * 1024
MIN_SIDE_PX = 64
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
_ID_PATTERN = re.compile(r"[0-9a-f]{32}\.jpg")


def store_image(src: Path, uploads_dir: Path) -> str:
    """Validate an image file and store a clean JPEG copy; return its opaque image id.

    The copy is re-encoded after applying the EXIF orientation, which drops all
    metadata (including GPS coordinates).
    """
    src = Path(src)
    if not src.is_file():
        raise ServiceError("Image file not found.")
    if src.stat().st_size > MAX_BYTES:
        raise ServiceError(f"Image is too large (limit {MAX_BYTES // (1024 * 1024)} MB).")

    try:
        with Image.open(src) as probe:
            image_format = probe.format
            probe.verify()
        if image_format not in ALLOWED_FORMATS:
            raise ServiceError("Unsupported image type. Use JPEG, PNG or WebP.")
        with Image.open(src) as image:
            image.load()
            if min(image.size) < MIN_SIDE_PX:
                raise ServiceError("Image is too small to analyse. Use a larger photo.")
            clean = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ServiceError("The file is not a readable image.") from exc

    uploads_dir = Path(uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    image_id = f"{uuid.uuid4().hex}.jpg"
    clean.save(uploads_dir / image_id, format="JPEG", quality=95)
    return image_id


def resolve_image(image_id: str, uploads_dir: Path) -> Path:
    """Map an opaque image id to a file inside uploads_dir. Never accepts arbitrary paths."""
    image_id = image_id.strip()
    if not _ID_PATTERN.fullmatch(image_id):
        raise ServiceError("Invalid image id.")
    path = Path(uploads_dir) / image_id
    if not path.is_file():
        raise ServiceError("Image not found. Ask the user to attach the photo again.")
    return path