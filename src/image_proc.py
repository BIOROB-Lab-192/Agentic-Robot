import base64
from io import BytesIO

from PIL import Image


def preprocess_image(image_path, max_size=(1024, 1024)):
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
