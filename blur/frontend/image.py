from io import BufferedReader, BytesIO
from base64 import b64encode
from PIL import Image


def open_image(image_fp: BufferedReader) -> Image:
    return Image.open(image_fp)


def image_b64encode(image: Image) -> str:
    with BytesIO() as io:
        image.save(io, format="png", quality=100)
        io.seek(0)
        return b64encode(io.read()).decode()


def image_to_img_src(image: Image) -> str:
    return f"data:image/png;base64,{image_b64encode(image)}"