from aiohttp.web import Response, View
from aiohttp_jinja2 import render_template
from image import image_to_img_src, open_image


class IndexView(View):
    template = "index.html"

    async def get(self) -> Response:
        ctx = {}
        return render_template(self.template, self.request, ctx)

    async def post(self) -> Response:
        try:
            form = await self.request.post()
            image = open_image(form["image"].file)
            image_b64 = image_to_img_src(image)
            ctx = {"image": image_b64, "words": "hello"}
        except AttributeError as err:
            ctx = {"error": f"Ошибка при загрузке файла {str(err), type(err)}"}
        except Exception as err:
            ctx = {"error": f"Неожиданная ошибка {str(err), type(err)}"}
        return render_template(self.template, self.request, ctx)
