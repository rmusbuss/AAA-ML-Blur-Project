from aiohttp.web import Response, View
from aiohttp_jinja2 import render_template
from image import image_to_img_src, open_image
from mlapi import MLApi


class IndexView(View):
    template = "index.html"

    async def get(self) -> Response:
        ctx = {}
        return render_template(self.template, self.request, ctx)

    async def post(self) -> Response:
        try:
            form = await self.request.post()
            print(form)
            image = open_image(form["image"].file)
            result = MLApi('aaa-ml-blur-project_blur-backend_1').run_model(image)
            image_b64 = image_to_img_src(result)
            ctx = {"processed_image": image_b64}
        except Exception as err:
            ctx = {"error": f"Обработка файла не удалась по причине: {err}"}
        return render_template(self.template, self.request, ctx)
