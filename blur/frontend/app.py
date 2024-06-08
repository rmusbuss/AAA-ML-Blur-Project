import aiohttp_jinja2
import jinja2
from aiohttp.web import Application
from config import TEMPLATES_PATH, STATIC_PATH
from views import IndexView


def create_app() -> Application:
    app = Application(client_max_size=1024**2 * 50)

    # setup routes
    app.router.add_static("/static/", STATIC_PATH)
    app.router.add_view("/", IndexView, name="index")

    # setup templates
    aiohttp_jinja2.setup(
        app=app,
        loader=jinja2.FileSystemLoader(TEMPLATES_PATH),
    )
    # app["model"] = create_model()
    return app


async def async_create_app() -> Application:
    return create_app()

