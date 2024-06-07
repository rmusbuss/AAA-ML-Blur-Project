from aiohttp.web import run_app
from app import create_app
from config import PORT


def main() -> None:
    app = create_app()
    run_app(app, port=PORT)


if __name__ == "__main__":
    main()
