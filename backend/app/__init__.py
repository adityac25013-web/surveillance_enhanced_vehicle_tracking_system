from flask import Flask
from flask_cors import CORS

from .config import load_config
from .db.session import init_engine
from .routes import register_routes


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    config = load_config()
    app.config.update(config)

    init_engine(config["SQLALCHEMY_DATABASE_URI"])

    register_routes(app)

    return app

