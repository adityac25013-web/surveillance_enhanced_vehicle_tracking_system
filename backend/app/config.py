import os
from pathlib import Path


def load_config() -> dict:
    base_dir = Path(__file__).resolve().parents[1]

    db_url = os.getenv("DATABASE_URL") or f"sqlite:///{base_dir / 'vehicle_surveillance.db'}"

    return {
        "SQLALCHEMY_DATABASE_URI": db_url,
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "MODEL_DIR": str(base_dir / "models"),
        "SECRET_KEY": os.getenv("SECRET_KEY", "dev-secret-key-change-me"),
    }

