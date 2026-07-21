"""
Motor de SQLite + fábrica de sesiones. Un solo archivo local (fitcoach.db);
no requiere levantar un servidor de base de datos aparte.
"""

from sqlmodel import Session, SQLModel, create_engine

DATABASE_URL = "sqlite:///fitcoach.db"

engine = create_engine(DATABASE_URL, echo=False)


def create_db_and_tables() -> None:
    # Importa los modelos para que su metadata quede registrada antes del create_all.
    from src.db import models  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
