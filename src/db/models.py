"""
Tablas SQLModel para persistencia local. Estrategia de mapeo (ver plan):

- UserProfileRecord hereda de UserProfileBase (src/schemas/output_schemas.py)
  para no duplicar los campos del perfil.
- RoutineRecord / DietRecord guardan WorkoutRoutine/DietPlan como JSON blob
  (no se normalizan a tablas hijas: para un solo usuario local no aporta).
- ConversationMessageRecord / FeedbackRecord no tienen contraparte Pydantic
  en el grafo, son filas planas.

Este es un proyecto de un solo usuario local: DEFAULT_USER_ID se usa en vez
de manejo de usuarios real.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel

from src.schemas.output_schemas import UserProfileBase

DEFAULT_USER_ID = 1


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserProfileRecord(UserProfileBase, table=True):
    __tablename__ = "user_profile"

    id: Optional[int] = Field(default=None, primary_key=True)
    dietary_restrictions: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    allergies: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ConversationMessageRecord(SQLModel, table=True):
    __tablename__ = "conversation_message"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(default=DEFAULT_USER_ID, index=True)
    role: str  # "human" | "ai" | "system"
    content: str
    turn_index: int
    created_at: datetime = Field(default_factory=_utcnow)


class RoutineRecord(SQLModel, table=True):
    __tablename__ = "routine"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(default=DEFAULT_USER_ID, index=True)
    week_number: int  # ISO (year, week) codificado como AAAASS, ej. 202629
    status: str = Field(default="active")  # "active" | "superseded"
    data: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


class DietRecord(SQLModel, table=True):
    __tablename__ = "diet"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(default=DEFAULT_USER_ID, index=True)
    week_number: int
    status: str = Field(default="active")
    data: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


class FeedbackRecord(SQLModel, table=True):
    __tablename__ = "feedback"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(default=DEFAULT_USER_ID, index=True)
    routine_id: Optional[int] = Field(default=None, foreign_key="routine.id")
    diet_id: Optional[int] = Field(default=None, foreign_key="diet.id")
    week_number: int
    raw_text: str
    classification: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)
