"""
Puente entre los esquemas usados por el grafo (GraphState/UserProfile/
WorkoutRoutine/DietPlan) y las tablas SQLModel persistidas en SQLite.

src/nodes.py no importa nada de aquí: toda la lectura/escritura a la DB
vive en las capas que invocan al grafo (main.py, src/api.py), para que los
nodos sigan siendo funciones puras de GraphState.
"""

from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlmodel import select

from src.db.engine import get_session
from src.db.models import (
    DEFAULT_USER_ID,
    ConversationMessageRecord,
    DietRecord,
    FeedbackRecord,
    RoutineRecord,
    UserProfileRecord,
)
from src.schemas.output_schemas import DietPlan, UserProfile, WorkoutRoutine
from src.state import GraphState

_ROLE_TO_MESSAGE = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
}
_MESSAGE_TO_ROLE = {
    HumanMessage: "human",
    AIMessage: "ai",
    SystemMessage: "system",
}


def current_week_number(when: Optional[datetime] = None) -> int:
    """Codifica una fecha como AAAASS (año, semana ISO), ej. 202629."""
    when = when or datetime.now(timezone.utc)
    iso_year, iso_week, _ = when.isocalendar()
    return iso_year * 100 + iso_week


def get_or_create_profile() -> UserProfile:
    with get_session() as session:
        record = session.get(UserProfileRecord, DEFAULT_USER_ID)
        if record is None:
            return UserProfile()
        data = record.model_dump(exclude={"id", "created_at", "updated_at"})
        return UserProfile(**data)


def save_profile(profile: UserProfile) -> None:
    with get_session() as session:
        record = session.get(UserProfileRecord, DEFAULT_USER_ID)
        data = profile.model_dump()
        if record is None:
            session.add(UserProfileRecord(id=DEFAULT_USER_ID, **data))
        else:
            for field, value in data.items():
                setattr(record, field, value)
            record.updated_at = datetime.now(timezone.utc)
            session.add(record)
        session.commit()


def load_messages() -> List[BaseMessage]:
    with get_session() as session:
        records = session.exec(
            select(ConversationMessageRecord)
            .where(ConversationMessageRecord.user_id == DEFAULT_USER_ID)
            .order_by(ConversationMessageRecord.turn_index)
        ).all()
        return [_ROLE_TO_MESSAGE[r.role](content=r.content) for r in records]


def append_message(message: BaseMessage) -> None:
    role = _MESSAGE_TO_ROLE.get(type(message))
    if role is None:
        raise ValueError(f"Tipo de mensaje no soportado para persistir: {type(message)}")
    with get_session() as session:
        last = session.exec(
            select(ConversationMessageRecord)
            .where(ConversationMessageRecord.user_id == DEFAULT_USER_ID)
            .order_by(ConversationMessageRecord.turn_index.desc())
        ).first()
        next_index = (last.turn_index + 1) if last else 0
        session.add(
            ConversationMessageRecord(
                user_id=DEFAULT_USER_ID,
                role=role,
                content=message.content,
                turn_index=next_index,
            )
        )
        session.commit()


def _latest_active(session, model):
    return session.exec(
        select(model)
        .where(model.user_id == DEFAULT_USER_ID, model.status == "active")
        .order_by(model.created_at.desc())
    ).first()


def get_latest_routine_record() -> Optional[RoutineRecord]:
    with get_session() as session:
        return _latest_active(session, RoutineRecord)


def get_latest_diet_record() -> Optional[DietRecord]:
    with get_session() as session:
        return _latest_active(session, DietRecord)


def save_routine(routine: WorkoutRoutine, week_number: int) -> RoutineRecord:
    with get_session() as session:
        for record in session.exec(
            select(RoutineRecord).where(
                RoutineRecord.user_id == DEFAULT_USER_ID, RoutineRecord.status == "active"
            )
        ).all():
            record.status = "superseded"
            session.add(record)
        new_record = RoutineRecord(
            user_id=DEFAULT_USER_ID,
            week_number=week_number,
            status="active",
            data=routine.model_dump(mode="json"),
        )
        session.add(new_record)
        session.commit()
        session.refresh(new_record)
        return new_record


def save_diet(diet: DietPlan, week_number: int) -> DietRecord:
    with get_session() as session:
        for record in session.exec(
            select(DietRecord).where(
                DietRecord.user_id == DEFAULT_USER_ID, DietRecord.status == "active"
            )
        ).all():
            record.status = "superseded"
            session.add(record)
        new_record = DietRecord(
            user_id=DEFAULT_USER_ID,
            week_number=week_number,
            status="active",
            data=diet.model_dump(mode="json"),
        )
        session.add(new_record)
        session.commit()
        session.refresh(new_record)
        return new_record


def save_feedback(
    raw_text: str,
    classification: dict,
    routine_id: Optional[int],
    diet_id: Optional[int],
    week_number: int,
) -> None:
    with get_session() as session:
        session.add(
            FeedbackRecord(
                user_id=DEFAULT_USER_ID,
                routine_id=routine_id,
                diet_id=diet_id,
                week_number=week_number,
                raw_text=raw_text,
                classification=classification,
            )
        )
        session.commit()


def load_full_state() -> GraphState:
    """Reconstruye el GraphState desde la DB para arrancar (o retomar) el grafo."""
    profile = get_or_create_profile()
    has_profile_data = any(
        getattr(profile, field) not in (None, [], "") for field in profile.model_fields
    )
    routine_record = get_latest_routine_record()
    diet_record = get_latest_diet_record()
    routine = WorkoutRoutine.model_validate(routine_record.data) if routine_record else None
    diet = DietPlan.model_validate(diet_record.data) if diet_record else None
    return {
        "messages": load_messages(),
        "user_profile": profile if has_profile_data else None,
        "missing_fields": [],
        "routine": routine,
        "diet": diet,
        "stage": "done" if (routine and diet) else "collecting",
        "safety_flag": False,
        "feedback_notes": None,
        "week_number": current_week_number(),
        "previous_routine": routine,
        "previous_diet": diet,
        # Semana en que se generó el plan activo: dispara el check-in semanal.
        "active_plan_week": routine_record.week_number if routine_record else None,
        "feedback_classification": None,
    }
