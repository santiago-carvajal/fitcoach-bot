import os

import pytest

# src/nodes.py instancia ChatAnthropic al importarse; una key dummy permite
# importar los nodos en tests sin credenciales reales (el LLM se mockea).
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

from src.schemas.output_schemas import (  # noqa: E402
    Equipment,
    ExperienceLevel,
    Goal,
    UserProfile,
)


class FakeStructuredLLM:
    """LLM falso para tests de seams de grafo: captura los mensajes que
    recibe y devuelve una respuesta enlatada."""

    def __init__(self, response):
        self.response = response
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return self.response


@pytest.fixture
def make_fake_llm():
    """Factory del LLM falso, para no duplicar la clase en cada test de nodo."""

    def _make(response) -> FakeStructuredLLM:
        return FakeStructuredLLM(response)

    return _make


@pytest.fixture
def make_state():
    """Estado base completo del grafo listo para generar, con overrides."""

    def _make(profile, **overrides) -> dict:
        state = {
            "messages": [],
            "user_profile": profile,
            "missing_fields": [],
            "routine": None,
            "diet": None,
            "stage": "ready_to_generate",
            "safety_flag": False,
            "feedback_notes": None,
            "week_number": 202630,
            "previous_routine": None,
            "previous_diet": None,
            "active_plan_week": None,
            "feedback_classification": None,
        }
        state.update(overrides)
        return state

    return _make


@pytest.fixture
def make_profile():
    """Factory de perfiles completos; varía solo equipo y objetivo."""

    def _make(
        equipment: Equipment, goal: Goal = Goal.GAIN_MUSCLE, **overrides
    ) -> UserProfile:
        return UserProfile(
            age=30,
            sex="masculino",
            weight_kg=80,
            height_cm=178,
            goal=goal,
            experience_level=ExperienceLevel.INTERMEDIATE,
            days_per_week=3,
            session_duration_min=60,
            equipment=equipment,
            **overrides,
        )

    return _make
