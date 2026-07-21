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
