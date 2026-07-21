"""
Esquemas de datos (Pydantic) para el perfil de usuario y las salidas
estructuradas del modelo. Usar with_structured_output(<Modelo>) en
LangChain fuerza al LLM a devolver JSON válido contra estos esquemas,
en vez de texto libre que hay que parsear a mano.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel
from sqlmodel import Field, SQLModel


# ---------- Perfil de usuario ----------

class Goal(str, Enum):
    LOSE_FAT = "perder_grasa"
    GAIN_MUSCLE = "ganar_musculo"
    MAINTAIN = "mantener"
    ENDURANCE = "resistencia"


class ExperienceLevel(str, Enum):
    BEGINNER = "principiante"
    INTERMEDIATE = "intermedio"
    ADVANCED = "avanzado"


class Equipment(str, Enum):
    FULL_GYM = "gimnasio_completo"
    HOME_DUMBBELLS = "casa_mancuernas"
    BODYWEIGHT = "sin_equipo"


class UserProfileBase(SQLModel):
    """Campos compartidos entre el UserProfile que ve el LLM y su
    representación persistida (UserProfileRecord en src/db/models.py)."""

    age: Optional[int] = Field(default=None, ge=12, le=90)
    sex: Optional[str] = None
    weight_kg: Optional[float] = Field(default=None, gt=0)
    height_cm: Optional[float] = Field(default=None, gt=0)
    goal: Optional[Goal] = None
    experience_level: Optional[ExperienceLevel] = None
    days_per_week: Optional[int] = Field(default=None, ge=1, le=7)
    session_duration_min: Optional[int] = Field(default=None, ge=15, le=180)
    equipment: Optional[Equipment] = None
    injuries_or_limitations: Optional[str] = None
    dietary_restrictions: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)


class UserProfile(UserProfileBase):
    """Esquema que usa with_structured_output(UserProfile) en collect_profile_node."""


# ---------- Rutina ----------

class Exercise(BaseModel):
    name: str
    sets: int
    reps: str  # ej. "8-12" o "AMRAP"
    rest_seconds: int
    notes: Optional[str] = None


class WorkoutDay(BaseModel):
    day_name: str  # ej. "Día 1"
    focus: str  # ej. "Empuje", "Piernas", "Full body"
    exercises: List[Exercise]


class WorkoutRoutine(BaseModel):
    days: List[WorkoutDay]
    general_notes: Optional[str] = None


# ---------- Dieta ----------

class Meal(BaseModel):
    name: str  # ej. "Desayuno"
    foods: List[str]
    approx_calories: int
    protein_g: float
    carbs_g: float
    fat_g: float


class DietPlan(BaseModel):
    daily_calories: int
    protein_g: float
    carbs_g: float
    fat_g: float
    meals: List[Meal]
    general_notes: Optional[str] = None
