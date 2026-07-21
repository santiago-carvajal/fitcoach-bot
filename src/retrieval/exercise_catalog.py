"""
Catálogo de ejercicios + retrieval TF-IDF en memoria (minsearch).

El dataset (data/exercises.csv) es original de este proyecto; solo se reusa
el patrón CSV + minsearch del proyecto de referencia, no sus datos.

Seam público: retrieve_exercises(profile) — perfil entra, candidatos salen.
Los nodos del grafo consumen esto para anclar la generación de rutinas en
ejercicios reales compatibles con el equipo declarado por el usuario.
"""

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from minsearch import Index

from src.schemas.output_schemas import Equipment, Goal, UserProfile

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "exercises.csv"

# Jerarquía de equipo: un tier superior también puede usar ejercicios de los
# tiers inferiores (gimnasio completo incluye mancuernas y peso corporal).
ALLOWED_TIERS: Dict[Equipment, List[str]] = {
    Equipment.BODYWEIGHT: [Equipment.BODYWEIGHT.value],
    Equipment.HOME_DUMBBELLS: [
        Equipment.BODYWEIGHT.value,
        Equipment.HOME_DUMBBELLS.value,
    ],
    Equipment.FULL_GYM: [
        Equipment.BODYWEIGHT.value,
        Equipment.HOME_DUMBBELLS.value,
        Equipment.FULL_GYM.value,
    ],
}

# Términos de búsqueda por objetivo, en el vocabulario del catálogo
# (columnas focus / movement_type / muscle_groups).
GOAL_QUERIES: Dict[Goal, str] = {
    Goal.LOSE_FAT: "cardio full body sistema cardiovascular circuito intensidad",
    Goal.GAIN_MUSCLE: (
        "empuje traccion fuerza pecho espalda hombros cuadriceps gluteos"
    ),
    Goal.ENDURANCE: "cardio resistencia sistema cardiovascular ritmo",
    Goal.MAINTAIN: "full body empuje traccion core cardio",
}

TEXT_FIELDS = ["name", "focus", "movement_type", "muscle_groups", "notes"]
BOOSTS = {"focus": 3.0, "movement_type": 2.0, "muscle_groups": 1.5}


class ExerciseCatalog:
    def __init__(self, csv_path: Path = DATA_PATH):
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.exercises: List[dict] = list(csv.DictReader(f))
        self._index = Index(text_fields=TEXT_FIELDS, keyword_fields=["equipment"])
        self._index.fit(self.exercises)

    def retrieve(self, profile: UserProfile, num_results: int = 15) -> List[dict]:
        """Candidatos compatibles con el equipo del perfil, ordenados por
        afinidad al objetivo. Los ejercicios permitidos que el query TF-IDF
        no puntúa se agregan al final para no perder cobertura."""
        allowed = ALLOWED_TIERS[profile.equipment]
        query = GOAL_QUERIES.get(profile.goal, "ejercicio full body")
        ranked = self._index.search(
            query,
            filter_dict={"equipment": allowed},
            boost_dict=BOOSTS,
            num_results=len(self.exercises),
        )
        ranked_names = {doc["name"] for doc in ranked}
        backfill = [
            e
            for e in self.exercises
            if e["equipment"] in allowed and e["name"] not in ranked_names
        ]
        return (ranked + backfill)[:num_results]


def format_candidates(candidates: List[dict]) -> str:
    """Lista legible de candidatos para inyectar en el prompt de generación."""
    lines = []
    for c in candidates:
        lines.append(
            f"- {c['name']} (foco: {c['focus']}, equipo: {c['equipment']}, "
            f"series: {c['default_sets']}, reps: {c['default_reps']}, "
            f"descanso: {c['rest_seconds']}s). {c['notes']}"
        )
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _default_catalog() -> ExerciseCatalog:
    return ExerciseCatalog()


def retrieve_exercises(profile: UserProfile, num_results: int = 15) -> List[dict]:
    return _default_catalog().retrieve(profile, num_results)
