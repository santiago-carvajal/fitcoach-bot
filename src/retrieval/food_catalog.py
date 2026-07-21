"""
Catálogo de alimentos + retrieval TF-IDF en memoria (minsearch).

El dataset (data/foods.csv) es un subconjunto curado de USDA FoodData
Central (dominio público, CC0): valores nutricionales por 100 g tomados
de las entradas genéricas (Foundation / SR Legacy) de cada alimento,
redondeados, con nombres y notas en español propios del proyecto. El CSV
no versiona el fdc_id por fila; ante una duda puntual, contrastar contra
https://fdc.nal.usda.gov/.

Seam público: retrieve_foods(profile) — perfil entra, candidatos salen.
La exclusión por alergias/restricciones declaradas ocurre acá, antes de
cualquier LLM: un alimento vetado jamás llega al prompt de generación.
"""

import csv
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set

from minsearch import Index

from src.schemas.output_schemas import Goal, UserProfile

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "foods.csv"

# Términos de búsqueda por objetivo, en el vocabulario del catálogo
# (columnas tags / category / notes).
GOAL_QUERIES: Dict[Goal, str] = {
    Goal.LOSE_FAT: "bajo en calorias saciante fibra verdura proteina magra",
    Goal.GAIN_MUSCLE: "proteina magra musculo fuerza",
    Goal.ENDURANCE: "carbohidrato complejo energia resistencia",
    Goal.MAINTAIN: "balanceado proteina carbohidrato verdura fruta",
}

TEXT_FIELDS = ["name", "category", "tags", "notes"]
BOOSTS = {"tags": 3.0, "category": 1.5}

# Palabra clave (normalizada, sin acentos) que puede aparecer en una alergia o
# restricción declarada → tags de alimento que quedan vetados. El match es por
# substring sobre el texto libre del usuario ("alergia al maní" contiene "mani").
RESTRICTION_RULES: Dict[str, Set[str]] = {
    "vegetarian": {"carne", "pescado", "marisco"},
    "vegan": {"carne", "pescado", "marisco", "lacteo", "huevo", "miel"},
    "celiaqu": {"gluten"},
    "celiac": {"gluten"},
    "gluten": {"gluten"},
    "trigo": {"trigo", "gluten"},
    "lactosa": {"lactosa"},
    "lacteo": {"lacteo"},
    "leche": {"lacteo"},
    "mani": {"mani"},
    "cacahu": {"mani"},
    "frutos secos": {"fruto_seco", "mani"},
    "fruto seco": {"fruto_seco", "mani"},
    "nuez": {"fruto_seco"},
    "nueces": {"fruto_seco"},
    "almendra": {"fruto_seco"},
    "carne": {"carne"},
    "marisco": {"marisco"},
    "crustaceo": {"marisco"},
    "camaron": {"marisco"},
    "pescado": {"pescado"},
    "huevo": {"huevo"},
    "soja": {"soja"},
    "soya": {"soja"},
    "cerdo": {"cerdo"},
}

# Tokens genéricos de los nombres que no identifican al alimento y no deben
# disparar el fallback por nombre ("cocido" aparece en media docena de filas).
_NAME_STOPWORDS = {
    "cocido", "cocida", "cocidos", "cocidas", "crudo", "cruda", "natural",
    "entero", "entera", "firme", "dulce", "rojo", "roja", "negra", "negros",
    "blanco", "blanca", "integral", "magra", "plancha", "lata", "agua",
    "grasa", "polvo", "hojuelas", "semidescremado", "descremado", "tostado",
    "bajo", "semillas", "mantequilla", "aceite", "griego",
}


def _normalize(text: str) -> str:
    """minúsculas y sin acentos, para comparar texto libre del usuario."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _excluded_tags(terms: List[str]) -> Set[str]:
    """Tags vetados según las alergias/restricciones declaradas."""
    tags: Set[str] = set()
    for term in terms:
        normalized = _normalize(term)
        for keyword, excluded in RESTRICTION_RULES.items():
            if keyword in normalized:
                tags |= excluded
    return tags


def _stems(word: str) -> Set[str]:
    """Variantes singular/plural aproximadas para comparar palabras sueltas."""
    stems = {word}
    if word.endswith("es"):
        stems.add(word[:-2])
    if word.endswith("s"):
        stems.add(word[:-1])
    return stems


def _is_excluded(food: dict, excluded_tags: Set[str], terms: List[str]) -> bool:
    food_tags = set(_normalize(food["allergen_tags"]).split())
    if food_tags & excluded_tags:
        return True
    # Fallback por nombre: cubre alergias a alimentos puntuales sin regla
    # propia ("alergia al banano" debe vetar la fila Banano), tolerando el
    # desajuste singular/plural entre lo declarado y el nombre del catálogo.
    name_tokens = {
        t
        for t in _normalize(food["name"]).split()
        if len(t) >= 4 and t not in _NAME_STOPWORDS
    }
    for term in terms:
        normalized = _normalize(term)
        term_words = {w for w in normalized.split() if len(w) >= 4}
        for token in name_tokens:
            if token in normalized:
                return True
            if any(_stems(token) & _stems(word) for word in term_words):
                return True
    return False


class FoodCatalog:
    def __init__(self, csv_path: Path = DATA_PATH):
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.foods: List[dict] = list(csv.DictReader(f))
        self._index = Index(text_fields=TEXT_FIELDS, keyword_fields=[])
        self._index.fit(self.foods)

    def retrieve(self, profile: UserProfile, num_results: int = 25) -> List[dict]:
        """Candidatos que respetan alergias/restricciones del perfil, ordenados
        por afinidad al objetivo. Los alimentos permitidos que el query TF-IDF
        no puntúa se agregan al final para no perder cobertura de grupos."""
        terms = list(profile.allergies) + list(profile.dietary_restrictions)
        excluded = _excluded_tags(terms)
        query = GOAL_QUERIES.get(profile.goal, "balanceado proteina verdura")
        ranked = self._index.search(
            query, boost_dict=BOOSTS, num_results=len(self.foods)
        )
        ranked = [f for f in ranked if not _is_excluded(f, excluded, terms)]
        ranked_names = {f["name"] for f in ranked}
        backfill = [
            f
            for f in self.foods
            if f["name"] not in ranked_names
            and not _is_excluded(f, excluded, terms)
        ]
        return (ranked + backfill)[:num_results]


def format_food_candidates(candidates: List[dict]) -> str:
    """Lista legible de candidatos (valores por 100 g) para el prompt."""
    lines = []
    for c in candidates:
        lines.append(
            f"- {c['name']} (por 100 g: {c['calories_kcal']} kcal, "
            f"proteína: {c['protein_g']} g, carbohidratos: {c['carbs_g']} g, "
            f"grasa: {c['fat_g']} g). {c['notes']}"
        )
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _default_catalog() -> FoodCatalog:
    return FoodCatalog()


def retrieve_foods(profile: UserProfile, num_results: int = 25) -> List[dict]:
    return _default_catalog().retrieve(profile, num_results)
