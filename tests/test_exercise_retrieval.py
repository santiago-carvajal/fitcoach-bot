"""
Tests del seam de retrieval de ejercicios: perfil entra, candidatos salen.
Sin LLM, sin grafo, sin HTTP — solo el catálogo y minsearch.
"""

from src.retrieval.exercise_catalog import retrieve_exercises
from src.schemas.output_schemas import Equipment, Goal


def test_bodyweight_user_only_gets_bodyweight_exercises(make_profile):
    candidates = retrieve_exercises(make_profile(Equipment.BODYWEIGHT))

    assert candidates, "un usuario sin equipo debe recibir candidatos"
    assert all(c["equipment"] == "sin_equipo" for c in candidates)


def test_home_dumbbell_user_never_gets_gym_exercises(make_profile):
    candidates = retrieve_exercises(make_profile(Equipment.HOME_DUMBBELLS))

    assert candidates
    equipments = {c["equipment"] for c in candidates}
    assert equipments <= {"sin_equipo", "casa_mancuernas"}
    assert "casa_mancuernas" in equipments, (
        "el tier propio del usuario debe estar representado"
    )


def test_full_gym_user_can_get_any_tier_including_gym_only(make_profile):
    candidates = retrieve_exercises(make_profile(Equipment.FULL_GYM), num_results=50)

    equipments = {c["equipment"] for c in candidates}
    assert "gimnasio_completo" in equipments
    assert equipments <= {"sin_equipo", "casa_mancuernas", "gimnasio_completo"}


def test_goal_changes_ranking_toward_goal_relevant_exercises(make_profile):
    fat_loss = retrieve_exercises(
        make_profile(Equipment.BODYWEIGHT, goal=Goal.LOSE_FAT), num_results=5
    )
    muscle = retrieve_exercises(
        make_profile(Equipment.BODYWEIGHT, goal=Goal.GAIN_MUSCLE), num_results=5
    )

    assert [c["name"] for c in fat_loss] != [c["name"] for c in muscle], (
        "objetivos distintos deben producir rankings distintos"
    )
    assert any(c["focus"].lower() in ("cardio", "full body") for c in fat_loss[:3]), (
        "perder grasa debe priorizar ejercicios de cardio / full body"
    )
    assert any(c["movement_type"] in ("empuje", "traccion") for c in muscle[:3]), (
        "ganar musculo debe priorizar trabajo de fuerza (empuje/traccion)"
    )


def test_candidates_are_catalog_rows_and_respect_num_results(make_profile):
    candidates = retrieve_exercises(make_profile(Equipment.FULL_GYM), num_results=8)

    assert len(candidates) == 8
    for c in candidates:
        assert {"name", "equipment", "focus", "default_sets", "default_reps"} <= set(c)
