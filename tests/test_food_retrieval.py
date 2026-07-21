"""
Tests del seam de retrieval de alimentos: perfil entra, candidatos salen.
Sin LLM, sin grafo, sin HTTP — solo el catálogo USDA y minsearch.
"""

from src.retrieval.food_catalog import retrieve_foods
from src.schemas.output_schemas import Equipment, Goal


def test_peanut_allergy_excludes_peanut_foods(make_profile):
    profile = make_profile(Equipment.FULL_GYM, allergies=["maní"])

    candidates = retrieve_foods(profile, num_results=100)

    assert candidates, "un perfil con alergias igual debe recibir candidatos"
    names = [c["name"].lower() for c in candidates]
    assert not any("maní" in n or "mani" in n for n in names), (
        "un alérgico al maní jamás debe ver maní entre los candidatos"
    )


def test_lactose_intolerance_excludes_dairy(make_profile):
    profile = make_profile(
        Equipment.FULL_GYM, dietary_restrictions=["intolerante a la lactosa"]
    )

    candidates = retrieve_foods(profile, num_results=100)

    assert candidates
    for c in candidates:
        assert "lacteo" not in c["allergen_tags"], (
            f"'{c['name']}' es lácteo y no debe aparecer con intolerancia a la lactosa"
        )


def test_vegetarian_excludes_meat_fish_and_seafood(make_profile):
    profile = make_profile(Equipment.FULL_GYM, dietary_restrictions=["vegetariano"])

    candidates = retrieve_foods(profile, num_results=100)

    assert candidates
    banned = {"carne", "pescado", "marisco"}
    for c in candidates:
        assert not (set(c["allergen_tags"].split()) & banned), (
            f"'{c['name']}' no es vegetariano"
        )
    # Huevo y lácteos siguen disponibles para un vegetariano.
    names = " ".join(c["name"] for c in candidates)
    assert "Huevo" in names
    assert "Yogur" in names


def test_vegan_excludes_all_animal_products(make_profile):
    profile = make_profile(Equipment.FULL_GYM, dietary_restrictions=["vegana"])

    candidates = retrieve_foods(profile, num_results=100)

    assert candidates
    banned = {"carne", "pescado", "marisco", "lacteo", "huevo", "miel"}
    for c in candidates:
        assert not (set(c["allergen_tags"].split()) & banned), (
            f"'{c['name']}' no es vegano"
        )
    # El tofu (proteína vegetal) debe seguir disponible.
    assert any("Tofu" in c["name"] for c in candidates)


def test_free_text_allergy_matches_food_by_name(make_profile):
    profile = make_profile(Equipment.FULL_GYM, allergies=["alergia al banano"])

    candidates = retrieve_foods(profile, num_results=100)

    assert candidates
    assert not any("banano" in c["name"].lower() for c in candidates), (
        "una alergia sin regla propia debe vetar el alimento por nombre"
    )


def test_singular_allergy_excludes_plural_named_food(make_profile):
    profile = make_profile(Equipment.FULL_GYM, allergies=["alergia a la fresa"])

    candidates = retrieve_foods(profile, num_results=100)

    assert candidates
    assert not any("fresa" in c["name"].lower() for c in candidates), (
        "una alergia en singular debe vetar el alimento aunque el nombre esté en plural"
    )


def test_generic_no_meat_restriction_excludes_all_meats(make_profile):
    profile = make_profile(Equipment.FULL_GYM, dietary_restrictions=["sin carne"])

    candidates = retrieve_foods(profile, num_results=100)

    assert candidates
    names = " ".join(c["name"] for c in candidates)
    assert "Pechuga de pollo" not in names
    assert "Carne magra de res" not in names
    assert "Lomo de cerdo" not in names, (
        "'sin carne' debe vetar todas las carnes, no solo las que dicen 'carne' en el nombre"
    )


def test_goal_changes_ranking_toward_goal_relevant_foods(make_profile):
    muscle = retrieve_foods(
        make_profile(Equipment.FULL_GYM, goal=Goal.GAIN_MUSCLE), num_results=8
    )
    fat_loss = retrieve_foods(
        make_profile(Equipment.FULL_GYM, goal=Goal.LOSE_FAT), num_results=8
    )

    assert [c["name"] for c in muscle] != [c["name"] for c in fat_loss], (
        "objetivos distintos deben producir rankings distintos"
    )
    assert sum("proteina" in c["tags"] for c in muscle[:5]) >= 3, (
        "ganar músculo debe priorizar alimentos proteicos"
    )
    assert sum("bajo en calorias" in c["tags"] for c in fat_loss[:5]) >= 3, (
        "perder grasa debe priorizar alimentos saciantes bajos en calorías"
    )


def test_candidates_are_catalog_rows_and_respect_num_results(make_profile):
    candidates = retrieve_foods(make_profile(Equipment.FULL_GYM), num_results=10)

    assert len(candidates) == 10
    for c in candidates:
        assert {
            "name",
            "category",
            "allergen_tags",
            "calories_kcal",
            "protein_g",
            "carbs_g",
            "fat_g",
        } <= set(c)
