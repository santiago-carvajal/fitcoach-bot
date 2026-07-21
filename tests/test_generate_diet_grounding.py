"""
Test del seam de grafo para generate_diet_node: LLM mockeado, se
verifica que el prompt de generación queda anclado en los candidatos
recuperados del catálogo USDA (y no en alimentos vetados por las
alergias/restricciones del perfil).
"""

from types import SimpleNamespace

from src.nodes import generate_diet_node
from src.retrieval.food_catalog import retrieve_foods
from src.schemas.output_schemas import DietPlan, Equipment


def test_diet_prompt_is_grounded_in_retrieved_candidates(
    monkeypatch, make_profile, make_fake_llm, make_state
):
    profile = make_profile(
        Equipment.FULL_GYM,
        allergies=["maní"],
        dietary_restrictions=["intolerante a la lactosa"],
    )
    canned = DietPlan(
        daily_calories=2200, protein_g=160, carbs_g=220, fat_g=70, meals=[]
    )
    fake = make_fake_llm(canned)
    monkeypatch.setattr(
        "src.nodes.llm",
        SimpleNamespace(with_structured_output=lambda schema: fake),
    )

    result = generate_diet_node(make_state(profile))

    assert result["diet"] is canned
    assert result["stage"] == "done"
    prompt_text = "\n".join(str(m.content) for m in fake.received_messages)

    candidates = retrieve_foods(profile)
    for candidate in candidates[:5]:
        assert candidate["name"] in prompt_text, (
            f"el candidato '{candidate['name']}' debe llegar al prompt"
        )
    # Alimentos vetados por alergia/restricción jamás deben llegar al prompt.
    assert "Maní tostado" not in prompt_text
    assert "Mantequilla de maní" not in prompt_text
    assert "Yogur griego natural descremado" not in prompt_text
    assert "Leche descremada" not in prompt_text
    # El prompt debe prohibir inventar alimentos o valores nutricionales.
    assert "No inventes" in prompt_text
