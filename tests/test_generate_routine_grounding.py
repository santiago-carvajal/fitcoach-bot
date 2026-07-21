"""
Test del seam de grafo para generate_routine_node: LLM mockeado, se
verifica que el prompt de generación queda anclado en los candidatos
recuperados del catálogo (y no en ejercicios incompatibles con el equipo).
"""

from types import SimpleNamespace

from src.nodes import generate_routine_node
from src.retrieval.exercise_catalog import retrieve_exercises
from src.schemas.output_schemas import Equipment, WorkoutRoutine


def test_routine_prompt_is_grounded_in_retrieved_candidates(
    monkeypatch, make_profile, make_fake_llm, make_state
):
    profile = make_profile(Equipment.BODYWEIGHT)
    canned = WorkoutRoutine(days=[], general_notes="ok")
    fake = make_fake_llm(canned)
    monkeypatch.setattr(
        "src.nodes.llm",
        SimpleNamespace(with_structured_output=lambda schema: fake),
    )

    result = generate_routine_node(make_state(profile))

    assert result["routine"] is canned
    prompt_text = "\n".join(str(m.content) for m in fake.received_messages)

    candidates = retrieve_exercises(profile)
    for candidate in candidates[:5]:
        assert candidate["name"] in prompt_text, (
            f"el candidato '{candidate['name']}' debe llegar al prompt"
        )
    # Un usuario sin equipo jamás debe ver ejercicios de gimnasio en el prompt.
    assert "Sentadilla con barra" not in prompt_text
    assert "Press de banca con barra" not in prompt_text
