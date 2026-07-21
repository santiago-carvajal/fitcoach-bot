"""
Test del seam de grafo para generate_routine_node: LLM mockeado, se
verifica que el prompt de generación queda anclado en los candidatos
recuperados del catálogo (y no en ejercicios incompatibles con el equipo).
"""

from types import SimpleNamespace

from src.nodes import generate_routine_node
from src.retrieval.exercise_catalog import retrieve_exercises
from src.schemas.output_schemas import Equipment, WorkoutRoutine


class FakeStructuredLLM:
    def __init__(self, response):
        self.response = response
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return self.response


def test_routine_prompt_is_grounded_in_retrieved_candidates(monkeypatch, make_profile):
    profile = make_profile(Equipment.BODYWEIGHT)
    canned = WorkoutRoutine(days=[], general_notes="ok")
    fake = FakeStructuredLLM(canned)
    monkeypatch.setattr(
        "src.nodes.llm",
        SimpleNamespace(with_structured_output=lambda schema: fake),
    )

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
    }

    result = generate_routine_node(state)

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
