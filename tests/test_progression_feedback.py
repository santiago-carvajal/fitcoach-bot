"""
Tests del seam de grafo para la fase de progresión (issue #4): con un plan
ya activo (stage 'done'), se invoca el grafo COMPLETO con el LLM mockeado y
se verifica el ruteo por feedback.

Casos cubiertos:
- Feedback que pide ajuste  -> regenera rutina y dieta (adjunta plan previo +
  feedback al prompt de generación).
- Feedback casual/conforme  -> respuesta conversacional, sin regeneración.
- Mención de lesión         -> el guardrail de seguridad deriva ANTES de que
  la clasificación de feedback corra (la seguridad tiene prioridad).
- Semana ISO transcurrida   -> el bot ofrece proactivamente un check-in.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.graph import build_graph
from src.schemas.output_schemas import DietPlan, WorkoutRoutine
from src.schemas.safety_schema import FeedbackClassification, SafetyClassification

# El LLM falso que enruta por esquema (FakeGraphLLM) vive en conftest y se
# obtiene con la fixture make_graph_llm, compartido con los tests de la API.


def _install_fake(monkeypatch, fake):
    monkeypatch.setattr("src.nodes.llm", fake)


def _active_plan_state(make_state, profile, user_text, **overrides):
    """Estado con un plan ya activo (stage 'done') y un mensaje de usuario."""
    prev_routine = WorkoutRoutine(days=[], general_notes="rutina previa")
    prev_diet = DietPlan(
        daily_calories=2000, protein_g=150, carbs_g=200, fat_g=60, meals=[]
    )
    defaults = {
        "stage": "done",
        "messages": [HumanMessage(content=user_text)],
        "routine": prev_routine,
        "diet": prev_diet,
        "previous_routine": prev_routine,
        "previous_diet": prev_diet,
        "week_number": 202630,
        "active_plan_week": 202630,
    }
    defaults.update(overrides)
    return make_state(profile, **defaults)


def test_feedback_pidiendo_ajuste_regenera_rutina_y_dieta(
    monkeypatch, make_profile, make_state, make_graph_llm
):
    from src.schemas.output_schemas import Equipment

    profile = make_profile(Equipment.FULL_GYM)
    new_routine = WorkoutRoutine(days=[], general_notes="rutina ajustada")
    new_diet = DietPlan(
        daily_calories=2300, protein_g=170, carbs_g=210, fat_g=65, meals=[]
    )
    fake = make_graph_llm(
        responses={
            type(profile): profile,
            SafetyClassification: SafetyClassification(
                mentions_injury=False,
                mentions_medical_condition=False,
                mentions_pregnancy=False,
            ),
            FeedbackClassification: FeedbackClassification(
                sentiment="too_hard",
                wants_regeneration=True,
                summary="la rutina está demasiado difícil",
            ),
            WorkoutRoutine: new_routine,
            DietPlan: new_diet,
        },
        plain_response=AIMessage(content="Aquí está tu plan ajustado."),
    )
    _install_fake(monkeypatch, fake)

    state = _active_plan_state(
        make_state, profile, "está muy difícil, no logro terminar"
    )
    result = build_graph().invoke(state)

    # Se regeneraron ambos planes (identidad distinta a la previa).
    assert result["routine"] is new_routine
    assert result["diet"] is new_diet
    assert result["stage"] == "done"
    # La clasificación queda disponible para persistir el FeedbackRecord.
    assert result["feedback_classification"]["wants_regeneration"] is True

    # El prompt de regeneración incluye el plan previo y el feedback: es un
    # AJUSTE, no una rutina/dieta nueva sin relación.
    routine_prompt = fake.prompt_for(WorkoutRoutine)
    assert "AJUSTE POR FEEDBACK" in routine_prompt
    assert "rutina previa" in routine_prompt
    assert "demasiado difícil" in routine_prompt
    diet_prompt = fake.prompt_for(DietPlan)
    assert "AJUSTE POR FEEDBACK" in diet_prompt


def test_feedback_casual_no_regenera_y_responde_conversacional(
    monkeypatch, make_profile, make_state, make_graph_llm
):
    from src.schemas.output_schemas import Equipment

    profile = make_profile(Equipment.FULL_GYM)
    fake = make_graph_llm(
        responses={
            type(profile): profile,
            SafetyClassification: SafetyClassification(
                mentions_injury=False,
                mentions_medical_condition=False,
                mentions_pregnancy=False,
            ),
            FeedbackClassification: FeedbackClassification(
                sentiment="satisfied",
                wants_regeneration=False,
                summary="está contento con el plan",
            ),
        },
        plain_response=AIMessage(content="¡Me alegra que te vaya bien! Sigue así."),
    )
    _install_fake(monkeypatch, fake)

    state = _active_plan_state(make_state, profile, "todo genial, gracias!")
    prev_routine = state["routine"]
    prev_diet = state["diet"]
    result = build_graph().invoke(state)

    # No hubo regeneración: la rutina/dieta siguen siendo las mismas.
    assert result["routine"] is prev_routine
    assert result["diet"] is prev_diet
    assert result["stage"] == "done"
    assert result["feedback_classification"]["wants_regeneration"] is False
    # El último mensaje es la respuesta conversacional, no un plan.
    assert result["messages"][-1].content == "¡Me alegra que te vaya bien! Sigue así."
    # Nunca se pidió generar WorkoutRoutine/DietPlan en este turno.
    called = [schema for schema, _ in fake.structured_calls]
    assert WorkoutRoutine not in called
    assert DietPlan not in called


def test_lesion_dispara_guardrail_antes_de_clasificar_feedback(
    monkeypatch, make_profile, make_state, make_graph_llm
):
    from src.schemas.output_schemas import Equipment

    profile = make_profile(Equipment.FULL_GYM)
    fake = make_graph_llm(
        responses={
            type(profile): profile,
            SafetyClassification: SafetyClassification(
                mentions_injury=True,
                mentions_medical_condition=False,
                mentions_pregnancy=False,
                details="dolor en la rodilla",
            ),
        },
        plain_response=AIMessage(content="no debería usarse"),
    )
    _install_fake(monkeypatch, fake)

    state = _active_plan_state(
        make_state, profile, "me lastimé la rodilla entrenando"
    )
    prev_routine = state["routine"]
    prev_diet = state["diet"]
    result = build_graph().invoke(state)

    # La seguridad tiene prioridad: se derivó y NO se clasificó el feedback
    # ni se regeneró nada.
    assert result["safety_flag"] is True
    assert result["feedback_classification"] is None
    assert result["routine"] is prev_routine
    assert result["diet"] is prev_diet
    called = [schema for schema, _ in fake.structured_calls]
    assert FeedbackClassification not in called
    assert WorkoutRoutine not in called
    # El último mensaje es la derivación del guardrail.
    assert "médico o fisioterapeuta" in result["messages"][-1].content


def test_semana_transcurrida_ofrece_check_in_proactivo(
    monkeypatch, make_profile, make_state, make_graph_llm
):
    from src.schemas.output_schemas import Equipment

    profile = make_profile(Equipment.FULL_GYM)
    fake = make_graph_llm(
        responses={
            type(profile): profile,
            SafetyClassification: SafetyClassification(
                mentions_injury=False,
                mentions_medical_condition=False,
                mentions_pregnancy=False,
            ),
            FeedbackClassification: FeedbackClassification(
                sentiment="other",
                wants_regeneration=False,
                summary="saluda",
            ),
        },
        plain_response=AIMessage(content="¡Hola! ¿Hacemos un check-in de tu plan?"),
    )
    _install_fake(monkeypatch, fake)

    # active_plan_week (202630) < week_number (202631): pasó una semana ISO.
    state = _active_plan_state(
        make_state, profile, "hola",
        week_number=202631, active_plan_week=202630,
    )
    result = build_graph().invoke(state)

    assert result["routine"] is state["routine"]  # sin regeneración
    # La instrucción de check-in llegó al prompt conversacional.
    conversational_prompt = "\n".join(
        str(m.content) for m in fake.plain_calls[-1]
    )
    assert "check-in" in conversational_prompt
    assert "semana completa" in conversational_prompt
