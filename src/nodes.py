import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.prompts.system_prompt import SYSTEM_PROMPT
from src.retrieval.exercise_catalog import format_candidates, retrieve_exercises
from src.schemas.output_schemas import DietPlan, UserProfile, WorkoutRoutine
from src.schemas.safety_schema import SafetyClassification
from src.state import GraphState


def _build_llm():
    """Proveedor configurable por LLM_PROVIDER (anthropic | openai).
    Verifica los nombres de modelo vigentes en la documentación de cada
    proveedor antes de desplegar; los strings de modelo cambian con el tiempo."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"), temperature=0.3
        )
    raise ValueError(f"LLM_PROVIDER no soportado: {provider}")


llm = _build_llm()

REQUIRED_FIELDS = [
    "age",
    "sex",
    "weight_kg",
    "height_cm",
    "goal",
    "experience_level",
    "days_per_week",
    "session_duration_min",
    "equipment",
]


def collect_profile_node(state: GraphState) -> GraphState:
    """Extrae (o actualiza) el UserProfile a partir de toda la conversación."""
    extractor = llm.with_structured_output(UserProfile)
    try:
        profile = extractor.invoke(
            [
                SystemMessage(
                    content=(
                        "Extrae el perfil del usuario a partir de la conversación "
                        "completa. Si un dato no fue mencionado explícitamente, "
                        "déjalo en None. No lo inventes ni lo asumas."
                    )
                )
            ]
            + state["messages"]
        )
    except Exception:
        profile = None

    missing = []
    if profile:
        for field in REQUIRED_FIELDS:
            if getattr(profile, field, None) in (None, ""):
                missing.append(field)
    else:
        missing = REQUIRED_FIELDS.copy()

    # Una vez que ya se generó rutina/dieta ("done"), esta función sigue
    # corriendo cada turno para mantener el perfil actualizado, pero no debe
    # pisar el stage: eso pasa a ser responsabilidad de process_feedback_node.
    if state.get("stage") == "done":
        new_stage = state["stage"]
    else:
        new_stage = "ready_to_generate" if not missing else "collecting"

    return {
        **state,
        "user_profile": profile,
        "missing_fields": missing,
        "stage": new_stage,
    }


def safety_guardrail_node(state: GraphState) -> GraphState:
    """Chequeo estructural de seguridad, aparte del SYSTEM_PROMPT: clasifica el
    último turno del usuario buscando lesión/condición médica/embarazo y, si
    aparece, corta el flujo hacia una derivación en vez de confiar en que el
    LLM generador respete la regla de forma discursiva en el 100% de los casos."""
    classifier = llm.with_structured_output(SafetyClassification)
    last_user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    if not last_user_message:
        return {**state, "safety_flag": False}

    classification = classifier.invoke(
        [
            SystemMessage(
                content=(
                    "Analiza este mensaje de un usuario de un bot de "
                    "entrenamiento y nutrición. Indica si menciona una lesión, "
                    "una condición médica, o un embarazo, propios y actuales. "
                    "No marques menciones vagas, hipotéticas o ya resueltas."
                )
            ),
            HumanMessage(content=last_user_message),
        ]
    )
    triggered = (
        classification.mentions_injury
        or classification.mentions_medical_condition
        or classification.mentions_pregnancy
    )
    if not triggered:
        return {**state, "safety_flag": False}

    deferral = (
        "Gracias por contarme. No soy un profesional de la salud, así que "
        "antes de seguir con la rutina o la dieta te recomiendo consultar a "
        f"un médico o fisioterapeuta sobre esto: {classification.details or 'lo que mencionaste'}. "
        "Cuando tengas el visto bueno, seguimos con gusto."
    )
    return {
        **state,
        "safety_flag": True,
        "messages": state["messages"] + [AIMessage(content=deferral)],
    }


def ask_missing_node(state: GraphState) -> GraphState:
    """Genera una pregunta natural pidiendo únicamente los datos que faltan."""
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)]
        + state["messages"]
        + [
            HumanMessage(
                content=(
                    "Faltan estos datos del perfil: "
                    f"{', '.join(state['missing_fields'])}. "
                    "Pídelos en un solo mensaje breve y conversacional, "
                    "sin sonar a formulario."
                )
            )
        ]
    )
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


def generate_routine_node(state: GraphState) -> GraphState:
    """Genera la rutina anclada en el catálogo real: los candidatos ya vienen
    filtrados por el equipo del usuario y rankeados hacia su objetivo, y el
    prompt prohíbe inventar ejercicios fuera de esa lista."""
    candidates = retrieve_exercises(state["user_profile"])
    generator = llm.with_structured_output(WorkoutRoutine)
    routine = generator.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Genera una rutina de ejercicios para este perfil "
                    f"(JSON): {state['user_profile'].model_dump_json()}\n\n"
                    "CATÁLOGO DE EJERCICIOS PERMITIDOS (ya filtrado según el "
                    "equipo declarado por el usuario y priorizado según su "
                    "objetivo):\n"
                    f"{format_candidates(candidates)}\n\n"
                    "Usa ÚNICAMENTE ejercicios de este catálogo, con sus "
                    "nombres exactos. No inventes ejercicios fuera de la "
                    "lista; solo se permiten variantes triviales de ejecución "
                    "(por ejemplo cambiar agarre, tempo o apoyo) indicadas en "
                    "las notas del ejercicio."
                )
            ),
        ]
    )
    return {**state, "routine": routine}


def generate_diet_node(state: GraphState) -> GraphState:
    generator = llm.with_structured_output(DietPlan)
    diet = generator.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Genera un plan de alimentación para este perfil "
                    f"(JSON): {state['user_profile'].model_dump_json()}"
                )
            ),
        ]
    )
    return {**state, "diet": diet, "stage": "done"}


def format_output_node(state: GraphState) -> GraphState:
    """Convierte routine + diet (JSON) en un mensaje legible y motivador."""
    summary = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Presenta de forma clara, ordenada y motivadora esta "
                    "rutina y dieta al usuario, en texto legible (no JSON):\n"
                    f"Rutina: {state['routine'].model_dump_json()}\n"
                    f"Dieta: {state['diet'].model_dump_json()}"
                )
            ),
        ]
    )
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=summary.content)],
    }
