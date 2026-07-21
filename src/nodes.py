from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.prompts.system_prompt import SYSTEM_PROMPT
from src.schemas.output_schemas import DietPlan, UserProfile, WorkoutRoutine
from src.schemas.safety_schema import SafetyClassification
from src.state import GraphState

# Verifica el nombre de modelo vigente en la documentación de Anthropic
# antes de desplegar; los strings de modelo cambian con el tiempo.
llm = ChatAnthropic(model="claude-sonnet-5", temperature=0.3)

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
    generator = llm.with_structured_output(WorkoutRoutine)
    routine = generator.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Genera una rutina de ejercicios para este perfil "
                    f"(JSON): {state['user_profile'].model_dump_json()}"
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
