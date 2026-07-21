import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.prompts.system_prompt import SYSTEM_PROMPT
from src.retrieval.exercise_catalog import format_candidates, retrieve_exercises
from src.retrieval.food_catalog import format_food_candidates, retrieve_foods
from src.schemas.output_schemas import DietPlan, UserProfile, WorkoutRoutine
from src.schemas.safety_schema import FeedbackClassification, SafetyClassification
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


def process_feedback_node(state: GraphState) -> GraphState:
    """Con un plan ya activo (stage 'done'), clasifica el feedback del usuario y
    decide si ajustar rutina+dieta o responder de forma conversacional. El
    guardrail de seguridad ya corrió antes en el grafo, así que aquí nunca llega
    una lesión/condición/embarazo sin derivar primero (esa regla tiene
    prioridad sobre esta lógica de progresión)."""
    classifier = llm.with_structured_output(FeedbackClassification)
    last_user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    if not last_user_message:
        return {**state, "feedback_classification": None}

    classification = classifier.invoke(
        [
            SystemMessage(
                content=(
                    "El usuario ya tiene una rutina y una dieta activas. "
                    "Clasifica su último mensaje. Marca wants_regeneration=True "
                    "solo si el feedback implica ajustar el plan: lo siente muy "
                    "fácil o muy difícil, o cambió su logística (días, equipo, "
                    "horario, duración). Un comentario casual, una duda puntual "
                    "o que esté conforme es wants_regeneration=False."
                )
            ),
            HumanMessage(content=last_user_message),
        ]
    )

    # Semana ISO codificada como AAAASS (ver current_week_number): al ser
    # monótona (semana <= 53 < 100), un '>' basta para saber si ya pasó una
    # semana completa desde que se generó el plan activo.
    week_elapsed = (
        state.get("active_plan_week") is not None
        and state["week_number"] > state["active_plan_week"]
    )
    feedback_notes = classification.summary or last_user_message

    if classification.wants_regeneration:
        # Deja el feedback como contexto del ajuste (generate_routine/diet lo
        # incorporan cuando hay previous_routine/previous_diet) y ruteamos a
        # regenerar; el mensaje al usuario lo arma después format_output.
        return {
            **state,
            "feedback_notes": feedback_notes,
            "feedback_classification": classification.model_dump(),
        }

    # No pide ajuste: respuesta conversacional. Si ya pasó una semana ISO
    # completa desde el plan activo, la aprovechamos para ofrecer un check-in.
    if week_elapsed:
        instruction = (
            "El usuario no pidió cambios, pero ya pasó una semana completa "
            "desde su plan actual. Respóndele de forma breve y natural y, de "
            "paso, ofrécele hacer un check-in para ajustar su rutina y dieta "
            "si quiere seguir progresando (ofrécelo, no lo obligues)."
        )
    else:
        instruction = (
            "El usuario no pidió cambios en su plan. Respóndele de forma "
            "breve, natural y motivadora, sin regenerar la rutina ni la dieta."
        )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)]
        + state["messages"]
        + [HumanMessage(content=instruction)]
    )
    return {
        **state,
        "feedback_notes": feedback_notes,
        "feedback_classification": classification.model_dump(),
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


def _adjustment_block(previous, feedback, plan_noun: str, adjust_hint: str) -> str:
    """Bloque de prompt que convierte la generación en un AJUSTE del plan previo
    (rutina o dieta) según el feedback. Vacío en la primera generación (sin
    plan previo ni feedback), para no alterar ese flujo. plan_noun nombra el
    plan ('plan de entrenamiento' / 'plan de alimentación') y adjust_hint da la
    pista de ajuste específica de cada uno."""
    if not (previous and feedback):
        return ""
    return (
        f"AJUSTE POR FEEDBACK: el usuario ya venía con este {plan_noun} "
        f"(JSON):\n{previous.model_dump_json()}\n"
        f'Y dio este feedback: "{feedback}".\n'
        f"Genera una versión AJUSTADA de ese {plan_noun} que responda al "
        "feedback (no uno nuevo sin relación): conserva lo que funcionaba y "
        f"{adjust_hint} Sigue anclado únicamente al catálogo permitido de "
        "abajo.\n\n"
    )


def generate_routine_node(state: GraphState) -> GraphState:
    """Genera la rutina anclada en el catálogo real: los candidatos ya vienen
    filtrados por el equipo del usuario y rankeados hacia su objetivo, y el
    prompt prohíbe inventar ejercicios fuera de esa lista."""
    # 20 candidatos: con el minimo de 5 ejercicios por dia, 15 se quedaban
    # cortos de variedad por focus (sobre todo en el tier sin_equipo).
    candidates = retrieve_exercises(state["user_profile"], num_results=20)
    adjustment = _adjustment_block(
        state.get("previous_routine"),
        state.get("feedback_notes"),
        "plan de entrenamiento",
        "cambia lo que el feedback indica: sube o baja intensidad/volumen, "
        "reemplaza ejercicios problemáticos y adapta a la nueva logística.",
    )
    generator = llm.with_structured_output(WorkoutRoutine)
    routine = generator.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Genera una rutina de ejercicios para este perfil "
                    f"(JSON): {state['user_profile'].model_dump_json()}\n\n"
                    f"{adjustment}"
                    "CATÁLOGO DE EJERCICIOS PERMITIDOS (ya filtrado según el "
                    "equipo declarado por el usuario y priorizado según su "
                    "objetivo):\n"
                    f"{format_candidates(candidates)}\n\n"
                    "Usa ÚNICAMENTE ejercicios de este catálogo, con sus "
                    "nombres exactos. No inventes ejercicios fuera de la "
                    "lista; solo se permiten variantes triviales de ejecución "
                    "(por ejemplo cambiar agarre, tempo o apoyo) indicadas en "
                    "las notas del ejercicio.\n\n"
                    "Reglas de composición de los días:\n"
                    "- Cada día declara un focus, y cada ejercicio de ese día "
                    "debe ser coherente con ese focus según el campo 'foco' "
                    "del catálogo (ej.: en un día de Piernas no va un remo ni "
                    "un press de pecho).\n"
                    "- Excepción: los ejercicios de foco Core o Cardio pueden "
                    "complementar cualquier día.\n"
                    "- Si un ejercicio no encaja con el focus de ningún día, "
                    "simplemente no lo uses.\n"
                    "- Cada día debe incluir MÍNIMO 5 ejercicios. Puedes "
                    "repetir un ejercicio del catálogo en más de un día si "
                    "hace falta para completarlo, y apoyarte en Core/Cardio "
                    "como complemento."
                )
            ),
        ]
    )
    return {**state, "routine": routine}


def generate_diet_node(state: GraphState) -> GraphState:
    """Genera la dieta anclada en el catálogo USDA real: los candidatos ya
    vienen depurados de alergias/restricciones del usuario y priorizados hacia
    su objetivo, y el prompt prohíbe inventar alimentos o valores fuera de esa
    lista."""
    candidates = retrieve_foods(state["user_profile"], num_results=25)
    adjustment = _adjustment_block(
        state.get("previous_diet"),
        state.get("feedback_notes"),
        "plan de alimentación",
        "ajusta lo que el feedback indica.",
    )
    generator = llm.with_structured_output(DietPlan)
    diet = generator.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Genera un plan de alimentación para este perfil "
                    f"(JSON): {state['user_profile'].model_dump_json()}\n\n"
                    f"{adjustment}"
                    "CATÁLOGO DE ALIMENTOS PERMITIDOS (subconjunto de USDA "
                    "FoodData Central, ya depurado según las alergias y "
                    "restricciones declaradas por el usuario y priorizado "
                    "según su objetivo):\n"
                    f"{format_food_candidates(candidates)}\n\n"
                    "Usa ÚNICAMENTE alimentos de este catálogo, con sus "
                    "nombres exactos. No inventes alimentos ni valores "
                    "nutricionales fuera de la lista: las calorías y macros "
                    "de cada comida deben salir de los valores por 100 g del "
                    "catálogo, escalados a la porción que indiques en gramos.\n\n"
                    "Reglas de composición del plan:\n"
                    "- Indica la porción en gramos de cada alimento dentro de "
                    "cada comida (ej.: 'Arroz blanco cocido (150 g)').\n"
                    "- Los totales diarios deben ser coherentes con la suma "
                    "de las comidas.\n"
                    "- No incluyas ningún alimento vetado por las alergias o "
                    "restricciones del perfil, ni siquiera como sugerencia "
                    "opcional."
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
