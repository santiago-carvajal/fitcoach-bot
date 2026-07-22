"""
Capa de orquestación compartida entre los entrypoints (terminal en main.py y
API en src/api.py): avanza un turno del grafo y persiste el resultado.

Vive aparte para que ambos entrypoints reusen exactamente la misma lógica de
invoke + persistencia en vez de duplicarla. Como el resto de capas que invocan
al grafo, esta sí toca la DB (via repository); los nodos siguen siendo puros.
"""

from langchain_core.messages import HumanMessage

from src.db import repository
from src.state import GraphState


def advance_turn(graph, state: GraphState, user_input: str) -> GraphState:
    """Corre un turno completo: agrega el mensaje del usuario, invoca el grafo,
    y persiste lo que haya cambiado (mensajes, perfil, feedback, rutina/dieta).
    Devuelve el nuevo estado ya con el bookkeeping en memoria actualizado, para
    que el terminal lo encadene turno a turno sin recargar de la DB. `graph` es
    el grafo compilado por build_graph()."""
    user_message = HumanMessage(content=user_input)
    repository.append_message(user_message)
    state["messages"].append(user_message)

    had_active_plan = state["stage"] == "done"
    previous_routine = state["routine"]
    previous_diet = state["diet"]

    previous_message_count = len(state["messages"])
    state = graph.invoke(state)
    for message in state["messages"][previous_message_count:]:
        repository.append_message(message)

    if state.get("user_profile") is not None:
        repository.save_profile(state["user_profile"])

    # Con un plan ya activo, process_feedback_node clasificó este turno (salvo
    # que el guardrail de seguridad derivara). Persistimos el feedback ligado
    # al plan que lo motivó, antes de tocar rutina/dieta.
    if (
        had_active_plan
        and not state["safety_flag"]
        and state.get("feedback_classification")
    ):
        routine_record = repository.get_latest_routine_record()
        diet_record = repository.get_latest_diet_record()
        repository.save_feedback(
            raw_text=user_input,
            classification=state["feedback_classification"],
            routine_id=routine_record.id if routine_record else None,
            diet_id=diet_record.id if diet_record else None,
            week_number=state["week_number"],
        )

    # Si en este turno se (re)generó rutina/dieta, se persiste: save_* marca la
    # anterior como "superseded" y deja la nueva "active".
    routine_regenerated = (
        state["routine"] is not None and state["routine"] is not previous_routine
    )
    diet_regenerated = state["diet"] is not None and state["diet"] is not previous_diet
    if routine_regenerated:
        repository.save_routine(state["routine"], state["week_number"])
    if diet_regenerated:
        repository.save_diet(state["diet"], state["week_number"])
    if routine_regenerated or diet_regenerated:
        # El plan activo pasa a ser el de esta semana: reinicia el disparador
        # del check-in y encadena el próximo ajuste sobre el plan vigente.
        state["active_plan_week"] = state["week_number"]
    state["previous_routine"] = state["routine"]
    state["previous_diet"] = state["diet"]

    return state
