from langgraph.graph import END, StateGraph

from src.nodes import (
    ask_missing_node,
    collect_profile_node,
    format_output_node,
    generate_diet_node,
    generate_routine_node,
    process_feedback_node,
    safety_guardrail_node,
)
from src.state import GraphState


def route_after_guardrail(state: GraphState) -> str:
    if state.get("safety_flag"):
        return "deferral"
    if state["stage"] == "done":
        return "process_feedback"
    if state["missing_fields"]:
        return "ask_missing"
    return "generate_routine"


def route_after_feedback(state: GraphState) -> str:
    """Tras clasificar el feedback: regenerar el plan o cerrar el turno con la
    respuesta conversacional que process_feedback_node ya dejó en messages."""
    classification = state.get("feedback_classification") or {}
    return "regenerate" if classification.get("wants_regeneration") else "reply"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("collect_profile", collect_profile_node)
    graph.add_node("safety_guardrail", safety_guardrail_node)
    graph.add_node("ask_missing", ask_missing_node)
    graph.add_node("process_feedback", process_feedback_node)
    graph.add_node("generate_routine", generate_routine_node)
    graph.add_node("generate_diet", generate_diet_node)
    graph.add_node("format_output", format_output_node)

    graph.set_entry_point("collect_profile")

    graph.add_edge("collect_profile", "safety_guardrail")
    graph.add_conditional_edges(
        "safety_guardrail",
        route_after_guardrail,
        {
            "deferral": END,
            "process_feedback": "process_feedback",
            "ask_missing": "ask_missing",
            "generate_routine": "generate_routine",
        },
    )
    # Con plan activo, el feedback ajusta (regenera) el plan o cierra el turno
    # con una respuesta conversacional; la regeneración reusa generate_routine.
    graph.add_conditional_edges(
        "process_feedback",
        route_after_feedback,
        {
            "regenerate": "generate_routine",
            "reply": END,
        },
    )
    graph.add_edge("ask_missing", END)  # espera el siguiente turno del usuario
    graph.add_edge("generate_routine", "generate_diet")
    graph.add_edge("generate_diet", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()
