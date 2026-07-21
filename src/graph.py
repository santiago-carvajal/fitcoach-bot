from langgraph.graph import END, StateGraph

from src.nodes import (
    ask_missing_node,
    collect_profile_node,
    format_output_node,
    generate_diet_node,
    generate_routine_node,
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


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("collect_profile", collect_profile_node)
    graph.add_node("safety_guardrail", safety_guardrail_node)
    graph.add_node("ask_missing", ask_missing_node)
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
            # TODO(fase 3): reemplazar por "process_feedback" cuando se agregue
            # process_feedback_node — hoy este branch es inalcanzable porque
            # main.py todavía corta el loop en cuanto stage == "done".
            "process_feedback": END,
            "ask_missing": "ask_missing",
            "generate_routine": "generate_routine",
        },
    )
    graph.add_edge("ask_missing", END)  # espera el siguiente turno del usuario
    graph.add_edge("generate_routine", "generate_diet")
    graph.add_edge("generate_diet", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()
