from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from src.db import repository  # noqa: E402
from src.db.engine import create_db_and_tables
from src.graph import build_graph


def run():
    create_db_and_tables()
    app = build_graph()
    state = repository.load_full_state()

    if state["stage"] == "done":
        # Ya hay un plan activo persistido: el loop de progresión (issue #4) sí
        # es alcanzable ahora, así que invitamos al usuario a dar feedback.
        print(
            "Coach IA: ¡Hola de nuevo! Ya tienes una rutina y una dieta "
            "activas. Cuéntame cómo te ha ido: si algo te resultó muy fácil o "
            "muy difícil, o si cambió tu disponibilidad, ajusto tu plan."
        )
    else:
        print(
            "Coach IA: ¡Hola! Cuéntame sobre ti y tu objetivo (edad, peso, "
            "altura, qué buscas lograr, cuántos días puedes entrenar y con "
            "qué equipo cuentas)."
        )

    # El loop ya no termina al generar el plan: sigue vivo para leer feedback,
    # regenerar y ofrecer el check-in semanal. Se sale con Ctrl-C o EOF.
    while True:
        try:
            user_input = input("Tú: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue

        user_message = HumanMessage(content=user_input)
        repository.append_message(user_message)
        state["messages"].append(user_message)

        had_active_plan = state["stage"] == "done"
        previous_routine = state["routine"]
        previous_diet = state["diet"]

        previous_message_count = len(state["messages"])
        state = app.invoke(state)
        for message in state["messages"][previous_message_count:]:
            repository.append_message(message)

        if state.get("user_profile") is not None:
            repository.save_profile(state["user_profile"])

        # Con un plan ya activo, process_feedback_node clasificó este turno
        # (salvo que el guardrail de seguridad derivara). Persistimos el
        # feedback ligado al plan que lo motivó, antes de tocar rutina/dieta.
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

        # Si en este turno se (re)generó rutina/dieta, se persiste: save_*
        # marca la anterior como "superseded" y deja la nueva "active".
        routine_regenerated = (
            state["routine"] is not None and state["routine"] is not previous_routine
        )
        diet_regenerated = (
            state["diet"] is not None and state["diet"] is not previous_diet
        )
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

        print(f"Coach IA: {state['messages'][-1].content}")


if __name__ == "__main__":
    run()
