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
        # Ya hay un plan activo persistido. El ajuste por feedback llega con
        # la fase de progresión (issue #4); mientras tanto, no regeneramos.
        print(
            "Coach IA: Ya tienes una rutina y una dieta activas guardadas. "
            "Por ahora no puedo ajustarlas por feedback; si quieres empezar "
            "de cero, borra fitcoach.db y vuelve a ejecutarme."
        )
        return

    print(
        "Coach IA: ¡Hola! Cuéntame sobre ti y tu objetivo (edad, peso, "
        "altura, qué buscas lograr, cuántos días puedes entrenar y con "
        "qué equipo cuentas)."
    )

    while state["stage"] != "done":
        user_input = input("Tú: ")
        user_message = HumanMessage(content=user_input)
        repository.append_message(user_message)
        state["messages"].append(user_message)

        previous_message_count = len(state["messages"])
        state = app.invoke(state)
        for message in state["messages"][previous_message_count:]:
            repository.append_message(message)

        if state.get("user_profile") is not None:
            repository.save_profile(state["user_profile"])

        # El loop solo corre mientras stage != "done", así que si el invoke lo
        # dejó en "done" la rutina/dieta se generaron en este mismo turno:
        # es el momento de persistirlas (la anterior queda "superseded").
        if state["stage"] == "done" and state["routine"] and state["diet"]:
            repository.save_routine(state["routine"], state["week_number"])
            repository.save_diet(state["diet"], state["week_number"])

        print(f"Coach IA: {state['messages'][-1].content}")


if __name__ == "__main__":
    run()
