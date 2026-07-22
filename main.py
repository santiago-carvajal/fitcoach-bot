from dotenv import load_dotenv

load_dotenv()

from src.db import repository  # noqa: E402
from src.db.engine import create_db_and_tables
from src.graph import build_graph
from src.session import advance_turn


def run():
    create_db_and_tables()
    graph = build_graph()
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
    # regenerar y ofrecer el check-in semanal. Se sale con Ctrl-C o EOF. La
    # orquestación de cada turno vive en session.advance_turn, compartida con
    # la API (src/api.py).
    while True:
        try:
            user_input = input("Tú: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue

        state = advance_turn(graph, state, user_input)
        print(f"Coach IA: {state['messages'][-1].content}")


if __name__ == "__main__":
    run()
