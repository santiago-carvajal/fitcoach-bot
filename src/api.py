"""
API HTTP local (FastAPI) que expone el mismo grafo y persistencia que usa el
terminal (main.py). No hay auth, HTTPS ni despliegue en la nube: corre local
para un frontend web. Comparte la orquestación de cada turno con el terminal
via session.advance_turn, en vez de duplicarla.

Correr local:  uvicorn --factory src.api:create_app --reload

Se expone como fábrica (create_app) y no como un `app` a nivel de módulo a
propósito: así importar este módulo no crea tablas ni toca la DB (eso permite
tests con una SQLite temporal por caso).
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.db import repository  # noqa: E402
from src.db.engine import create_db_and_tables  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.session import advance_turn  # noqa: E402
from src.state import GraphState  # noqa: E402

_MESSAGE_ROLE = {HumanMessage: "human", AIMessage: "ai", SystemMessage: "system"}


class ChatRequest(BaseModel):
    message: str


def _serialize_state(state: GraphState) -> dict:
    """Aplana el GraphState a JSON para la respuesta HTTP: conversación,
    perfil, rutina y dieta actuales. Los objetos Pydantic se vuelcan con
    model_dump(mode='json'); los mensajes de LangChain a {role, content}."""
    profile = state.get("user_profile")
    routine = state.get("routine")
    diet = state.get("diet")
    return {
        "stage": state["stage"],
        "messages": [
            {"role": _MESSAGE_ROLE.get(type(m), "ai"), "content": m.content}
            for m in state["messages"]
        ],
        "profile": profile.model_dump(mode="json") if profile else None,
        "routine": routine.model_dump(mode="json") if routine else None,
        "diet": diet.model_dump(mode="json") if diet else None,
    }


def create_app() -> FastAPI:
    """Fábrica de la app: crea las tablas si faltan y compila el grafo una sola
    vez para reusarlo entre requests (el grafo compilado no guarda estado por
    conversación; el estado vive en SQLite y se recarga con load_full_state en
    cada request, así la API sobrevive reinicios)."""
    create_db_and_tables()
    graph = build_graph()
    app = FastAPI(title="FitCoach API")

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict:
        if not request.message.strip():
            raise HTTPException(
                status_code=400, detail="El mensaje no puede estar vacío."
            )
        state = repository.load_full_state()
        state = advance_turn(graph, state, request.message)
        return _serialize_state(state)

    @app.get("/state")
    def get_state() -> dict:
        return _serialize_state(repository.load_full_state())

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
