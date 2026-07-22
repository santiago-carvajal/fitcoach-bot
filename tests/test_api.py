"""
Tests del seam HTTP (issue #5): FastAPI + TestClient con LLM mockeado. Cubren
el contrato request/response de /chat y /state, y que el estado persiste a
través de un "reinicio" simulado (una app nueva que lee la misma SQLite).

La DB se aísla por test apuntando el engine a un archivo SQLite temporal
(monkeypatch de src.db.engine.engine), que get_session()/create_db_and_tables()
resuelven en cada llamada.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlmodel import create_engine

import src.db.engine as db_engine
from src.api import create_app
from src.db import repository
from src.schemas.output_schemas import (
    DietPlan,
    Equipment,
    ExperienceLevel,
    Goal,
    Meal,
    UserProfile,
    WorkoutDay,
    WorkoutRoutine,
)
from src.schemas.safety_schema import SafetyClassification


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Apunta el engine a una SQLite temporal y crea las tablas. Devuelve la
    ruta para poder reabrir la misma DB desde una app 'reiniciada'."""
    db_file = tmp_path / "fitcoach_test.db"
    monkeypatch.setattr(
        db_engine, "engine", create_engine(f"sqlite:///{db_file}", echo=False)
    )
    db_engine.create_db_and_tables()
    return str(db_file)


def _full_profile() -> UserProfile:
    return UserProfile(
        age=30,
        sex="masculino",
        weight_kg=80,
        height_cm=178,
        goal=Goal.GAIN_MUSCLE,
        experience_level=ExperienceLevel.INTERMEDIATE,
        days_per_week=3,
        session_duration_min=60,
        equipment=Equipment.FULL_GYM,
    )


def test_state_endpoint_devuelve_estado_persistido(temp_db, make_graph_llm, monkeypatch):
    # Sembramos perfil + rutina + dieta directamente en la DB.
    repository.save_profile(_full_profile())
    routine = WorkoutRoutine(
        days=[WorkoutDay(day_name="Día 1", focus="Empuje", exercises=[])],
        general_notes="rutina sembrada",
    )
    diet = DietPlan(
        daily_calories=2200,
        protein_g=160,
        carbs_g=220,
        fat_g=70,
        meals=[Meal(name="Desayuno", foods=["Avena"], approx_calories=400,
                    protein_g=20, carbs_g=60, fat_g=8)],
    )
    repository.save_routine(routine, 202630)
    repository.save_diet(diet, 202630)

    with TestClient(create_app()) as client:
        response = client.get("/state")

    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == "done"
    assert body["profile"]["age"] == 30
    assert body["profile"]["goal"] == Goal.GAIN_MUSCLE.value
    assert body["routine"]["general_notes"] == "rutina sembrada"
    assert body["routine"]["days"][0]["focus"] == "Empuje"
    assert body["diet"]["daily_calories"] == 2200
    assert body["diet"]["meals"][0]["name"] == "Desayuno"


def test_state_endpoint_vacio_al_inicio(temp_db):
    with TestClient(create_app()) as client:
        body = client.get("/state").json()
    assert body["stage"] == "collecting"
    assert body["messages"] == []
    assert body["profile"] is None
    assert body["routine"] is None
    assert body["diet"] is None


def test_chat_endpoint_avanza_y_persiste(temp_db, make_graph_llm, monkeypatch):
    # Perfil incompleto -> el grafo rutea a ask_missing (pregunta, no genera).
    partial = UserProfile(age=30, goal=Goal.LOSE_FAT)
    fake = make_graph_llm(
        responses={
            UserProfile: partial,
            SafetyClassification: SafetyClassification(
                mentions_injury=False,
                mentions_medical_condition=False,
                mentions_pregnancy=False,
            ),
        },
        plain_response=AIMessage(content="¿Cuántos días por semana puedes entrenar?"),
    )
    monkeypatch.setattr("src.nodes.llm", fake)

    with TestClient(create_app()) as client:
        response = client.post("/chat", json={"message": "Hola, quiero bajar de peso"})

    assert response.status_code == 200
    body = response.json()
    # La conversación devuelta incluye el turno del usuario y la respuesta.
    assert body["messages"][0] == {"role": "human", "content": "Hola, quiero bajar de peso"}
    assert body["messages"][-1] == {
        "role": "ai",
        "content": "¿Cuántos días por semana puedes entrenar?",
    }
    assert body["stage"] == "collecting"
    assert body["routine"] is None
    assert body["diet"] is None

    # Persistió: un GET /state posterior ve los mismos mensajes.
    with TestClient(create_app()) as client:
        state_body = client.get("/state").json()
    assert [m["content"] for m in state_body["messages"]] == [
        "Hola, quiero bajar de peso",
        "¿Cuántos días por semana puedes entrenar?",
    ]


def test_chat_endpoint_rechaza_mensaje_vacio(temp_db):
    with TestClient(create_app()) as client:
        response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400


def test_sirve_la_pagina_estatica_y_sus_assets(temp_db):
    # El frontend (issue #6) se sirve desde la misma app: index.html en "/" y
    # los assets en "/static".
    with TestClient(create_app()) as client:
        index = client.get("/")
        app_js = client.get("/static/app.js")
        styles = client.get("/static/styles.css")

    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    assert "<title>Coach IA" in index.text
    assert app_js.status_code == 200
    assert "/state" in app_js.text and "/chat" in app_js.text
    assert styles.status_code == 200


def test_estado_persiste_a_traves_de_un_reinicio(temp_db, make_graph_llm, monkeypatch):
    partial = UserProfile(age=25, goal=Goal.GAIN_MUSCLE)
    fake = make_graph_llm(
        responses={
            UserProfile: partial,
            SafetyClassification: SafetyClassification(
                mentions_injury=False,
                mentions_medical_condition=False,
                mentions_pregnancy=False,
            ),
        },
        plain_response=AIMessage(content="¿Con qué equipo cuentas?"),
    )
    monkeypatch.setattr("src.nodes.llm", fake)

    # App 1: manda un mensaje.
    with TestClient(create_app()) as client1:
        client1.post("/chat", json={"message": "Quiero ganar músculo"})

    # App 2: instancia nueva (reinicio simulado) leyendo la MISMA SQLite.
    with TestClient(create_app()) as client2:
        body = client2.get("/state").json()

    assert [m["content"] for m in body["messages"]] == [
        "Quiero ganar músculo",
        "¿Con qué equipo cuentas?",
    ]
    assert body["profile"]["age"] == 25
