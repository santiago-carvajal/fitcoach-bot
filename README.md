# FitCoach Bot — esqueleto

Chatbot tipo "entrenador personal" que recolecta el perfil del usuario
por conversación y genera una rutina de ejercicios + un plan de
alimentación personalizados, usando Claude + LangGraph.

## Estructura

```
fitcoach-bot/
├── main.py                    # loop de prueba por terminal
├── requirements.txt
├── .env.example
├── data/
│   └── exercises.csv           # catálogo original de ejercicios (RAG)
├── tests/                      # seams: retrieval (sin LLM) y nodos (LLM mockeado)
└── src/
    ├── state.py                # estado compartido del grafo (GraphState)
    ├── graph.py                # arma el grafo y el enrutamiento
    ├── nodes.py                # lógica de cada nodo (llamadas al LLM)
    ├── db/                     # persistencia SQLite (SQLModel)
    ├── retrieval/
    │   └── exercise_catalog.py # retrieval TF-IDF (minsearch) sobre el catálogo
    ├── prompts/
    │   └── system_prompt.py    # persona y reglas del coach
    └── schemas/
        └── output_schemas.py   # UserProfile, WorkoutRoutine, DietPlan (Pydantic)
```

## Flujo del grafo

```
collect_profile → (¿faltan datos?) ─sí→ ask_missing → END (espera respuesta del usuario)
                                    └no→ generate_routine → generate_diet → format_output → END
```

Cada turno del usuario vuelve a entrar por `collect_profile`, que va
completando el `UserProfile` acumulado hasta que no falte nada crítico.
Ahí recién se generan rutina y dieta como JSON estructurado (Pydantic),
y `format_output` los convierte en un mensaje legible.

`generate_routine` no inventa ejercicios: antes de llamar al LLM
recupera candidatos de `data/exercises.csv` vía `minsearch` (TF-IDF en
memoria), filtrados por el equipo declarado del usuario (jerarquía
sin_equipo ⊂ casa_mancuernas ⊂ gimnasio_completo) y rankeados hacia su
objetivo, y el prompt le prohíbe salirse de esa lista. El dataset es
original de este proyecto; del repo de referencia solo se reusa el
patrón CSV + minsearch (ver sección siguiente).

## Cómo correrlo

```bash
pip install -r requirements.txt
cp .env.example .env   # y pega tu ANTHROPIC_API_KEY
python main.py
```

## Fuente candidata para el RAG de ejercicios

Repo evaluado: [alexeygrigorev/fitness-assistant](https://github.com/alexeygrigorev/fitness-assistant)
(proyecto RAG hecho para el curso LLM Zoomcamp).

**Qué sirve de ahí:**

- `data/data.csv` — 207 ejercicios con: nombre, tipo de actividad
  (fuerza / movilidad / cardio), equipo requerido, parte del cuerpo,
  tipo de movimiento, grupos musculares activados e instrucciones
  paso a paso. Encaja casi directo con los campos que ya usamos en
  `WorkoutRoutine`/`Exercise` (`schemas/output_schemas.py`) — nos
  ahorra construir un dataset de ejercicios desde cero.
- El patrón `ingest.py` + `rag.py` + `minsearch.py`: cargan el CSV a
  un índice en memoria (`minsearch`, búsqueda tipo TF-IDF con boosting
  por campo) y recuperan sobre eso antes de armar el prompt. Para
  ~200 registros esto es más simple y barato que montar un vector
  store — probablemente nos sirve tal cual como primer approach de
  retrieval antes de pensar en embeddings.
- `notebooks/evaluation-data-generation.ipynb` y sus métricas de
  retrieval (hit rate / MRR comparando boostings) — buena referencia
  para evaluar si nuestro propio retrieval devuelve los ejercicios
  correctos antes de confiar en él en producción.

**Qué NO cubre:**

- No trae dataset de nutrición/dieta — solo resuelve el lado de
  rutina, no el de dieta.
- El dataset fue generado con ChatGPT (sintético), no validado
  clínica ni fisioterapéuticamente — reduce el riesgo de que el LLM
  invente ejercicios que no existen, pero no garantiza que las
  instrucciones sean correctas para todos los casos.
- Está construido con OpenAI + Flask, no Claude + LangGraph — no es
  código reusable tal cual, es referencia de patrón. `minsearch` sí
  es instalable por separado (`pip install minsearch`) y podría
  conectarse como un nodo de retrieval dentro de nuestro grafo.
- No se encontró una licencia explícita (archivo LICENSE) en el
  repo — hay que confirmarla antes de empaquetar su dataset dentro
  de un producto de Beecker.

## Próximos pasos sugeridos (llévalos a Claude Code con `/grill-me`)

Este esqueleto cubre el flujo mínimo. Antes de seguir construyendo,
usa `/grill-me` para resolver estas decisiones de diseño que quedaron
abiertas a propósito:

- **Persistencia**: ¿el perfil del usuario se guarda entre sesiones
  (DB) o se recolecta desde cero cada vez?
- **Canal**: ¿CLI de prueba, API + frontend web, bot de Telegram/WhatsApp?
- **Base de conocimiento**: ya tenemos una fuente candidata para
  ejercicios (ver sección "Fuente candidata para el RAG de
  ejercicios" arriba) — falta decidir si se adopta `minsearch` tal
  cual o se migra a un vector store, confirmar la licencia del
  dataset, y resolver si la dieta también lleva RAG o queda
  generativa por ahora.
- **Progresión**: ¿la rutina se ajusta semana a semana según feedback
  del usuario, o es un plan estático?
- **Validación de seguridad**: ¿qué tan estricto debe ser el bot al
  detectar lesiones/condiciones médicas antes de generar contenido?

Después de `/grill-me`, usa `/to-prd` para documentar las respuestas y
`/to-issues` para partir el trabajo restante (persistencia, canal, RAG,
etc.) en tickets independientes. `/tdd` es especialmente útil para el
parser de `UserProfile` — es la pieza que más falla si no hay tests
(usuarios que responden todo junto, en desorden, con unidades distintas).
