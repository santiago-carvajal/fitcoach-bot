// Coach IA — cliente estático (vanilla JS, sin frameworks).
// Es un cliente delgado de la API (#5): GET /state al cargar para restaurar
// la conversación y el plan; POST /chat por cada mensaje. Ambos devuelven el
// estado completo, que se re-renderiza entero.

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const errorEl = document.getElementById("error");
const planEmptyEl = document.getElementById("plan-empty");
const planContentEl = document.getElementById("plan-content");
const routineEl = document.getElementById("routine");
const dietEl = document.getElementById("diet");

// Escapa texto antes de meterlo al DOM como HTML (los planes/mensajes vienen
// del LLM, así que nunca los inyectamos crudos).
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}

function clearError() {
  errorEl.hidden = true;
}

function renderMessages(messages) {
  messagesEl.innerHTML = "";
  if (!messages || messages.length === 0) {
    const hint = document.createElement("div");
    hint.className = "empty-hint";
    hint.textContent =
      "¡Hola! Soy Coach IA. Cuéntame tu edad, peso, altura, objetivo, " +
      "cuántos días puedes entrenar y con qué equipo cuentas.";
    messagesEl.appendChild(hint);
    return;
  }
  for (const msg of messages) {
    const bubble = document.createElement("div");
    const role = msg.role === "human" ? "human" : "ai";
    bubble.className = `bubble ${role}`;
    bubble.textContent = msg.content;
    messagesEl.appendChild(bubble);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function macro(label, value, unit) {
  return `<span class="macro">${escapeHtml(label)}: <strong>${escapeHtml(
    value
  )}${unit ? " " + escapeHtml(unit) : ""}</strong></span>`;
}

function renderRoutine(routine) {
  if (!routine) return "";
  const days = (routine.days || [])
    .map((day) => {
      const exercises = (day.exercises || [])
        .map((ex) => {
          const detail = [
            `${escapeHtml(ex.sets)}x${escapeHtml(ex.reps)}`,
            ex.rest_seconds != null
              ? `descanso ${escapeHtml(ex.rest_seconds)}s`
              : null,
            ex.notes ? escapeHtml(ex.notes) : null,
          ]
            .filter(Boolean)
            .join(" · ");
          return `<div class="exercise">• ${escapeHtml(
            ex.name
          )} <span class="detail">(${detail})</span></div>`;
        })
        .join("");
      return `<div class="day">
        <div class="day-title">${escapeHtml(day.day_name)}
          <span class="day-focus">— ${escapeHtml(day.focus)}</span></div>
        ${exercises}
      </div>`;
    })
    .join("");
  const notes = routine.general_notes
    ? `<div class="notes">${escapeHtml(routine.general_notes)}</div>`
    : "";
  return `<h3>Rutina</h3>${days}${notes}`;
}

function renderDiet(diet) {
  if (!diet) return "";
  const macros = `<div class="macro-row">
    ${macro("Calorías", diet.daily_calories, "kcal")}
    ${macro("Proteína", diet.protein_g, "g")}
    ${macro("Carbos", diet.carbs_g, "g")}
    ${macro("Grasa", diet.fat_g, "g")}
  </div>`;
  const meals = (diet.meals || [])
    .map((meal) => {
      const foods = (meal.foods || [])
        .map((f) => `<div class="food">• ${escapeHtml(f)}</div>`)
        .join("");
      return `<div class="meal">
        <div class="meal-title">${escapeHtml(meal.name)}
          <span class="day-focus">— ${escapeHtml(
            meal.approx_calories
          )} kcal</span></div>
        ${foods}
      </div>`;
    })
    .join("");
  const notes = diet.general_notes
    ? `<div class="notes">${escapeHtml(diet.general_notes)}</div>`
    : "";
  return `<h3>Dieta</h3>${macros}${meals}${notes}`;
}

function renderPlan(routine, diet) {
  if (!routine && !diet) {
    planEmptyEl.hidden = false;
    planContentEl.hidden = true;
    routineEl.innerHTML = "";
    dietEl.innerHTML = "";
    return;
  }
  planEmptyEl.hidden = true;
  planContentEl.hidden = false;
  routineEl.innerHTML = renderRoutine(routine);
  dietEl.innerHTML = renderDiet(diet);
}

function render(state) {
  renderMessages(state.messages);
  renderPlan(state.routine, state.diet);
}

async function loadState() {
  try {
    const res = await fetch("/state");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    render(await res.json());
  } catch (err) {
    showError("No se pudo cargar el estado: " + err.message);
  }
}

function setPending(pending) {
  sendBtn.disabled = pending;
  inputEl.disabled = pending;
  sendBtn.textContent = pending ? "…" : "Enviar";
}

async function sendMessage(text) {
  setPending(true);
  clearError();
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }
    render(await res.json());
    // Solo se limpia el input si el envío tuvo éxito; si falló, el texto
    // queda para que el usuario reintente sin reescribirlo.
    inputEl.value = "";
  } catch (err) {
    showError("No se pudo enviar el mensaje: " + err.message);
  } finally {
    setPending(false);
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  sendMessage(text);
});

// Al abrir (o recargar) la pestaña, restaura conversación + plan desde /state.
loadState();
