// Creative AI Engine — panel de una pantalla, tres estados.
// Consume el endpoint SSE /api/v1/evolution/stream y muestra el abanico
// de ideas agrupado en familias, apareciendo en vivo.

const $ = (id) => document.getElementById(id);

const state = { domain: "generic", running: false };

// ---- Estado 1: selección de ámbito y validación ----
const challengeEl = $("challenge");
const goBtn = $("go");

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
    chip.setAttribute("aria-pressed", "true");
    state.domain = chip.dataset.domain;
  });
});

document.querySelectorAll(".ex").forEach((ex) => {
  ex.addEventListener("click", () => {
    challengeEl.value = ex.textContent;
    challengeEl.dispatchEvent(new Event("input"));
    challengeEl.focus();
  });
});

challengeEl.addEventListener("input", () => {
  goBtn.disabled = challengeEl.value.trim().length < 10;
});

challengeEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !goBtn.disabled) start();
});

goBtn.addEventListener("click", start);
$("again").addEventListener("click", reset);

// ---- Transición a estado 2 ----
function start() {
  const challenge = challengeEl.value.trim();
  if (challenge.length < 10) return;
  state.running = true;

  $("ask").style.display = "none";
  $("live").classList.add("on");
  $("liveQ").textContent = challenge;
  $("status").textContent = "Preparando…";
  $("progress").style.width = "0%";
  $("note").textContent = "Generando la primera oleada de ideas…";
  $("families").innerHTML = "";
  $("skeleton").style.display = "grid";
  $("actions").classList.remove("on");
  $("err").classList.remove("on");
  window.scrollTo({ top: 0, behavior: "smooth" });

  stream(challenge, state.domain);
}

// ---- Streaming SSE (vía fetch, para poder enviar POST) ----
async function stream(challenge, domain) {
  let total = 10;
  try {
    const resp = await fetch("/api/v1/evolution/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ challenge, domain }),
    });
    if (!resp.ok || !resp.body) throw new Error(`El servidor respondió ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const chunks = buffer.split("\n\n");
      buffer = chunks.pop(); // resto incompleto

      for (const chunk of chunks) {
        const ev = parseSSE(chunk);
        if (ev) handleEvent(ev, total, (t) => (total = t));
      }
    }
  } catch (e) {
    showError(e.message || "No se pudo conectar con el motor.");
  }
}

function parseSSE(chunk) {
  let event = "message", data = "";
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7).trim();
    else if (line.startsWith("data: ")) data += line.slice(6);
    else if (line.startsWith(":")) return null; // keep-alive
  }
  if (!data) return null;
  try { return { event, data: JSON.parse(data) }; } catch { return null; }
}

// ---- Manejo de eventos ----
function handleEvent(ev, total, setTotal) {
  switch (ev.event) {
    case "start":
      if (ev.data.total_generations) setTotal(ev.data.total_generations);
      $("status").textContent = "Explorando enfoques…";
      break;

    case "progress": {
      const gen = ev.data.generation || 0;
      const pct = Math.min(100, Math.round((gen / total) * 100));
      $("progress").style.width = pct + "%";
      $("status").textContent = `Explorando enfoques · ${gen} de ${total}`;
      if (ev.data.families && ev.data.families.length) {
        $("skeleton").style.display = "none";
        const n = ev.data.families.length;
        $("note").textContent = `${n} enfoque${n === 1 ? "" : "s"} distinto${n === 1 ? "" : "s"} hasta ahora — siguen evolucionando.`;
        renderFamilies(ev.data.families, false);
      }
      break;
    }

    case "done":
      $("progress").style.width = "100%";
      $("status").textContent = "Listo";
      $("skeleton").style.display = "none";
      renderFamilies(ev.data.families || [], true, ev.data.run_id);
      finish(ev.data);
      break;

    case "error":
      showError(ev.data.message || "El motor encontró un problema.");
      break;
  }
}

// ---- Render de familias ----
function renderFamilies(families, final, runId) {
  const host = $("families");
  host.innerHTML = "";
  families.forEach((fam, i) => {
    const rep = fam.representative || {};
    const el = document.createElement("article");
    el.className = "family";
    el.style.animationDelay = `${Math.min(i * 60, 400)}ms`;

    const adv = (rep.advantages || [])
      .map((a) => `<span class="tag">${escapeHtml(a)}</span>`)
      .join("");

    const variants = (fam.members || [])
      .filter((m) => m.id !== rep.id)
      .map((m) => `<li>${escapeHtml(m.title)}</li>`)
      .join("");

    const variantsBlock = variants
      ? `<details class="variants"><summary>${fam.size - 1} variante${fam.size - 1 === 1 ? "" : "s"} de este enfoque</summary><ul>${variants}</ul></details>`
      : "";

    const novelty = rep.novelty != null ? Math.round(rep.novelty * 100) : null;

    el.innerHTML = `
      <div class="family-top">
        <div class="family-idx">${String(i + 1).padStart(2, "0")}</div>
        <div class="family-body">
          <h3 class="family-title">${escapeHtml(rep.title || "Idea")}</h3>
          <p class="family-desc">${escapeHtml(rep.description || "")}</p>
          ${adv ? `<div class="family-adv">${adv}</div>` : ""}
          <div class="family-meta">
            <span>Solidez <span class="meta-val">${Math.round((rep.fitness || 0) * 100)}%</span></span>
            ${novelty != null ? `<span>Originalidad <span class="meta-val">${novelty}%</span></span>` : ""}
          </div>
          ${variantsBlock}
          ${final ? `<button class="report-btn" data-id="${rep.id}">Generar informe</button><div class="report-out"></div>` : ""}
        </div>
      </div>`;
    host.appendChild(el);
  });

  if (final) wireReportButtons();
}

// ---- Informe por familia (bajo demanda) ----
function wireReportButtons() {
  document.querySelectorAll(".report-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const out = btn.nextElementSibling;
      btn.disabled = true;
      btn.textContent = "Redactando…";
      try {
        const r = await fetch(`/api/v1/ideas/${btn.dataset.id}/report`, { method: "POST" });
        if (!r.ok) throw new Error();
        const data = await r.json();
        out.textContent = data.report || "Sin contenido.";
        out.classList.add("on");
        btn.style.display = "none";
      } catch {
        btn.textContent = "No se pudo generar — reintentar";
        btn.disabled = false;
      }
    });
  });
}

// ---- Estado 3: cierre ----
function finish(data) {
  state.running = false;
  const n = (data.families || []).length;
  $("actions").classList.add("on");
  if (data.run_id) {
    const dl = $("download");
    dl.href = `/api/v1/runs/${data.run_id}/export`;
    dl.hidden = false;
  }
  $("summary").textContent = `${data.total_ideas || 0} ideas exploradas · ${n} enfoque${n === 1 ? "" : "s"} destacado${n === 1 ? "" : "s"}.`;
}

function showError(msg) {
  state.running = false;
  $("skeleton").style.display = "none";
  $("err").innerHTML = `<b>No se pudo completar.</b> ${escapeHtml(msg)} Revisa que el motor esté configurado y vuelve a intentarlo.`;
  $("err").classList.add("on");
  $("actions").classList.add("on");
  $("status").textContent = "Detenido";
}

function reset() {
  $("download").hidden = true;
  $("live").classList.remove("on");
  $("ask").style.display = "block";
  challengeEl.value = "";
  goBtn.disabled = true;
  challengeEl.focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}
