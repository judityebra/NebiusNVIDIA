const $ = (s) => document.querySelector(s);
const chatHistory = [];

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
const sev = (s) => `<span class="sev ${esc(s)}">${esc(s)}</span>`;

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}

async function loadStatus() {
  const s = await api("/api/status");
  const el = $("#mode");
  el.textContent = s.mock ? "Mock mode (no API key)" : `Live · ${s.agent_model}`;
  el.classList.toggle("mock", s.mock);
}

async function loadZones() {
  const zones = await api("/api/zones");
  $("#zones").innerHTML = zones.map((z) => `
    <div class="zone"><h3>${esc(z.name)}</h3><small>${esc(z.id)} · ${esc(z.camera)}</small>
      <ul>${z.rules.map((r) => `<li>${sev(r.severity)}${esc(r.text)}</li>`).join("")}</ul></div>`).join("");
  $("#camera").innerHTML = zones.map((z) => `<option value="${esc(z.camera)}">${esc(z.camera)} · ${esc(z.name)}</option>`).join("");
}

async function loadFeed() {
  const only = $("#only-v").checked;
  const obs = await api(`/api/observations?only_violations=${only}`);
  $("#feed").innerHTML = obs.reverse().map((o) => `
    <li><div class="meta">obs ${o.id} · ${esc(o.ts)} · ${esc(o.zone_id)}</div>
      ${esc(o.summary)}
      ${o.violations.map((v) => `<div>${sev(v.severity)}${esc(v.rule)} <span class="meta">(${Math.round(v.confidence * 100)}%)</span></div>`).join("")}
    </li>`).join("");
}

async function loadIncidents() {
  const inc = await api("/api/incidents");
  $("#incidents").innerHTML = inc.length
    ? inc.map((i) => `<li>${sev(i.severity)}<strong>${esc(i.title)}</strong> <span class="meta">obs ${i.observation_ids.join(", ")}</span></li>`).join("")
    : "<li>None filed yet.</li>";
}

function addMsg(role, text, trace) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  if (trace && trace.length) {
    const d = document.createElement("details");
    d.innerHTML = `<summary>${trace.length} tool call(s)</summary>` +
      trace.map((t) => `<pre>${esc(t.tool)}(${esc(t.arguments)})\n→ ${esc(t.result.slice(0, 600))}</pre>`).join("");
    div.appendChild(d);
  }
  $("#chat").appendChild(div);
  $("#chat").scrollTop = $("#chat").scrollHeight;
}

async function ask(question) {
  const btn = $("#ask button");
  btn.disabled = true;
  addMsg("user", question);
  try {
    const r = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: chatHistory.slice(-6) }),
    });
    addMsg("bot", r.answer, r.trace);
    chatHistory.push({ role: "user", content: question }, { role: "assistant", content: r.answer });
    loadIncidents();
  } catch (e) {
    addMsg("bot", `Error: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

$("#ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (q) { $("#question").value = ""; ask(q); }
});
document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => ask(c.textContent)));
$("#only-v").addEventListener("change", loadFeed);

$("#upload").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("#upload button");
  const fd = new FormData();
  fd.append("camera", $("#camera").value);
  fd.append("image", $("#frame").files[0]);
  btn.disabled = true;
  try {
    await api("/api/frames", { method: "POST", body: fd });
    await loadFeed();
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
  }
});

loadStatus(); loadZones(); loadFeed(); loadIncidents();
