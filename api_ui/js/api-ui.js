// Minimal helper used in each page, scoped to API UI
const API = {
  get: async (url) => {
    const res = await fetch(url, { headers: { "Accept": "application/json" } });
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, data };
  },
  post: async (url, body = {}) => {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, data };
  }
};

function renderJSON(el, obj) {
  el.textContent = JSON.stringify(obj, null, 2);
}

function badge(text, type="ok") {
  const span = document.createElement("span");
  span.className = `api-badge ${type}`;
  span.textContent = text;
  return span;
}