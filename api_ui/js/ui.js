function renderJSON(selector, obj){
  const el = document.querySelector(selector);
  try { el.textContent = JSON.stringify(obj, null, 2); }
  catch { el.textContent = String(obj); }
}

async function fetchGET(url, outSel){
  const res = await fetch(url, { method: 'GET' });
  let body;
  try { body = await res.json(); } catch { body = await res.text(); }
  renderJSON(outSel, body);
  return body;
}

async function fetchPOST(url, data, outSel){
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify(data || {})
  });
  let body;
  try { body = await res.json(); } catch { body = await res.text(); }
  renderJSON(outSel, body);
  return body;
}

// Auto-bind any <button class="btn" data-get="/api/...">
window.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-get]');
  if (btn) {
    const url = btn.getAttribute('data-get');
    const outSel = btn.getAttribute('data-out') || '#result';
    fetchGET(url, outSel);
  }
});
