const $ = id => document.getElementById(id);
const key = $('apiKey'), filesEl = $('files'), fileList = $('fileList');
try { key.value = localStorage.getItem('gemini_key') || ''; } catch {}
key.addEventListener('input', () => { try { localStorage.setItem('gemini_key', key.value); } catch {} });

fetch('/api/config').then(r => r.json()).then(c => {
  $('keyHint').textContent = c.has_server_key ? '(server key set — field optional)' : '(no server key — required)';
  const sel = $('model');
  if (c.default_model && [...sel.options].some(o => o.value === c.default_model)) {
    sel.value = c.default_model;
  } else if (c.default_model) {
    sel.value = 'custom';
    $('modelCustom').value = c.default_model;
  }
  if (c.default_base_url) $('baseUrl').value = c.default_base_url;
  if (c.default_reasoning) $('reasoning').value = c.default_reasoning;
  if (c.default_reasoning_max_tokens != null) $('reasoningMax').value = c.default_reasoning_max_tokens;
  if (c.default_pdf_as_file) $('pdfMode').value = 'file';
  if (c.default_pdf_engine && [...$('pdfEngine').options].some(o => o.value === c.default_pdf_engine)) {
    $('pdfEngine').value = c.default_pdf_engine;
  }
  if (c.transcribe) {
    const t = c.transcribe;
    $('txEngine').textContent = ` — ${t.gpu ? 'GPU' : 'CPU'} (${t.device}/${t.compute_type}`
      + (t.batch_size > 0 ? `, batch=${t.batch_size}` : '') + ')';
  }
  sel.dispatchEvent(new Event('change'));
}).catch(() => {});

function selectedModel() {
  return $('model').value === 'custom' ? $('modelCustom').value.trim() : $('model').value;
}
$('model').addEventListener('change', () => {
  $('modelCustom').classList.toggle('hidden', $('model').value !== 'custom');
});

let picked = [];
let manualSrc = false;
const AV = /\.(mp3|wav|m4a|aac|flac|ogg|opus|mp4|mov|avi|mkv|webm)$/i;
function render() {
  fileList.innerHTML = picked.map((f, i) =>
    `<div>${manualSrc ? `<input type="checkbox" class="psrc" data-i="${i}" ${AV.test(f.name) ? 'checked' : ''} title="primary lesson source"/> ` : ''}${f.name} — ${(f.size/1e6).toFixed(1)} MB</div>`).join('');
}
$('srcMode').addEventListener('change', () => { manualSrc = $('srcMode').value === 'manual'; render(); });
filesEl.addEventListener('change', () => { picked = [...filesEl.files]; render(); });
const drop = $('drop');
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.style.borderColor = '#2f6fed'; }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.style.borderColor = ''; }));
drop.addEventListener('drop', ev => { picked = [...ev.dataTransfer.files]; render(); });

let timer = null;
let currentId = null;
$('go').onclick = async () => {
  if (!picked.length) { alert('Add files first.'); return; }
  const fd = new FormData();
  picked.forEach(f => fd.append('files', f));
  fd.append('api_key', key.value || '');
  fd.append('model', selectedModel());
  fd.append('base_url', $('baseUrl').value);
  fd.append('temperature', $('temperature').value);
  fd.append('language', $('language').value);
  fd.append('transcribe_lang', $('transcribe_lang').value);
  if (manualSrc) {
    const idx = [...document.querySelectorAll('.psrc:checked')].map(c => c.dataset.i);
    fd.append('primary_idx', idx.join(','));
  }
  fd.append('screenshots_fps', $('shots').value);
  fd.append('transcribe', $('transcribe').checked ? 'true' : 'false');
  fd.append('reasoning', $('reasoning').value);
  const rmax = $('reasoningMax').value.trim();
  if (rmax !== '') fd.append('reasoning_max_tokens', rmax);
  if ($('pdfMode').value === 'file') fd.append('pdf_as_file', 'true');
  const pdfEng = $('pdfEngine').value;
  if (pdfEng && pdfEng !== 'auto') fd.append('pdf_engine', pdfEng);
  if ($('tonly').checked) fd.append('transcribe_only', 'true');
  $('go').disabled = true;
  const r = await fetch('/api/jobs', { method: 'POST', body: fd });
  $('go').disabled = false;
  if (!r.ok) { alert('Upload failed: ' + await r.text()); return; }
  const { id } = await r.json();
  currentId = id;
  $('jobCard').classList.remove('hidden');
  $('jobId').textContent = id;
  clearInterval(timer);
  timer = setInterval(() => poll(id), 2500);
  poll(id);
};

$('fix').onclick = async () => {
  if (!currentId) return;
  const fd = new FormData();
  fd.append('api_key', key.value || '');
  fd.append('model', selectedModel());
  fd.append('base_url', $('baseUrl').value);
  fd.append('reasoning', $('reasoning').value);
  const rmax = $('reasoningMax').value.trim();
  if (rmax !== '') fd.append('reasoning_max_tokens', rmax);
  const r = await fetch('/api/jobs/' + currentId + '/fix', { method: 'POST', body: fd });
  if (!r.ok) { alert('Fix failed to start: ' + await r.text()); return; }
  clearInterval(timer);
  timer = setInterval(() => poll(currentId), 2500);
};

async function poll(id) {
  const r = await fetch('/api/jobs/' + id);
  const j = await r.json();
  $('status').textContent = j.status;
  $('log').textContent = j.log || '';
  if (j.usage && j.usage.calls) {
    const u = j.usage, mods = Object.entries(u.by_modality || {}).map(([k, v]) => `${k}:${v}`).join(' ');
    const c = j.cost || {};
    $('usage').textContent = `tokens: in ${u.prompt} out ${u.output}` +
      (u.thoughts ? ` (of which ${u.thoughts} reasoning)` : '') +
      ` (${mods || 'no breakdown'}) · ` +
      (c.total_usd != null ? `est. $${c.total_usd} (${c.note})` : (c.note || 'cost n/a'));
  }
  $('compileErrors').textContent = j.compile_errors || '—';
  $('fix').classList.toggle('hidden', !(j.status === 'done_no_pdf' || j.status === 'fixing'));
  $('downloads').innerHTML = (j.artifacts || []).map(a =>
    `<a href="${a.url}" download>⬇ ${a.name} (${(a.size/1024).toFixed(0)} KB)</a>`).join('')
    || '<span class="hint">no artifacts yet</span>';
  const tex = (j.artifacts || []).find(a => a.name.endsWith('.tex'));
  if (tex && !$('texPreview').dataset.loaded) {
    const t = await fetch(tex.url).then(x => x.text());
    $('texPreview').textContent = t.slice(0, 6000);
    if (j.status.startsWith('done')) $('texPreview').dataset.loaded = '1';
  }
  if (j.status.startsWith('done') || j.status.startsWith('error')) clearInterval(timer);
}
