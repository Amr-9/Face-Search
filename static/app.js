// Enable dynamic API path based on window location so it works on any host
const API = window.location.origin;

/* ──────────────── Tabs ──────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', ['index', 'search', 'persons', 'backup'][i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel' + name.charAt(0).toUpperCase() + name.slice(1)).classList.add('active');
  if (name === 'persons') loadPersons();
}

/* ──────────────── Image Preview (single) ──────────────── */
function previewImage(input, previewId) {
  const file = input.files[0];
  if (!file) return;
  const wrap = document.getElementById(previewId + 'Wrap');
  const img = document.getElementById(previewId);
  const reader = new FileReader();
  reader.onload = e => {
    img.src = e.target.result;
    wrap.classList.add('show');
  };
  reader.readAsDataURL(file);
}

/* ──────────────── Image Preview (multiple) ──────────────── */
function previewMultipleImages(input, gridId) {
  const grid = document.getElementById(gridId);
  grid.innerHTML = '';
  const files = Array.from(input.files);
  if (!files.length) { grid.style.display = 'none'; return; }
  grid.style.display = 'flex';
  grid.className = 'preview-grid';
  files.forEach(file => {
    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    const img = document.createElement('img');
    img.alt = file.name;
    const label = document.createElement('div');
    label.className = 'thumb-label';
    label.textContent = file.name;
    const reader = new FileReader();
    reader.onload = e => { img.src = e.target.result; };
    reader.readAsDataURL(file);
    thumb.appendChild(img);
    thumb.appendChild(label);
    grid.appendChild(thumb);
  });
}

/* ── Drag & Drop ── */
function setupDrop(dzId, inputId, previewFn) {
  const dz = document.getElementById(dzId);
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('dragover');
    const inp = document.getElementById(inputId);
    const images = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    if (!images.length) return;
    const dt = new DataTransfer();
    images.forEach(f => dt.items.add(f));
    inp.files = dt.files;
    previewFn(inp);
  });
}
setupDrop('dzIndex', 'fileIndex', inp => previewMultipleImages(inp, 'previewIndexGrid'));
setupDrop('dzSearch', 'fileSearch', inp => previewImage(inp, 'previewSearch'));

/* ──────────────── Backup & Restore ──────────────── */
async function downloadBackup() {
  setLoading('btnBackupDown', 'spinBackupDown', true);
  const out = document.getElementById('backupDownResult');
  out.innerHTML = '';
  try {
    const res = await fetch(`${API}/backup/download`);
    if (!res.ok) throw new Error('Server returned an error.');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const cd = res.headers.get('Content-Disposition') || '';
    const match = cd.match(/filename=(.+)/);
    a.download = match ? match[1] : 'backup.zip';
    a.href = url;
    a.click();
    URL.revokeObjectURL(url);
    showAlert(out, 'success', '✅ Backup downloaded successfully.');
  } catch (e) {
    showAlert(out, 'error', '❌ ' + e.message);
  } finally {
    setLoading('btnBackupDown', 'spinBackupDown', false);
  }
}

function previewRestoreFile(input) {
  const file = input.files[0];
  const info = document.getElementById('restoreFileInfo');
  const wrap = document.getElementById('restoreModeWrap');
  const btn  = document.getElementById('btnRestore');
  if (!file) {
    info.style.display = 'none';
    wrap.style.display = 'none';
    btn.disabled = true;
    return;
  }
  info.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  info.style.display = 'block';
  wrap.style.display = 'block';
  btn.disabled = false;
  updateRestoreBtn();
}

function updateRestoreBtn() {
  const mode = document.querySelector('input[name="restoreMode"]:checked')?.value || 'replace';
  const btn  = document.getElementById('btnRestore');
  const lbl  = document.getElementById('btnRestoreLabel');
  const replaceCard = document.getElementById('modeReplaceCard');
  const mergeCard   = document.getElementById('modeMergeCard');

  if (mode === 'replace') {
    btn.className = 'btn btn-danger';
    lbl.textContent = '🗑️ Replace & Restore';
    replaceCard.style.borderColor = 'var(--danger)';
    replaceCard.style.background  = 'rgba(248,81,73,.07)';
    mergeCard.style.borderColor   = 'var(--border)';
    mergeCard.style.background    = 'transparent';
  } else {
    btn.className = 'btn btn-primary';
    lbl.textContent = '➕ Merge & Add';
    mergeCard.style.borderColor   = 'var(--secondary)';
    mergeCard.style.background    = 'rgba(63,185,80,.07)';
    replaceCard.style.borderColor = 'var(--border)';
    replaceCard.style.background  = 'transparent';
  }
}

async function restoreBackup() {
  const file = document.getElementById('fileRestore').files[0];
  const mode = document.querySelector('input[name="restoreMode"]:checked')?.value || 'replace';
  const out  = document.getElementById('backupRestoreResult');
  if (!file) return;

  const confirmMsg = mode === 'replace'
    ? '⚠️ This will DELETE all current data and restore from backup. Are you sure?'
    : '➕ Backup faces will be added on top of your current data. Continue?';
  if (!confirm(confirmMsg)) return;

  setLoading('btnRestore', 'spinRestore', true);
  out.innerHTML = '';
  const fd = new FormData();
  fd.append('backup_file', file);
  fd.append('mode', mode);
  try {
    const res  = await fetch(`${API}/backup/restore`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Restore failed.');
    const msg = mode === 'replace'
      ? `✅ Replaced successfully — ${data.total_indexed} face(s) loaded.`
      : `✅ Merged successfully — total ${data.total_indexed} face(s) now indexed.`;
    showAlert(out, 'success', msg);
    loadStats();
  } catch (e) {
    showAlert(out, 'error', '❌ ' + e.message);
  } finally {
    setLoading('btnRestore', 'spinRestore', false);
  }
}

/* ──────────────── Stats ──────────────── */
async function loadStats() {
  try {
    const res = await fetch(`${API}/faces/stats`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    document.getElementById('statTotal').textContent = data.total_indexed;
    document.getElementById('statStatus').textContent = '🟢 Online';
  } catch {
    document.getElementById('statStatus').textContent = '🔴 Disconnected';
  }
}

/* ──────────────── Index ──────────────── */
async function indexFace() {
  const files = Array.from(document.getElementById('fileIndex').files);
  const out = document.getElementById('indexResult');

  if (!files.length) return showAlert(out, 'error', 'Please select at least one image.');

  setLoading('btnIndex', 'spinIndex', true);
  out.innerHTML = '';

  const fd = new FormData();
  files.forEach(f => fd.append('images', f));

  try {
    const res = await fetch(`${API}/faces/index`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map(d => d.msg || JSON.stringify(d)).join(' | ')
        : (data.detail || 'Unknown error');
      throw new Error(detail);
    }
    let msg = `✅ Indexed <strong>${data.faces_found}</strong> face(s) from <strong>${data.images_processed}</strong> image(s).`;
    if (data.skipped && data.skipped.length) {
      msg += `<br>⚠️ Skipped <strong>${data.skipped.length}</strong>: ` +
        data.skipped.map(s => `${s.filename} (${s.reason})`).join(', ');
    }
    msg += `<br>Total indexed: <strong>${data.total_indexed}</strong>`;
    showAlert(out, 'success', msg);
    document.getElementById('fileIndex').value = '';
    const grid = document.getElementById('previewIndexGrid');
    grid.innerHTML = '';
    grid.style.display = 'none';
    loadStats();
  } catch (e) {
    showAlert(out, 'error', '❌ ' + e.message);
  } finally {
    setLoading('btnIndex', 'spinIndex', false);
  }
}

/* ──────────────── Search ──────────────── */
async function searchFace() {
  const file = document.getElementById('fileSearch').files[0];
  const topK = document.getElementById('topK').value || 9;
  const out = document.getElementById('searchResults');

  if (!file) return showAlert(out, 'error', 'Please select an image to search.');

  setLoading('btnSearch', 'spinSearch', true);
  out.innerHTML = '';

  const fd = new FormData();
  fd.append('image', file);
  fd.append('top_k', topK);

  try {
    const res = await fetch(`${API}/faces/search`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map(d => d.msg || JSON.stringify(d)).join(' | ')
        : (data.detail || 'Unknown error');
      throw new Error(detail);
    }

    if (!data.results || data.results.length === 0) {
      showAlert(out, 'info', 'No match found in the index.');
      return;
    }

    const header = document.createElement('p');
    header.style.cssText = 'color:var(--textMuted);font-size:.85rem;margin-bottom:12px;grid-column:1/-1;';
    header.textContent = `Face detection score: ${(data.query_det_score * 100).toFixed(1)}% | ${data.results.length} result(s)`;
    out.appendChild(header);

    data.results.forEach((r, i) => {
      const pct = Math.round(r.similarity * 100);
      const card = document.createElement('div');
      card.className = 'result-card';
      card.innerHTML = `
        <img src="${API}${r.image_path}" alt="${escHtml(r.name)}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22><rect fill=%22%231e2530%22 width=%22200%22 height=%22200%22 rx=%2212%22/><text x=%2250%25%22 y=%2255%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 font-size=%2260%22>👤</text></svg>'" />
        <div class="result-info">
          ${i === 0 ? '<div style="margin-bottom:4px;"><span class="badge-top">Best Match</span></div>' : ''}
          <div class="result-name" style="font-size: 1.15rem; margin-bottom: 2px;">
            Face #${r.person_id}
          </div>
          <div class="result-meta" style="font-size: 0.9rem;">${new Date(r.created_at).toLocaleDateString('en-US')}</div>
          <div class="similarity-bar-wrap" style="margin-top:8px;">
            <div class="similarity-bar">
              <div class="similarity-fill" style="width:${pct}%"></div>
            </div>
            <div class="similarity-pct">${pct}%</div>
          </div>
        </div>
      `;
      out.appendChild(card);
    });

  } catch (e) {
    showAlert(out, 'error', '❌ ' + e.message);
  } finally {
    setLoading('btnSearch', 'spinSearch', false);
  }
}

/* ──────────────── Persons ──────────────── */
const PAGE_SIZE  = 200;
let currentPage  = 1;
let sortOrder    = 'desc';  // 'desc' = newest first
let totalPersons = 0;

function toggleSort() {
  sortOrder = (sortOrder === 'desc') ? 'asc' : 'desc';
  document.getElementById('btnSort').textContent =
    sortOrder === 'desc' ? '🕓 Newest First ↓' : '🕐 Oldest First ↑';
  currentPage = 1;
  loadPersons();
}

function changePage(delta) {
  const totalPages = Math.max(1, Math.ceil(totalPersons / PAGE_SIZE));
  const newPage = currentPage + delta;
  if (newPage < 1 || newPage > totalPages) return;
  currentPage = newPage;
  loadPersons();
}

async function loadPersons() {
  const grid = document.getElementById('personsGrid');
  const bar  = document.getElementById('paginationBar');
  grid.innerHTML = '<div class="empty-state">⏳ Loading…</div>';
  bar.style.display = 'none';

  const skip = (currentPage - 1) * PAGE_SIZE;
  try {
    const res  = await fetch(`${API}/faces?skip=${skip}&limit=${PAGE_SIZE}&order=${sortOrder}`);
    const data = await res.json();
    totalPersons = data.total;
    const totalPages = Math.max(1, Math.ceil(totalPersons / PAGE_SIZE));

    grid.innerHTML = '';
    if (!data.persons || data.persons.length === 0) {
      grid.innerHTML = '<div class="empty-state">📭 No indexed faces yet.</div>';
      return;
    }
    data.persons.forEach(p => {
      const tile = document.createElement('div');
      tile.className = 'person-tile';
      const date = new Date(p.created_at).toLocaleDateString('en-US');
      tile.innerHTML = `
        <button class="del-btn" title="Delete" onclick="deletePerson(${p.id}, this)">✕</button>
        <img src="${API}${p.image_path}" alt="${escHtml(p.name)}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22160%22 height=%22160%22><rect fill=%22%231e2530%22 width=%22160%22 height=%22160%22 rx=%2212%22/><text x=%2250%25%22 y=%2255%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 font-size=%2260%22>👤</text></svg>'" />
        <div class="p-name" style="font-size: 1.05rem; margin-bottom: 4px;">Face #${p.id}</div>
        <div class="p-date">${date}</div>
      `;
      grid.appendChild(tile);
    });

    if (totalPersons > PAGE_SIZE) {
      document.getElementById('pageInfo').textContent =
        `Page ${currentPage} of ${totalPages} (${totalPersons} total)`;
      document.getElementById('btnPrev').disabled = (currentPage === 1);
      document.getElementById('btnNext').disabled = (currentPage === totalPages);
      bar.style.display = 'flex';
    }
  } catch {
    grid.innerHTML = '<div class="empty-state">❌ Failed to connect to server.</div>';
  }
}

async function deletePerson(id, btn) {
  if (!confirm(`Are you sure you want to delete face #${id}?`)) return;
  btn.disabled = true;
  try {
    const res = await fetch(`${API}/faces/${id}`, { method: 'DELETE' });
    if (!res.ok) { const d = await res.json(); throw new Error(d.detail); }
    btn.closest('.person-tile').remove();
    loadStats();
  } catch (e) {
    alert('Error: ' + e.message);
    btn.disabled = false;
  }
}

/* ──────────────── Helpers ──────────────── */
function showAlert(container, type, html) {
  const div = document.createElement('div');
  div.className = `alert alert-${type}`;
  div.innerHTML = html;
  container.appendChild(div);
}

function setLoading(btnId, spinId, loading) {
  document.getElementById(btnId).disabled = loading;
  document.getElementById(spinId).style.display = loading ? 'block' : 'none';
}

function escHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* ──────────────── Init ──────────────── */
loadStats();
setInterval(loadStats, 15000);
