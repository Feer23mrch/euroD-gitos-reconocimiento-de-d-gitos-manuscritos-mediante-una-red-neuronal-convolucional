'use strict';
const $ = id => document.getElementById(id);
const canvas = $('drawing');
const ctx = canvas.getContext('2d');
let mode = 'draw', hasDrawing = false, upload = null, drawing = false;
let history = [], previous = null, revision = 0, busy = false, exampleIndex = 0;
let totalExamples = 10000;
const percent = value => new Intl.NumberFormat('es', {style: 'percent', maximumFractionDigits: 1}).format(value);
function blankCanvas() {ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, 320, 320);}
blankCanvas();
function controls() {
  $('recognize').disabled = busy || (mode === 'draw' ? !hasDrawing : !upload);
  $('recognize').firstChild.textContent = busy ? 'Reconociendo… ' : 'Reconocer dígito ';
  $('undo').disabled = history.length === 0 || busy;
  $('example').disabled = busy;
  $('draw-hint').hidden = hasDrawing;
  document.querySelector('.result-card').setAttribute('aria-busy', String(busy));
}
function invalidate() {
  revision++;
  busy = false;
  $('result').hidden = true;
  $('empty-result').hidden = false;
  $('result-badge').textContent = 'En espera';
  $('status').textContent = '';
  controls();
}
function position(e) {
  const rect = canvas.getBoundingClientRect();
  const size = Math.min(rect.width, rect.height);
  return [(e.clientX - rect.left - (rect.width - size) / 2) * 320 / size,
          (e.clientY - rect.top - (rect.height - size) / 2) * 320 / size];
}
canvas.addEventListener('pointerdown', e => {
  if (e.button !== 0 || drawing) return;
  canvas.setPointerCapture(e.pointerId);
  history.push({image: ctx.getImageData(0, 0, 320, 320), hasDrawing});
  if (history.length > 30) history.shift();
  drawing = true;
  previous = position(e);
  ctx.fillStyle = '#172238';
  ctx.beginPath();ctx.arc(...previous, 10, 0, Math.PI * 2);ctx.fill();
  hasDrawing = true;
  invalidate();
});
canvas.addEventListener('pointermove', e => {
  if (!drawing) return;
  const point = position(e);
  ctx.strokeStyle = '#172238';ctx.lineWidth = 20;ctx.lineCap = 'round';ctx.lineJoin = 'round';
  ctx.beginPath();ctx.moveTo(...previous);ctx.lineTo(...point);ctx.stroke();
  previous = point;
});
for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) {
  canvas.addEventListener(event, () => {drawing = false; previous = null;});
}
$('undo').addEventListener('click', () => {
  const last = history.pop();
  if (!last) return;
  ctx.putImageData(last.image, 0, 0);hasDrawing = last.hasDrawing;invalidate();
});
$('clear').addEventListener('click', () => {
  blankCanvas();hasDrawing = false;history = [];upload = null;drawing = false;
  $('file').value = '';$('upload-preview').hidden = true;$('upload-preview').removeAttribute('src');
  $('filename').textContent = 'Usa un número centrado y un fondo uniforme.';
  invalidate();
});
function setMode(next) {
  mode = next;
  $('draw-area').hidden = mode !== 'draw';$('upload-area').hidden = mode !== 'upload';
  for (const [id, active] of [['dibujar', mode === 'draw'], ['subir', mode === 'upload']]) {
    $(id).classList.toggle('selected', active);$(id).setAttribute('aria-pressed', String(active));
  }
  invalidate();
}
$('dibujar').addEventListener('click', () => setMode('draw'));
$('subir').addEventListener('click', () => setMode('upload'));
async function readFile(file) {
  if (!file) return;
  upload = null;$('upload-preview').hidden = true;
  invalidate();
  const current = revision;
  if (!['image/png', 'image/jpeg'].includes(file.type)) {
    $('status').textContent = 'Selecciona una imagen PNG o JPG.';return;
  }
  if (file.size > 5 * 1024 * 1024) {
    $('status').textContent = 'La imagen es demasiado grande. El límite es 5 MB.';return;
  }
  $('filename').textContent = 'Cargando imagen…';
  try {
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader();reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('No se pudo leer el archivo.'));reader.readAsDataURL(file);
    });
    const image = new Image();image.src = data;await image.decode();
    if (current !== revision) return;
    upload = data;$('upload-preview').src = data;$('upload-preview').hidden = false;
    $('filename').textContent = file.name;controls();
  } catch (error) {
    if (current !== revision) return;
    $('filename').textContent = 'Selecciona otra imagen.';
    $('status').textContent = 'No se pudo abrir la imagen. Prueba con otro archivo.';
  }
}
$('file').addEventListener('change', e => readFile(e.target.files[0]));
for (const event of ['dragover', 'dragenter']) $('drop-zone').addEventListener(event, e => {
  e.preventDefault();$('drop-zone').classList.add('dragging');
});
for (const event of ['dragleave', 'drop']) $('drop-zone').addEventListener(event, e => {
  e.preventDefault();$('drop-zone').classList.remove('dragging');
  if (event === 'drop') readFile(e.dataTransfer.files[0]);
});
async function api(path, options) {
  let response;
  try {response = await fetch(path, options);} catch (_) {throw new Error('No hay conexión con el programa. Comprueba que el servidor siga abierto.');}
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'No se pudo completar la operación.');
  return data;
}
function showResult(data, example = false) {
  $('empty-result').hidden = true;$('result').hidden = false;
  const best = data.predicciones[0];
  $('digit').textContent = best.digito;
  $('probability').textContent = `${percent(best.probabilidad)} de probabilidad estimada`;
  $('result-badge').textContent = example ? `Ejemplo · real: ${data.real}` : 'Resultado listo';
  $('confidence-note').textContent = example
    ? `Imagen del conjunto de prueba. ${best.digito === data.real ? 'La red acertó.' : 'La red confundió este dígito.'}`
    : best.probabilidad < 0.7 ? 'Hay varias posibilidades. Prueba un trazo más claro y centrado.' : 'Esta es la opción más probable para la red.';
  $('bars').replaceChildren();
  for (const prediction of data.predicciones.slice(1, 4)) {
    const row = document.createElement('div');row.className = 'bar-row';
    const label = document.createElement('strong');label.textContent = prediction.digito;
    const track = document.createElement('div');track.className = 'bar-track';
    const fill = document.createElement('div');fill.className = 'bar-fill';fill.style.width = `${prediction.probabilidad * 100}%`;
    const value = document.createElement('span');value.textContent = percent(prediction.probabilidad);
    track.append(fill);row.append(label, track, value);$('bars').append(row);
  }
  $('pixels').width = data.pixeles[0].length;
  $('pixels').height = data.pixeles.length;
  const pixels = $('pixels').getContext('2d');
  data.pixeles.forEach((row, y) => row.forEach((value, x) => {
    const tone = Math.round(Math.min(1, Math.max(0, value)) * 255);
    pixels.fillStyle = `rgb(${tone},${tone},${tone})`;pixels.fillRect(x, y, 1, 1);
  }));
  $('status').textContent = '';
}
async function recognize(example = false) {
  if (busy) return;
  const current = ++revision;
  busy = true;controls();$('status').textContent = 'La red está analizando la imagen…';
  try {
    const data = example
      ? await api(`/api/ejemplo/${exampleIndex % totalExamples}`)
      : await api('/api/reconocer', {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({imagen: mode === 'draw' ? canvas.toDataURL('image/png') : upload})});
    if (current !== revision) return;
    if (example) exampleIndex++;
    showResult(data, example);
  } catch (error) {
    if (current === revision) {$('status').textContent = error.message;$('result-badge').textContent = 'Inténtalo de nuevo';}
  } finally {
    if (current === revision) {busy = false;controls();}
  }
}
$('recognize').addEventListener('click', () => recognize());
$('example').addEventListener('click', () => recognize(true));
let evaluationLoaded = false;
async function loadEvaluation() {
  if (evaluationLoaded) return;
  $('evaluation-status').textContent = 'Cargando evaluación…';
  try {
    const data = await api('/api/evaluacion');
    $('accuracy').textContent = percent(data.exactitud);$('test-total').textContent = data.total;totalExamples = data.total;
    const table = $('matrix');
    const head = document.createElement('thead'), header = document.createElement('tr');
    for (const label of ['Real ↓', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) {
      const cell = document.createElement('th');cell.scope = 'col';cell.textContent = label;header.append(cell);
    }
    head.append(header);table.append(head);
    const body = document.createElement('tbody');
    data.matriz.forEach((row, actual) => {
      const tr = document.createElement('tr'), title = document.createElement('th');title.scope = 'row';title.textContent = actual;tr.append(title);
      row.forEach((value, predicted) => {
        const td = document.createElement('td');td.textContent = value;
        if (actual === predicted) td.className = 'diagonal';tr.append(td);
      });body.append(tr);
    });table.append(body);
    evaluationLoaded = true;$('evaluation-status').textContent = '';
  } catch (error) {$('evaluation-status').textContent = `${error.message} Vuelve a esta sección para reintentar.`;}
}
function section(evaluate) {
  $('probar-panel').hidden = evaluate;$('evaluar-panel').hidden = !evaluate;
  for (const [id, active] of [['probar-tab', !evaluate], ['evaluar-tab', evaluate]]) {
    $(id).classList.toggle('active', active);$(id).setAttribute('aria-pressed', String(active));
  }
  if (evaluate) loadEvaluation();
}
$('probar-tab').addEventListener('click', () => section(false));
$('evaluar-tab').addEventListener('click', () => section(true));
controls();
