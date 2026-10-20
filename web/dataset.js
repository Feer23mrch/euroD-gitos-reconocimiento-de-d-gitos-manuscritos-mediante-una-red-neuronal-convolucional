'use strict';
const $ = id => document.getElementById(id);
let page = 0, pages = 1, requestVersion = 0;
async function load() {
  const version = ++requestVersion;
  $('gallery').setAttribute('aria-busy', 'true');
  $('dataset-status').textContent = 'Cargando imágenes originales…';
  $('prev').disabled = true;$('next').disabled = true;$('retry').hidden = true;
  $('gallery').replaceChildren();$('count').textContent = '';$('page').textContent = '';
  const query = new URLSearchParams({parte: $('split').value, digito: $('digit-filter').value, pagina: page});
  try {
    const response = await fetch(`/api/dataset?${query}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'No se pudo cargar el dataset.');
    if (version !== requestVersion) return;
    pages = data.paginas;
    for (const sample of data.ejemplos) {
      const card = document.createElement('figure');card.className = 'digit-card';
      const image = document.createElement('img');image.src = sample.imagen;image.alt = `Dígito ${sample.etiqueta}, índice ${sample.indice} de ${sample.origen}`;
      image.width = 84;image.height = 84;
      const caption = document.createElement('figcaption');caption.textContent = `Etiqueta: ${sample.etiqueta}`;
      const index = document.createElement('small');index.textContent = `x_${sample.origen}[${sample.indice}]`;
      const download = document.createElement('a');download.href = sample.imagen;
      download.download = `mnist_${sample.origen}_${sample.indice}_digito_${sample.etiqueta}.png`;download.textContent = 'Guardar PNG';
      caption.append(index, download);card.append(image, caption);$('gallery').append(card);
    }
    $('count').textContent = `${data.total.toLocaleString('es')} imágenes en esta selección`;
    $('page').textContent = `Página ${page + 1} de ${pages}`;
    $('dataset-status').textContent = `Mostrando ${data.ejemplos.length} imágenes. Etiquetas originales de MNIST.`;
    $('prev').disabled = page === 0;$('next').disabled = page + 1 >= pages;
  } catch (error) {
    if (version !== requestVersion) return;
    $('dataset-status').textContent = 'No se pudo cargar el dataset. Comprueba que el servidor esté activo.';
    $('retry').hidden = false;
  } finally {if (version === requestVersion) $('gallery').setAttribute('aria-busy', 'false');}
}
for (const id of ['split','digit-filter']) $(id).addEventListener('change', () => {page = 0;load();});
$('prev').addEventListener('click', () => {if (page > 0) {page--;load();}});
$('next').addEventListener('click', () => {if (page + 1 < pages) {page++;load();}});
$('retry').addEventListener('click', load);
load();
