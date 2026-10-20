"""Genera figuras y tablas del informe desde datos y métricas reales; no entrena."""
from pathlib import Path
import json
import os
import sys

BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(BASE / 'docs' / '.matplotlib'))
sys.path.insert(0, str(BASE))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from motor import cargar, datos_mnist, preparar
from PIL import Image

SALIDA = BASE / 'docs' / 'ieee'
FIGURAS = SALIDA / 'figuras'
FIGURAS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
metricas = json.loads((BASE / 'metricas_entrenamiento.json').read_text())
m = np.array(metricas['matriz'])
historial = metricas['historial']
epocas = [e['epoca'] for e in historial]
fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.8), layout='constrained')
for clave, titulo, color in [('entrenamiento', 'Entrenamiento', '#345ee8'), ('validacion', 'Validación', '#d35c26')]:
    ax[0].plot(epocas, [100 * e['exactitud_' + clave] for e in historial], '.-', label=titulo, color=color)
    ax[1].plot(epocas, [e['perdida_' + clave] for e in historial], '.-', label=titulo, color=color)
ax[0].set_ylabel('Exactitud (%)');ax[0].legend(fontsize=7);ax[1].set_ylabel('Entropía cruzada')
for a in ax:
    a.set_xlabel('Época');a.set_xticks([1, 2, 4, 6, 8, 10]);a.grid(alpha=.2)
fig.savefig(FIGURAS / 'aprendizaje.pdf');plt.close(fig)
fig, ax = plt.subplots(figsize=(3.45, 3.25), layout='constrained')
ax.imshow(m / m.sum(axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(m[i, j]), ha='center', va='center', fontsize=6, color='white' if i == j else '#24364a')
ax.set_xticks(range(10));ax.set_yticks(range(10));ax.set_xlabel('Predicción');ax.set_ylabel('Etiqueta real')
fig.savefig(FIGURAS / 'confusion.pdf');plt.close(fig)
with np.load(BASE / 'datos' / 'mnist.npz', allow_pickle=False) as datos:
    raw, labels = datos['x_test'], datos['y_test']
fig, axs = plt.subplots(2, 10, figsize=(7.1, 1.65), layout='constrained')
for digit in range(10):
    indices = np.flatnonzero(labels == digit)[:2]
    for row, indice in enumerate(indices):
        axs[row, digit].imshow(raw[indice], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axs[row, digit].axis('off')
        if row == 0:
            axs[row, digit].set_title(str(digit))
fig.savefig(FIGURAS / 'mnist.pdf');plt.close(fig)
# Selección determinista: los primeros errores según el orden oficial de prueba.
_, prepared, _, y = datos_mnist()
model = cargar();pred = model.predict(prepared.astype(np.float32) / 255)
assert np.array_equal(np.bincount(y[pred == y], minlength=10), np.diag(m))
wrong = np.flatnonzero(pred != y)[:6]
fig, axs = plt.subplots(2, 3, figsize=(3.45, 2.6), layout='constrained')
for ax, indice in zip(axs.flat, wrong):
    ax.imshow(prepared[indice], cmap='gray', vmin=0, vmax=255, interpolation='nearest');ax.axis('off')
    ax.set_title(f'Real {y[indice]} / pred. {pred[indice]}\nÍndice {indice}', fontsize=7)
fig.savefig(FIGURAS / 'errores.pdf');plt.close(fig)
# La tabla usa la matriz de prueba, no probabilidades ni redondeos de pantalla.
precision = np.diag(m) / m.sum(axis=0)
recall = np.diag(m) / m.sum(axis=1)
f1 = 2 * precision * recall / (precision + recall)
rows = [f'{d} & {precision[d]:.4f} & {recall[d]:.4f} & {f1[d]:.4f} & {m[d].sum()} \\\\' for d in range(10)]
rows += [r'\hline', f'Macro & {precision.mean():.4f} & {recall.mean():.4f} & {f1.mean():.4f} & {m.sum()} \\\\']
cabecera = r'\begin{tabular}{crrrr}' + '\n' + r'\hline' + '\n' + r'Dígito & Precisión & Exhaustividad & F1 & Soporte \\' + '\n' + r'\hline' + '\n'
(SALIDA / 'tabla_metricas.tex').write_text(cabecera + '\n'.join(rows) + '\n' + r'\hline' + '\n' + r'\end{tabular}' + '\n')
params = sum(p.numel() for p in model.red.parameters())
params_tex = f'{params:,}'.replace(',', r'\,')
(SALIDA / 'valores.tex').write_text(
    f'\\newcommand{{\\ExactitudPrueba}}{{{100 * metricas["exactitud_prueba"]:.2f}}}\n'
    f'\\newcommand{{\\NumeroErrores}}{{{int(m.sum() - np.trace(m))}}}\n'
    f'\\newcommand{{\\ParametrosModelo}}{{{params_tex}}}\n'
    f'\\newcommand{{\\FMacro}}{{{f1.mean():.4f}}}\n')
print(f'Figuras y tablas generadas. Parámetros: {params}, F1 macro: {f1.mean():.6f}, errores: {len(np.flatnonzero(pred != y))}.')
