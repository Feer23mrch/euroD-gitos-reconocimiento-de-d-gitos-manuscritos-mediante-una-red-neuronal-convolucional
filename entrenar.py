"""Entrena CNN con validación independiente: python entrenar.py --epocas 10."""
import argparse
import json
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from motor import (RedDigitos, Clasificador, datos_mnist, evaluar, RUTA_MODELO,
                   RUTA_METRICAS, VERSION_PREPARACION)


def aumentar(x):
    """Transformaciones distintas en cada época, solo sobre entrenamiento."""
    n = len(x)
    angulo = (torch.rand(n, device=x.device) * 2 - 1) * 0.21
    escala = 0.88 + torch.rand(n, device=x.device) * 0.24
    sesgo = (torch.rand(n, device=x.device) * 2 - 1) * 0.15
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = angulo.cos() * escala
    theta[:, 0, 1] = -angulo.sin() + sesgo
    theta[:, 1, 0] = angulo.sin()
    theta[:, 1, 1] = angulo.cos() * escala
    theta[:, :, 2] = (torch.rand(n, 2, device=x.device) * 2 - 1) * 0.11
    x = F.grid_sample(x, F.affine_grid(theta, x.size(), align_corners=False), align_corners=False)
    # Variaciones suaves de grosor sin destruir trazos delgados.
    selector = torch.rand(n, 1, 1, 1, device=x.device)
    grueso = F.max_pool2d(x, 3, stride=1, padding=1)
    fino = -F.max_pool2d(-x, 3, stride=1, padding=1)
    x = torch.where(selector < 0.15, 0.6 * x + 0.4 * grueso, x)
    x = torch.where(selector > 0.85, 0.7 * x + 0.3 * fino, x)
    return x.clamp(0, 1)


def entrenar_modelo(epocas=10, dispositivo='auto'):
    if epocas < 1:
        raise ValueError('El número de épocas debe ser positivo.')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if dispositivo == 'auto':
        dispositivo = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(dispositivo)
    imagenes, _, etiquetas, _ = datos_mnist()
    indices_tr, indices_val = train_test_split(np.arange(len(etiquetas)), test_size=5000,
                                               stratify=etiquetas, random_state=42)
    x = torch.from_numpy(imagenes.copy()).unsqueeze(1).float() / 255
    y = torch.from_numpy(etiquetas.astype(np.int64))
    x_val, y_val = x[indices_val].to(device), y[indices_val].to(device)
    red = RedDigitos().to(device)
    optimizador = torch.optim.AdamW(red.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=epocas, eta_min=0.0001)
    perdida = nn.CrossEntropyLoss()
    mejor = float('inf');mejores_pesos = None;historial = [];inicio = time.monotonic()
    print(f'CNN · {device} · 55 000 entrenamiento / 5 000 validación / 10 000 prueba', flush=True)
    for epoca in range(1, epocas + 1):
        red.train();suma = 0.;aciertos = 0
        orden = np.random.permutation(indices_tr)
        for offset in range(0, len(orden), 256):
            ids = orden[offset:offset + 256]
            lote, objetivos = aumentar(x[ids].to(device)), y[ids].to(device)
            optimizador.zero_grad(set_to_none=True)
            logits = red(lote);loss = perdida(logits, objetivos)
            loss.backward();optimizador.step()
            suma += loss.item() * len(ids)
            aciertos += (logits.argmax(1) == objetivos).sum().item()
        red.eval();val_loss = 0.;val_aciertos = 0
        with torch.inference_mode():
            for offset in range(0, len(x_val), 256):
                logits = red(x_val[offset:offset + 256]);objetivos = y_val[offset:offset + 256]
                val_loss += perdida(logits, objetivos).item() * len(objetivos)
                val_aciertos += (logits.argmax(1) == objetivos).sum().item()
        val_loss /= len(x_val)
        registro = {'epoca': epoca, 'perdida_entrenamiento': suma / len(orden),
                    'exactitud_entrenamiento': aciertos / len(orden),
                    'perdida_validacion': val_loss, 'exactitud_validacion': val_aciertos / len(x_val)}
        historial.append(registro)
        if val_loss < mejor:
            mejor = val_loss;mejor_epoca = epoca
            mejores_pesos = {k: v.detach().cpu().clone() for k, v in red.state_dict().items()}
        scheduler.step()
        print(f'Época {epoca}/{epocas}: entrenamiento {aciertos / len(orden):.2%} · '
              f'validación {val_aciertos / len(x_val):.2%} · pérdida val {val_loss:.4f} · '
              f'{time.monotonic() - inicio:.0f}s', flush=True)
    red = RedDigitos();red.load_state_dict(mejores_pesos)
    modelo = Clasificador(red)
    # El conjunto de prueba solo se utiliza después de seleccionar el modelo por validación.
    info = evaluar(modelo)
    metricas = {'modelo': 'CNN 24-48-64 + 128', 'dataset': 'MNIST', 'semilla': 42,
                'version_preparacion': VERSION_PREPARACION, 'entrenamiento': 55000,
                'validacion': 5000, 'prueba': 10000, 'epocas': epocas, 'mejor_epoca': mejor_epoca,
                'exactitud_prueba': float(info['exactitud']), 'matriz': info['matriz'].tolist(),
                'historial': historial, 'segundos': round(time.monotonic() - inicio, 1),
                'fecha_utc': datetime.now(timezone.utc).isoformat(), 'torch': str(torch.__version__)}
    temporal = RUTA_MODELO.with_suffix('.tmp.pt')
    torch.save({'pesos': mejores_pesos, 'version_preparacion': VERSION_PREPARACION,
                'metricas': metricas}, temporal)
    temporal.replace(RUTA_MODELO)
    RUTA_METRICAS.write_text(json.dumps(metricas, indent=2, ensure_ascii=False) + '\n')
    print(f'Modelo guardado: {RUTA_MODELO}\nPrueba final MNIST: {info["exactitud"]:.2%}', flush=True)
    return modelo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epocas', type=int, default=10)
    parser.add_argument('--dispositivo', choices=['auto', 'cpu', 'mps'], default='auto')
    args = parser.parse_args()
    entrenar_modelo(args.epocas, args.dispositivo)
