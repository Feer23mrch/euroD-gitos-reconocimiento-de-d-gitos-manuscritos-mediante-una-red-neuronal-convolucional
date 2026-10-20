"""Acceso de solo lectura a las particiones originales MNIST del proyecto."""
import base64
from functools import lru_cache
import io

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from motor import descargar_datos


@lru_cache(maxsize=1)
def particiones():
    with np.load(descargar_datos(), allow_pickle=False) as datos:
        x, y = datos['x_train'], datos['y_train']
        test_x, test_y = datos['x_test'], datos['y_test']
    entrenamiento, validacion = train_test_split(np.arange(len(y)), test_size=5000,
                                                 stratify=y, random_state=42)
    return {'entrenamiento': (x, y, entrenamiento, 'train'),
            'validacion': (x, y, validacion, 'train'),
            'prueba': (test_x, test_y, np.arange(len(test_y)), 'test')}


def consultar(parte='prueba', digito=None, pagina=0):
    conjuntos = particiones()
    if parte not in conjuntos or pagina < 0 or (digito is not None and digito not in range(10)):
        raise ValueError('Selecciona una partición, un dígito y una página válidos.')
    imagenes, etiquetas, indices, origen = conjuntos[parte]
    if digito is not None:
        indices = indices[etiquetas[indices] == digito]
    tamano = 30
    paginas = max(1, (len(indices) + tamano - 1) // tamano)
    if pagina >= paginas:
        raise ValueError('Esa página no existe.')
    ejemplos = []
    for indice in indices[pagina * tamano:(pagina + 1) * tamano]:
        buffer = io.BytesIO()
        Image.fromarray(imagenes[indice]).save(buffer, format='PNG')
        ejemplos.append({'indice': int(indice), 'etiqueta': int(etiquetas[indice]),
                         'origen': origen,
                         'imagen': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()})
    return {'parte': parte, 'digito': digito, 'pagina': pagina, 'paginas': paginas,
            'total': len(indices), 'ejemplos': ejemplos,
            'distribucion': np.bincount(etiquetas[conjuntos[parte][2]], minlength=10).tolist()}
