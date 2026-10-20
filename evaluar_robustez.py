"""Compara ambas redes sobre las MISMAS imágenes, sin entrenar ni ajustar parámetros."""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from motor import BASE, RUTA_DATOS, cargar, preparar


def preparar_anterior(imagen):
    """Preparación original, preservada exclusivamente para una comparación reproducible."""
    imagen = imagen.convert('L')
    arr = np.asarray(imagen)
    if np.median(arr) > 127:
        imagen = Image.fromarray(255 - arr)
    imagen = imagen.point(lambda v: 0 if v < 18 else v)
    caja = imagen.getbbox()
    if caja is None:
        return np.zeros((1, 64))
    recorte = imagen.crop(caja)
    w, h = recorte.size
    factor = min(6 / w, 6 / h)
    recorte = recorte.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.Resampling.LANCZOS)
    lienzo = Image.new('L', (8, 8), 0)
    lienzo.paste(recorte, ((8 - recorte.width) // 2, (8 - recorte.height) // 2))
    return np.asarray(lienzo, dtype=float).reshape(1, 64) * 16 / 255


def variante(imagen, indice):
    """Simula una imagen subida: desplazamiento, escala, giro, grosor y polaridad."""
    rng = np.random.default_rng(1000 + indice)
    imagen = imagen.resize((112, 112), Image.Resampling.BICUBIC)
    imagen = imagen.rotate(float(rng.uniform(-12, 12)), Image.Resampling.BICUBIC, expand=True)
    if indice % 3 == 0:
        imagen = imagen.filter(ImageFilter.MaxFilter(3))
    elif indice % 3 == 1:
        imagen = imagen.filter(ImageFilter.MinFilter(3))
    lado = int(rng.integers(145, 220))
    imagen = imagen.resize((lado, lado), Image.Resampling.LANCZOS)
    lienzo = Image.new('L', (320, 320), 0)
    lienzo.paste(imagen, (int(rng.integers(10, 310 - lado)), int(rng.integers(10, 310 - lado))))
    if indice % 2 == 0:
        lienzo = Image.fromarray(255 - np.asarray(lienzo))
    return lienzo


def main(solo_anterior=False):
    with np.load(RUTA_DATOS, allow_pickle=False) as datos:
        imagenes, etiquetas = datos['x_test'], datos['y_test']
    _, indices = train_test_split(np.arange(len(etiquetas)), test_size=1000, stratify=etiquetas, random_state=123)
    casos = [('MNIST original (10000)', [Image.fromarray(x) for x in imagenes], etiquetas),
             ('Variantes de imagen subida (1000)', [variante(Image.fromarray(imagenes[i]), int(i)) for i in indices], etiquetas[indices])]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        anterior = joblib.load(BASE / 'modelo_digitos.joblib')
    nuevo = None if solo_anterior else cargar()
    informe = {}
    for nombre, ejemplos, reales in casos:
        print(f'Evaluando {nombre}…', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pred_anterior = anterior.predict(np.concatenate([preparar_anterior(im) for im in ejemplos]))
        resultado = {'muestras': len(reales), 'mlp_anterior': float(accuracy_score(reales, pred_anterior))}
        if nuevo:
            vectores = [preparar(im) for im in ejemplos]
            assert all(v is not None for v in vectores)
            pred_nuevo = nuevo.predict(np.concatenate(vectores))
            resultado['cnn_nueva'] = float(accuracy_score(reales, pred_nuevo))
            resultado['por_digito_cnn'] = {str(d): float(accuracy_score(reales[reales == d], pred_nuevo[reales == d])) for d in range(10)}
        informe[nombre] = resultado
        print(json.dumps(resultado, ensure_ascii=False), flush=True)
    salida = BASE / ('comparacion_anterior.json' if solo_anterior else 'comparacion_modelos.json')
    salida.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solo-anterior', action='store_true')
    main(parser.parse_args().solo_anterior)
