"""Red convolucional y preparación coherente de imágenes de 28 × 28."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import urllib.request

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
from torch import nn

BASE = Path(__file__).resolve().parent
RUTA_MODELO = BASE / 'modelo_digitos.pt'
RUTA_METRICAS = BASE / 'metricas_entrenamiento.json'
RUTA_DATOS = BASE / 'datos' / 'mnist.npz'
LADO = 28
VERSION_PREPARACION = 2
SHA_MNIST = '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'
torch.set_num_threads(4)


class RedDigitos(nn.Module):
    def __init__(self):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 10))

    def forward(self, x):
        return self.capas((x - 0.1307) / 0.3081)


class Clasificador:
    classes_ = np.arange(10)

    def __init__(self, red):
        self.red = red.cpu().eval()

    def predict_proba(self, imagenes):
        x = np.asarray(imagenes, dtype=np.float32).reshape(-1, 1, LADO, LADO)
        resultados = []
        with torch.inference_mode():
            for inicio in range(0, len(x), 256):
                lote = torch.from_numpy(x[inicio:inicio + 256].copy())
                resultados.append(self.red(lote).softmax(dim=1).numpy())
        return np.concatenate(resultados)

    def predict(self, imagenes):
        return self.classes_[self.predict_proba(imagenes).argmax(axis=1)]


def preparar(imagen):
    """Encuadra, conserva proporciones y centra por masa; salida [1, 784], 0..1."""
    imagen = ImageOps.exif_transpose(imagen)
    rgba = imagen.convert('RGBA')
    imagen = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('L')
    imagen.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    arr = np.asarray(imagen, dtype=np.float32)
    bordes = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]))
    fondo = float(np.median(bordes))
    arr = np.clip(fondo - arr if fondo > 127 else arr - fondo, 0, 255)
    pico = float(arr.max())
    if pico < 18:
        return None
    arr[arr < max(18, pico * 0.12)] = 0
    componentes, cantidad = ndimage.label(arr > 0)
    if cantidad:
        tamanos = np.bincount(componentes.ravel())
        tamanos[0] = 0
        conservar = tamanos >= max(2, tamanos.max() * 0.035)
        conservar[0] = False
        arr[~conservar[componentes]] = 0
    filas, columnas = np.nonzero(arr)
    if len(filas) < 2:
        return None
    recorte = arr[filas.min():filas.max() + 1, columnas.min():columnas.max() + 1]
    recorte = Image.fromarray(np.uint8(recorte / pico * 255))
    factor = 20 / max(recorte.size)
    recorte = recorte.resize(tuple(max(1, round(t * factor)) for t in recorte.size), Image.Resampling.LANCZOS)
    lienzo = Image.new('L', (LADO, LADO), 0)
    lienzo.paste(recorte, ((LADO - recorte.width) // 2, (LADO - recorte.height) // 2))
    pixeles = np.asarray(lienzo, dtype=np.float32)
    cy, cx = ndimage.center_of_mass(pixeles)
    pixeles = ndimage.shift(pixeles, (13.5 - cy, 13.5 - cx), order=1, mode='constant', cval=0)
    return (np.rint(np.clip(pixeles, 0, 255)) / 255).astype(np.float32).reshape(1, -1)


def descargar_datos():
    RUTA_DATOS.parent.mkdir(exist_ok=True)
    if not RUTA_DATOS.exists():
        temporal = RUTA_DATOS.with_suffix('.descarga')
        print('Descargando MNIST (11 MB)…', flush=True)
        urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', temporal)
        if hashlib.sha256(temporal.read_bytes()).hexdigest() != SHA_MNIST:
            temporal.unlink()
            raise ValueError('La descarga MNIST no coincide con su SHA-256.')
        temporal.replace(RUTA_DATOS)
    if hashlib.sha256(RUTA_DATOS.read_bytes()).hexdigest() != SHA_MNIST:
        raise ValueError('El archivo MNIST está dañado. Elimina datos/mnist.npz y vuelve a entrenar.')
    return RUTA_DATOS


@lru_cache(maxsize=1)
def datos_mnist():
    ruta = descargar_datos()
    cache = RUTA_DATOS.with_name(f'mnist_preparado_v{VERSION_PREPARACION}.npz')
    if not cache.exists():
        with np.load(ruta, allow_pickle=False) as datos:
            preparados = {}
            for parte in ['train', 'test']:
                print(f'Preparando MNIST {parte} con el mismo proceso que tus dibujos…', flush=True)
                imagenes = []
                for imagen in datos[f'x_{parte}']:
                    vector = preparar(Image.fromarray(imagen))
                    if vector is None:
                        raise ValueError('Una imagen MNIST quedó vacía durante la preparación.')
                    imagenes.append(np.rint(vector.reshape(LADO, LADO) * 255).astype(np.uint8))
                preparados[f'x_{parte}'] = np.stack(imagenes)
                preparados[f'y_{parte}'] = datos[f'y_{parte}']
        temporal = cache.with_name(cache.stem + '.tmp.npz')
        np.savez_compressed(temporal, **preparados)
        temporal.replace(cache)
    with np.load(cache, allow_pickle=False) as datos:
        return tuple(datos[nombre] for nombre in ['x_train', 'x_test', 'y_train', 'y_test'])


def datos_prueba():
    x_tr, x_te, y_tr, y_te = datos_mnist()
    return x_tr.reshape(-1, 784).astype(np.float32) / 255, x_te.reshape(-1, 784).astype(np.float32) / 255, y_tr, y_te


def entrenar():
    from entrenar import entrenar_modelo
    return entrenar_modelo()


def cargar():
    if not RUTA_MODELO.exists():
        raise FileNotFoundError('Falta el modelo CNN. Ejecuta: python entrenar.py')
    checkpoint = torch.load(RUTA_MODELO, map_location='cpu', weights_only=True)
    if checkpoint['version_preparacion'] != VERSION_PREPARACION:
        raise ValueError('El modelo necesita reentrenamiento: python entrenar.py')
    red = RedDigitos()
    red.load_state_dict(checkpoint['pesos'])
    return Clasificador(red)


def evaluar(modelo):
    _, x_te, _, y_te = datos_mnist()
    imagenes = x_te.reshape(-1, 784).astype(np.float32) / 255
    pred = modelo.predict(imagenes)
    return {'exactitud': accuracy_score(y_te, pred),
            'matriz': confusion_matrix(y_te, pred, labels=np.arange(10)),
            'informe': classification_report(y_te, pred, digits=4, zero_division=0),
            'imagenes': imagenes, 'etiquetas': y_te}
