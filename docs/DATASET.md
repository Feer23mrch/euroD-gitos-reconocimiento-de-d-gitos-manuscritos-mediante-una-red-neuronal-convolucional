# Cómo entrar y entender el dataset

El modelo actual usa **MNIST**. No es el pequeño dataset `digits` de 8 × 8 que utilizaba la versión anterior.

## Ver las imágenes en la aplicación

Con `python servidor.py` en ejecución, abre:

http://127.0.0.1:8000/dataset

Selecciona entrenamiento, validación o prueba; filtra por dígito y cambia de página. Cada imagen muestra su **etiqueta real**, su posición original en el archivo y un enlace para guardarla como PNG. Los índices empiezan en cero. Una etiqueta no es una predicción del modelo.

## Acceso a la fuente

- Documentación y descripción de los arrays: https://keras.io/api/datasets/mnist/
- Archivo original usado en este proyecto: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
- Página histórica: http://yann.lecun.com/exdb/mnist/ (su disponibilidad puede variar).

El archivo ya está descargado en `datos/mnist.npz`. No necesitas descargarlo nuevamente ni instalar TensorFlow para leerlo: basta NumPy.

## Qué contiene

| Array | Forma | Significado |
|---|---|---|
| `x_train` | `(60000, 28, 28)` | Imágenes del conjunto original de entrenamiento |
| `y_train` | `(60000,)` | Etiquetas reales correspondientes |
| `x_test` | `(10000, 28, 28)` | Imágenes reservadas para prueba |
| `y_test` | `(10000,)` | Etiquetas reales de prueba |

Cada imagen contiene 784 píxeles enteros entre 0 y 255, no una fotografía RGB. Cero es negro y 255 es blanco. Las etiquetas van de 0 a 9.

En este proyecto, los 60 000 ejemplos iniciales se separan en **55 000 para entrenar y 5 000 para validar**, con estratificación por clase y semilla 42. Los 10 000 de prueba no se utilizan para actualizar los pesos ni elegir la época. El archivo original no contiene un array `x_val`: la validación se obtiene seleccionando índices de `x_train`.

## Abrirlo en Python

Desde la carpeta principal del proyecto y con el entorno activado:

```python
import numpy as np
from PIL import Image

with np.load('datos/mnist.npz', allow_pickle=False) as datos:
    print(datos.files)
    print(datos['x_train'].shape)  # (60000, 28, 28)
    indice = 0
    imagen = datos['x_test'][indice]
    etiqueta = int(datos['y_test'][indice])

print('Etiqueta real:', etiqueta)
Image.fromarray(imagen).resize((280, 280), Image.Resampling.NEAREST).save('ejemplo_mnist.png')
```

Abre `ejemplo_mnist.png` para verlo. La ampliación solo facilita la visualización: los datos siguen teniendo 28 × 28 píxeles.

`datos/mnist_preparado_v2.npz` es una caché derivada por nuestro preprocesamiento (recorte, ruido, escala y centrado). Para estudiar la fuente utiliza **mnist.npz**; para estudiar qué recibe la CNN, consulta la vista procesada de la aplicación o `preparar` en `motor.py`.

## Atribución y límites

Según Keras, MNIST se atribuye a Yann LeCun y Corinna Cortes, deriva de NIST y usa licencia CC BY-SA 3.0. Cita la fuente al reutilizar las imágenes. El documento IEEE incluye la referencia correspondiente.

La exactitud sobre MNIST no mide automáticamente la precisión con cualquier fotografía o con la escritura de una persona concreta. Las variantes sintéticas del proyecto derivan del mismo conjunto de prueba; no son un nuevo dataset independiente.
