"""Interfaz web local: python servidor.py."""
import argparse
import base64
import binascii
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image, UnidentifiedImageError

from motor import cargar, evaluar, preparar, LADO
from dataset import consultar

WEB = Path(__file__).with_name('web')
MAX_BODY = 8 * 1024 * 1024


def crear_handler(modelo):
    info = evaluar(modelo)

    class Handler(BaseHTTPRequestHandler):
        def responder(self, contenido, estado=200, tipo='application/json; charset=utf-8'):
            if isinstance(contenido, dict):
                contenido = json.dumps(contenido, allow_nan=False).encode()
            self.send_response(estado)
            self.send_header('Content-Type', tipo)
            self.send_header('Content-Length', str(len(contenido)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(contenido)

        def resultado(self, vector):
            probabilidades = modelo.predict_proba(vector)[0]
            if not np.isfinite(probabilidades).all():
                raise ValueError('El modelo no pudo calcular un resultado válido.')
            orden = np.argsort(probabilidades)[::-1]
            return {'predicciones': [
                {'digito': int(modelo.classes_[i]), 'probabilidad': float(probabilidades[i])}
                for i in orden], 'pixeles': vector.reshape(LADO, LADO).tolist()}

        def do_GET(self):
            url = urlsplit(self.path)
            ruta = url.path
            archivos = {'/': ('index.html', 'text/html; charset=utf-8'),
                        '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                        '/style.css': ('style.css', 'text/css; charset=utf-8'),
                        '/dataset': ('dataset.html', 'text/html; charset=utf-8'),
                        '/dataset.js': ('dataset.js', 'text/javascript; charset=utf-8'),
                        '/dataset.css': ('dataset.css', 'text/css; charset=utf-8')}
            if ruta in archivos:
                archivo, tipo = archivos[ruta]
                return self.responder((WEB / archivo).read_bytes(), tipo=tipo)
            if ruta == '/api/dataset':
                parametros = parse_qs(url.query)
                try:
                    parte = parametros.get('parte', ['prueba'])[0]
                    valor = parametros.get('digito', ['todos'])[0]
                    digito = None if valor == 'todos' else int(valor)
                    pagina = int(parametros.get('pagina', ['0'])[0])
                    return self.responder(consultar(parte, digito, pagina))
                except ValueError as exc:
                    return self.responder({'error': str(exc)}, 400)
            if ruta == '/api/evaluacion':
                return self.responder({'exactitud': float(info['exactitud']),
                                       'total': len(info['etiquetas']),
                                       'matriz': info['matriz'].tolist()})
            if ruta.startswith('/api/ejemplo/'):
                try:
                    indice = int(ruta.rsplit('/', 1)[1])
                    if not 0 <= indice < len(info['etiquetas']):
                        raise ValueError()
                    resultado = self.resultado(info['imagenes'][indice:indice + 1])
                    resultado['real'] = int(info['etiquetas'][indice])
                    return self.responder(resultado)
                except ValueError:
                    return self.responder({'error': 'Ejemplo no disponible.'}, 400)
            self.responder({'error': 'Página no encontrada.'}, 404)

        def do_POST(self):
            if self.path != '/api/reconocer':
                return self.responder({'error': 'Ruta no encontrada.'}, 404)
            try:
                longitud = int(self.headers.get('Content-Length', '0'))
                if not 0 < longitud <= MAX_BODY:
                    return self.responder({'error': 'Usa una imagen de menos de 5 MB.'}, 413)
                datos = json.loads(self.rfile.read(longitud))
                if not isinstance(datos, dict) or not isinstance(datos.get('imagen'), str):
                    raise ValueError('Selecciona una imagen o dibuja un dígito.')
                cabecera, separador, codificado = datos['imagen'].partition(',')
                if not separador or not cabecera.startswith('data:image/') or not cabecera.endswith(';base64'):
                    raise ValueError('El formato de imagen no es válido.')
                contenido = base64.b64decode(codificado, validate=True)
                with Image.open(io.BytesIO(contenido)) as fuente:
                    if fuente.width * fuente.height > 16_000_000:
                        raise ValueError('Usa una imagen de hasta 16 megapíxeles.')
                    vector = preparar(fuente)
                if vector is None:
                    raise ValueError('No encuentro trazos. Dibuja un dígito o elige otra imagen.')
                self.responder(self.resultado(vector))
            except (ValueError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                mensaje = str(exc) if isinstance(exc, ValueError) else 'No se pudo leer la imagen. Prueba con PNG o JPG.'
                self.responder({'error': mensaje}, 400)

    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--puerto', type=int, default=8000)
    args = parser.parse_args()
    servidor = HTTPServer(('127.0.0.1', args.puerto), crear_handler(cargar()))
    print(f'NeuroDígitos listo: http://127.0.0.1:{args.puerto}', flush=True)
    print('Abre esa dirección en tu navegador. Para salir, pulsa Ctrl+C.', flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()
