"""Pruebas del contrato HTTP: python -m unittest test_servidor.py."""
import base64
import io
import json
import threading
import unittest
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw
from motor import cargar
from servidor import crear_handler


class WebTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(('127.0.0.1', 0), crear_handler(cargar()))
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, data=None):
        request = Request(self.url + path, data=json.dumps(data).encode() if data is not None else None,
                          headers={'Content-Type': 'application/json'})
        try:
            response = urlopen(request, timeout=10)
        except HTTPError as error:
            response = error
        with response:
            content = response.read()
            if 'application/json' in response.headers['Content-Type']:
                content = json.loads(content)
            return response.status, content

    def image_data(self, image):
        buffer = io.BytesIO()
        image.save(buffer, 'PNG')
        return {'imagen': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}

    def test_static_files(self):
        for path in ['/', '/app.js', '/style.css', '/dataset', '/dataset.js', '/dataset.css']:
            status, body = self.request(path)
            self.assertEqual(status, 200)
            self.assertTrue(body)
        self.assertEqual(self.request('/../motor.py')[0], 404)

    def test_evaluation_and_example(self):
        status, info = self.request('/api/evaluacion')
        self.assertEqual(status, 200)
        self.assertEqual(info['total'], 10000)
        self.assertEqual(sum(map(sum, info['matriz'])), info['total'])
        status, result = self.request('/api/ejemplo/0')
        self.assertEqual(status, 200)
        self.assertIn(result['real'], range(10))
        self.assertEqual(len(result['predicciones']), 10)
        self.assertAlmostEqual(sum(p['probabilidad'] for p in result['predicciones']), 1)
        self.assertEqual(self.request('/api/ejemplo/-1')[0], 400)

    def test_uploaded_digit_and_transparency(self):
        for background in ['white', (0, 0, 0, 0)]:
            image = Image.new('RGBA', (100, 100), background)
            ImageDraw.Draw(image).line([(20, 20), (75, 20), (40, 85)], fill='black', width=10)
            status, result = self.request('/api/reconocer', self.image_data(image))
            self.assertEqual(status, 200)
            self.assertEqual(len(result['pixeles']), 28)
            self.assertTrue(any(v > 0 for row in result['pixeles'] for v in row))

    def test_invalid_input(self):
        for data in [[], {}, {'imagen': 'invalid'}, {'imagen': 'data:image/png;base64,??'},
                     self.image_data(Image.new('RGB', (32, 32), 'white'))]:
            status, result = self.request('/api/reconocer', data)
            self.assertEqual(status, 400)
            self.assertIn('error', result)

    def test_dataset_filter_and_pagination(self):
        status, result = self.request('/api/dataset?parte=prueba&digito=7&pagina=0')
        self.assertEqual(status, 200)
        self.assertEqual(result['total'], 1028)
        self.assertEqual(len(result['ejemplos']), 30)
        self.assertTrue(all(x['etiqueta'] == 7 for x in result['ejemplos']))
        self.assertTrue(all(x['origen'] == 'test' for x in result['ejemplos']))
        data = base64.b64decode(result['ejemplos'][0]['imagen'].split(',')[1])
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.size, (28, 28))
        _, second = self.request('/api/dataset?parte=prueba&digito=7&pagina=1')
        self.assertFalse(set(x['indice'] for x in result['ejemplos']) &
                         set(x['indice'] for x in second['ejemplos']))
        for query in ['parte=otro', 'digito=10', 'pagina=-1', 'pagina=999999', 'digito=abc']:
            self.assertEqual(self.request('/api/dataset?' + query)[0], 400)

    def test_dataset_partitions(self):
        from dataset import particiones
        parts = particiones()
        train = set(parts['entrenamiento'][2])
        validation = set(parts['validacion'][2])
        self.assertEqual(len(train), 55000)
        self.assertEqual(len(validation), 5000)
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, set(range(60000)))


if __name__ == '__main__':
    unittest.main()
