"""Regresiones del preprocesamiento que comparten entrenamiento e interfaz."""
import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import center_of_mass
from motor import preparar


class PreparacionTest(unittest.TestCase):
    def dibujo(self, fondo='white', tinta='black', posicion=(70, 30)):
        im = Image.new('RGB', (320, 320), fondo)
        x, y = posicion
        ImageDraw.Draw(im).line([(x, y), (x + 110, y), (x + 30, y + 180)], fill=tinta, width=15)
        return im

    def test_fondos_vacios(self):
        for fondo in ['white', 'black', '#eeeeee']:
            self.assertIsNone(preparar(Image.new('RGB', (320, 320), fondo)))

    def test_polaridad_y_rango(self):
        a = preparar(self.dibujo())
        b = preparar(self.dibujo('black', 'white'))
        self.assertEqual(a.shape, (1, 784))
        self.assertGreater(a.max(), 0)
        self.assertTrue(np.all((a >= 0) & (a <= 1)))
        np.testing.assert_allclose(a, b)

    def test_traslacion_y_centrado(self):
        a = preparar(self.dibujo(posicion=(25, 25)))
        b = preparar(self.dibujo(posicion=(165, 115)))
        np.testing.assert_allclose(a, b)
        cy, cx = center_of_mass(a.reshape(28, 28))
        self.assertAlmostEqual(cx, 13.5, delta=0.1)
        self.assertAlmostEqual(cy, 13.5, delta=0.1)

    def test_transparencia(self):
        rgba = Image.new('RGBA', (320, 320), (0, 0, 0, 0))
        ImageDraw.Draw(rgba).line([(70, 30), (180, 30), (100, 210)], fill='black', width=15)
        np.testing.assert_allclose(preparar(rgba), preparar(self.dibujo()))

    def test_ruido_aislado_no_cambia_encuadre(self):
        limpio = self.dibujo()
        ruido = limpio.copy()
        ruido.putpixel((310, 310), (0, 0, 0))
        np.testing.assert_allclose(preparar(limpio), preparar(ruido))


if __name__ == '__main__':
    unittest.main()
