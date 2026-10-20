"""Ejecuta: python evaluar.py [--reentrenar]"""
import argparse
from motor import cargar, evaluar, entrenar, RUTA_MODELO

parser = argparse.ArgumentParser()
parser.add_argument('--reentrenar', action='store_true')
args = parser.parse_args()
modelo = entrenar() if args.reentrenar else cargar()
info = evaluar(modelo)
print(f"Exactitud: {info['exactitud']:.2%}")
print(info['informe'])
print('Matriz de confusión:\n', info['matriz'])
print('Modelo:', RUTA_MODELO)
