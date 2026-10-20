# Informe IEEE de NeuroDígitos

Este borrador está redactado en español con la clase `IEEEtran` en modo `conference`: dos columnas, citas numéricas, figuras y tablas referenciadas. Los resultados corresponden a la ejecución guardada del proyecto, no a valores de ejemplo.

## Editarlo en Overleaf

1. Entra en https://www.overleaf.com e inicia sesión.
2. Crea un proyecto mediante la opción de subir un proyecto y selecciona `NeuroDigitos_IEEE.zip`.
3. Abre `main.tex` como documento principal y usa el compilador **pdfLaTeX**.
4. Completa `autores.tex`: autor(es), carrera, universidad, ciudad y correo.
5. Recompila y descarga el PDF. Las citas se procesan con BibTeX y el estilo `IEEEtran`.

Los nombres de los botones pueden variar con el idioma de la interfaz. También puedes usar la plantilla oficial de IEEE y cargar los archivos de esta carpeta:
https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn

No hace falta instalar Python, entrenar la red ni subir MNIST a Overleaf: las figuras y tablas ya están incluidas.

## Qué debes personalizar antes de entregar

- Completa los datos de autor; no se han inventado nombres ni afiliaciones.
- Contrasta el límite de páginas, idioma y modalidad de plantilla con lo que pidió el docente. Este informe usa IEEE para conferencia, no la variante de revista.
- Curso y docente están anotados como campos pendientes en `autores.tex`. Añádelos al bloque de afiliación solo si la consigna lo requiere.
- Lee el texto y adapta el alcance a tu trabajo. No declares que se hizo un estudio con personas: solo se evaluó MNIST y variantes sintéticas.
- Si reentrenas, vuelve a generar métricas, comparación, tablas y figuras; revisa también las cifras escritas en los párrafos y el resumen.

## Estructura del artículo

1. Resumen y palabras clave.
2. Introducción y objetivo.
3. Dataset y separación de entrenamiento/validación/prueba.
4. Preparación, arquitectura y entrenamiento.
5. Implementación de la aplicación.
6. Resultados, errores y comparación.
7. Limitaciones y reproducibilidad.
8. Conclusiones y referencias.

No se incluye portada, índice ni márgenes personalizados, porque se mantiene la composición de `IEEEtran`. Es un informe académico con formato IEEE; no implica que esté publicado o aceptado por IEEE.

## Archivos

- `main.tex`: texto del artículo.
- `autores.tex`: datos que debes completar.
- `referencias.bib`: fuentes citadas.
- `figuras/*.pdf`: ejemplos MNIST, aprendizaje, confusión y errores.
- `tabla_metricas.tex` y `valores.tex`: datos calculados desde los resultados.
- `evidencia/*.json`: copia de las métricas de la ejecución documentada.

## Compilación local

Con una distribución LaTeX que incluya IEEEtran y español:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

También se comprobó con Tectonic:

```bash
tectonic main.tex
```

Las figuras del documento son PDFs vectoriales (con las imágenes MNIST rasterizadas a su resolución original). Para regenerarlas desde la raíz del proyecto:

```bash
python -m pip install -r docs/requirements.txt
python docs/generar_figuras.py
```
