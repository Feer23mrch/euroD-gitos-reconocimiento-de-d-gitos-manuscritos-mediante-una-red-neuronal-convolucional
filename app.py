"""Interfaz del reconocedor: python app.py"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from motor import cargar, evaluar, entrenar, preparar

BG = '#111827'
CARD = '#1f2937'
INK = '#f9fafb'
MUTED = '#9ca3af'
ACCENT = '#38bdf8'
DRAW_SIZE = 300


class Aplicacion:
    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title('NeuroDígitos | Reconocimiento del 0 al 9')
        raiz.geometry('900x670')
        raiz.minsize(860, 640)
        raiz.configure(bg=BG)
        self.modelo = cargar()
        self.info = evaluar(self.modelo)
        self.imagen = Image.new('L', (DRAW_SIZE, DRAW_SIZE), 0)
        self.pincel = ImageDraw.Draw(self.imagen)
        self.anterior = None
        self.foto = None
        self.muestra = 0
        self.construir()
        self.ver_muestra()

    def texto(self, padre, contenido, tam=12, color=INK, negrita=False, **kw):
        return tk.Label(padre, text=contenido, bg=padre['bg'], fg=color,
                        font=('Arial', tam, 'bold' if negrita else 'normal'), **kw)

    def boton(self, padre, texto, accion, primario=False):
        return tk.Button(padre, text=texto, command=accion,
                         bg=ACCENT if primario else '#374151',
                         fg='#09121c' if primario else INK,
                         activebackground='#7dd3fc' if primario else '#4b5563',
                         activeforeground='#09121c' if primario else INK,
                         relief='flat', padx=14, pady=9, cursor='hand2',
                         font=('Arial', 11, 'bold'))

    def construir(self):
        cab = tk.Frame(self.raiz, bg=BG)
        cab.pack(fill='x', padx=30, pady=(22, 12))
        self.texto(cab, 'NeuroDígitos', 23, negrita=True).pack(anchor='w')
        self.texto(cab, 'Red convolucional CNN · MNIST · imágenes de 28 × 28',
                   11, MUTED).pack(anchor='w')
        pestanas = ttk.Notebook(self.raiz)
        pestanas.pack(fill='both', expand=True, padx=26, pady=(0, 20))
        estilo = ttk.Style()
        estilo.configure('TNotebook', background=BG, borderwidth=0)
        estilo.configure('TNotebook.Tab', padding=(18, 9), font=('Arial', 11))
        self.tab_dibujo = tk.Frame(pestanas, bg=BG)
        self.tab_eval = tk.Frame(pestanas, bg=BG)
        pestanas.add(self.tab_dibujo, text='  Probar dígito  ')
        pestanas.add(self.tab_eval, text='  Evaluación del modelo  ')
        self.crear_dibujo()
        self.crear_evaluacion()

    def crear_dibujo(self):
        izq = tk.Frame(self.tab_dibujo, bg=CARD, padx=20, pady=15)
        izq.pack(side='left', fill='y', padx=(10, 10), pady=14)
        self.texto(izq, '1. Dibuja un número', 15, negrita=True).pack(anchor='w', pady=(0, 10))
        self.canvas = tk.Canvas(izq, width=DRAW_SIZE, height=DRAW_SIZE,
                                bg='#050a14', highlightthickness=2,
                                highlightbackground='#475569', cursor='crosshair')
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.iniciar)
        self.canvas.bind('<B1-Motion>', self.pintar)
        self.canvas.bind('<ButtonRelease-1>', lambda e: setattr(self, 'anterior', None))
        botones = tk.Frame(izq, bg=CARD)
        botones.pack(fill='x', pady=(14, 8))
        self.boton(botones, 'Reconocer', self.reconocer, True).pack(side='left', padx=(0, 6))
        self.boton(botones, 'Limpiar', self.limpiar).pack(side='left', padx=6)
        self.boton(izq, 'Abrir imagen PNG/JPG', self.abrir).pack(anchor='w', pady=8)
        self.texto(izq, 'Traza un dígito grande. Puedes cargar una imagen.',
                   10, MUTED, wraplength=300, justify='left').pack(anchor='w')
        der = tk.Frame(self.tab_dibujo, bg=CARD, padx=22, pady=16)
        der.pack(side='left', fill='both', expand=True, padx=(0, 10), pady=14)
        self.texto(der, '2. Resultado', 15, negrita=True).pack(anchor='w')
        self.pred = self.texto(der, '—', 66, ACCENT, True)
        self.pred.pack(anchor='w', pady=(0, 2))
        self.estado = self.texto(der, 'Dibuja y pulsa Reconocer', 11, MUTED)
        self.estado.pack(anchor='w', pady=(0, 20))
        self.texto(der, 'Tres predicciones más probables', 12, negrita=True).pack(anchor='w')
        self.barras = []
        for _ in range(3):
            fila = tk.Frame(der, bg=CARD)
            fila.pack(fill='x', pady=8)
            titulo = self.texto(fila, '—', 12, INK, True, width=3, anchor='w')
            titulo.pack(side='left')
            barra = tk.Canvas(fila, width=170, height=13, bg='#374151', highlightthickness=0)
            barra.pack(side='left', padx=8)
            porcentaje = self.texto(fila, '—', 11, MUTED)
            porcentaje.pack(side='left')
            self.barras.append((titulo, barra, porcentaje))
        self.texto(der, 'Vista de la imagen que recibe la red (28 × 28)',
                   11, MUTED).pack(anchor='w', pady=(22, 8))
        self.vista = tk.Label(der, bg='#050a14', width=110, height=110)
        self.vista.pack(anchor='w')

    def iniciar(self, evento):
        self.anterior = (evento.x, evento.y)
        self.pintar(evento)

    def pintar(self, evento):
        x, y = max(0, min(DRAW_SIZE-1, evento.x)), max(0, min(DRAW_SIZE-1, evento.y))
        previo = self.anterior or (x, y)
        self.canvas.create_line(*previo, x, y, fill='white', width=22,
                                capstyle='round', smooth=True)
        self.pincel.line((*previo, x, y), fill=255, width=22)
        self.pincel.ellipse((x-11, y-11, x+11, y+11), fill=255)
        self.anterior = (x, y)

    def limpiar(self):
        self.canvas.delete('all')
        self.pincel.rectangle((0, 0, DRAW_SIZE, DRAW_SIZE), fill=0)
        self.pred.config(text='—')
        self.estado.config(text='Dibuja y pulsa Reconocer')
        for t, barra, pct in self.barras:
            t.config(text='—'); barra.delete('all'); pct.config(text='—')
        self.vista.config(image='')
        self.anterior = None

    def mostrar_resultado(self, vector):
        probs = self.modelo.predict_proba(vector)[0]
        indices = np.argsort(probs)[::-1][:3]
        self.pred.config(text=str(int(indices[0])))
        self.estado.config(text=f'Probabilidad estimada: {probs[indices[0]]:.1%}')
        for (titulo, barra, pct), indice in zip(self.barras, indices):
            titulo.config(text=str(int(indice)))
            barra.delete('all')
            barra.create_rectangle(0, 0, int(170*probs[indice]), 13,
                                   fill=ACCENT, outline='')
            pct.config(text=f'{probs[indice]:.1%}')
        pixeles = np.uint8(np.clip(vector.reshape(28, 28) * 255, 0, 255))
        self.foto = ImageTk.PhotoImage(Image.fromarray(pixeles).resize((112, 112), Image.Resampling.NEAREST))
        self.vista.config(image=self.foto)

    def reconocer(self):
        vector = preparar(self.imagen)
        if vector is None:
            self.estado.config(text='Primero dibuja un dígito')
            return
        self.mostrar_resultado(vector)

    def abrir(self):
        ruta = filedialog.askopenfilename(filetypes=[('Imágenes', '*.png *.jpg *.jpeg *.bmp')])
        if not ruta:
            return
        try:
            with Image.open(ruta) as fuente:
                vector = preparar(fuente)
            if vector is None:
                raise ValueError('La imagen está vacía')
            self.limpiar()
            self.mostrar_resultado(vector)
            self.estado.config(text=f'Imagen cargada · probabilidad: {self.modelo.predict_proba(vector).max():.1%}')
        except Exception as exc:
            messagebox.showerror('No se pudo abrir la imagen', str(exc))

    def crear_evaluacion(self):
        marco = tk.Frame(self.tab_eval, bg=CARD, padx=22, pady=15)
        marco.pack(fill='both', expand=True, padx=10, pady=14)
        self.texto(marco, 'Resultados con datos reservados para prueba',
                   17, negrita=True).pack(anchor='w')
        total = len(self.info['etiquetas'])
        self.texto(marco, f"Exactitud: {self.info['exactitud']:.2%}  ·  {total} imágenes de prueba",
                   14, ACCENT, True).pack(anchor='w', pady=(8, 5))
        self.texto(marco, 'Matriz de confusión · filas: clase real · columnas: predicción',
                   11, MUTED).pack(anchor='w')
        self.matriz = tk.Canvas(marco, width=400, height=300, bg=CARD, highlightthickness=0)
        self.matriz.pack(side='left', padx=(0, 35), pady=10, anchor='n')
        m = self.info['matriz']
        for i in range(10):
            self.matriz.create_text(45+i*31, 10, text=str(i), fill=INK)
            self.matriz.create_text(10, 42+i*25, text=str(i), fill=INK)
            for j in range(10):
                v = int(m[i, j]); fuerza = v / max(1, int(m[i].max()))
                color = '#075985' if fuerza > .65 else '#164e63' if v else '#374151'
                x, y = 30+j*31, 28+i*25
                self.matriz.create_rectangle(x, y, x+28, y+23, fill=color, outline=BG)
                self.matriz.create_text(x+14, y+12, text=str(v), fill=INK, font=('Arial', 9))
        columna = tk.Frame(marco, bg=CARD)
        columna.pack(side='left', fill='both', expand=True, pady=10)
        self.texto(columna, 'Ejemplo del conjunto de prueba', 12, negrita=True).pack(anchor='w')
        self.ejemplo_img = tk.Label(columna, bg='#050a14')
        self.ejemplo_img.pack(anchor='w', pady=12)
        self.ejemplo_txt = self.texto(columna, '', 11, MUTED)
        self.ejemplo_txt.pack(anchor='w')
        self.boton(columna, 'Siguiente ejemplo', self.siguiente).pack(anchor='w', pady=12)
        self.texto(columna, 'Entrenamiento: 55 000 · validación: 5 000\n'
                   'CNN: 28 × 28 → 10 dígitos · prueba: 10 000\n'
                   'Aumento de datos + centrado del trazo',
                   11, MUTED, justify='left').pack(anchor='w', pady=12)

    def ver_muestra(self):
        img = self.info['imagenes'][self.muestra].reshape(28, 28)
        pic = Image.fromarray(np.uint8(img * 255)).resize((128, 128), Image.Resampling.NEAREST)
        self.foto_ejemplo = ImageTk.PhotoImage(pic)
        self.ejemplo_img.config(image=self.foto_ejemplo)
        real = int(self.info['etiquetas'][self.muestra])
        pred = int(self.modelo.predict(self.info['imagenes'][self.muestra:self.muestra+1])[0])
        self.ejemplo_txt.config(text=f'Imagen {self.muestra+1} · real: {real} · predicción: {pred}')

    def siguiente(self):
        self.muestra = (self.muestra + 1) % len(self.info['etiquetas'])
        self.ver_muestra()


if __name__ == '__main__':
    raiz = tk.Tk()
    Aplicacion(raiz)
    raiz.mainloop()
