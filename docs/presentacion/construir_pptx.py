"""Construye entregables/EritroFase_Sustentacion.pptx (16:9, 16 diapositivas).

Todas las cifras provienen de results/tables/ (ver comentarios por diapositiva).
Ejecutar desde la raíz de la rama:
    python docs/presentacion/construir_pptx.py
"""
from datetime import datetime
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

RAIZ = Path(__file__).resolve().parents[2]
FIG = RAIZ / "results" / "figures"
SALIDA = RAIZ / "entregables" / "EritroFase_Sustentacion.pptx"

ACENTO = RGBColor(0x0B, 0x5C, 0x8E)
TEXTO = RGBColor(0x1F, 0x29, 0x33)
GRIS = RGBColor(0x6B, 0x72, 0x80)
CLARO = RGBColor(0xEE, 0xF5, 0xFB)
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)
FUENTE = "Calibri"
TOTAL = 18

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
LAYOUT_TITULO = prs.slide_layouts[5]  # solo título: el título queda en el esquema del documento


def _fuente(run, size, bold=False, color=TEXTO, name=FUENTE):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def _vinetas(paragraph, nivel=0):
    pPr = paragraph._p.get_or_add_pPr()
    sangria = Inches(0.3 + 0.3 * nivel)
    pPr.set("marL", str(int(sangria)))
    pPr.set("indent", str(int(-Inches(0.25))))
    buFont = etree.SubElement(pPr, qn("a:buFont"))
    buFont.set("typeface", "Arial")
    buChar = etree.SubElement(pPr, qn("a:buChar"))
    buChar.set("char", "•" if nivel == 0 else "–")


def caja(slide, x, y, w, h, items, size=18, espacio=10, vinetas=True):
    """items: lista de str o (str, nivel). Devuelve la caja de texto."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        texto, nivel = (it, 0) if isinstance(it, str) else it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(espacio)
        _fuente(p.add_run(), size - 2 * nivel)
        p.runs[0].text = texto
        if vinetas:
            _vinetas(p, nivel)
    return tb


def texto(slide, x, y, w, h, parrafos, size=16, color=TEXTO, bold=False, alinear=PP_ALIGN.LEFT,
          ancla=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = ancla
    for i, par in enumerate(parrafos):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = alinear
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = par
        _fuente(r, size, bold=bold, color=color)
    return tb


def rectangulo(slide, x, y, w, h, relleno=CLARO, borde=None, forma=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(forma, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = relleno
    if borde is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = borde
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    return s


def rotulo_forma(shape, lineas, size=14, color=TEXTO, bold_primera=True):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right"):
        setattr(tf, m, Inches(0.06))
    for i, l in enumerate(lineas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = l
        _fuente(r, size, bold=(bold_primera and i == 0), color=color)


def imagen(slide, ruta, x, y, w, h):
    """Inserta la imagen ajustada dentro de la caja (x, y, w, h) sin deformarla."""
    ruta = Path(ruta)
    with Image.open(ruta) as im:
        iw, ih = im.size
    escala = min(w / iw, h / ih)
    ancho, alto = iw * escala, ih * escala
    px = x + (w - ancho) / 2
    py = y + (h - alto) / 2
    return slide.shapes.add_picture(str(ruta), Inches(px), Inches(py), Inches(ancho), Inches(alto))


def tabla(slide, x, y, w, filas, anchos, size=14, alto_fila=0.42, resaltar=None):
    """filas[0] es el encabezado. resaltar: conjunto de índices de fila a destacar en negrita."""
    resaltar = resaltar or set()
    n, m = len(filas), len(filas[0])
    gf = slide.shapes.add_table(n, m, Inches(x), Inches(y), Inches(w), Inches(alto_fila * n))
    t = gf.table
    total = sum(anchos)
    for j, a in enumerate(anchos):
        t.columns[j].width = Inches(w * a / total)
    for i, fila in enumerate(filas):
        t.rows[i].height = Inches(alto_fila)
        for j, val in enumerate(fila):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.text = ""
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            if i == 0:
                _fuente(r, size, bold=True, color=BLANCO)
                c.fill.solid()
                c.fill.fore_color.rgb = ACENTO
            else:
                _fuente(r, size, bold=(i in resaltar), color=TEXTO)
                c.fill.solid()
                c.fill.fore_color.rgb = CLARO if i in resaltar else BLANCO
    return gf


def nueva(num, titulo, responsable, guion):
    s = prs.slides.add_slide(LAYOUT_TITULO)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.95)
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.BOTTOM
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = titulo
    _fuente(r, 30, bold=True, color=TEXTO)
    barra = rectangulo(s, 0.6, 1.34, 1.3, 0.07, relleno=ACENTO, forma=MSO_SHAPE.RECTANGLE)
    barra.name = "Acento"
    texto(s, 0.6, 7.0, 9.0, 0.3, ["EritroFase · SI3014 · Grupo 1"], size=10, color=GRIS)
    texto(s, 11.7, 7.0, 1.0, 0.3, [f"{num} / {TOTAL}"], size=10, color=GRIS, alinear=PP_ALIGN.RIGHT)
    notas = s.notes_slide.notes_text_frame
    notas.text = f"Responsable: {responsable}.\n\n{guion}"
    return s


# ---------------------------------------------------------------- 1. Portada
s = prs.slides.add_slide(LAYOUT_TITULO)
t = s.shapes.title
t.left, t.top, t.width, t.height = Inches(0.9), Inches(1.9), Inches(11.5), Inches(1.9)
tf = t.text_frame
tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.BOTTOM
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.LEFT
r = p.add_run()
r.text = "Corrección del fondo en imágenes de fase de eritrocitos"
_fuente(r, 40, bold=True)
rectangulo(s, 0.9, 4.0, 1.6, 0.08, relleno=ACENTO, forma=MSO_SHAPE.RECTANGLE).name = "Acento"
texto(s, 0.9, 4.3, 11.5, 0.6, ["Redes neuronales informadas por la física con representaciones implícitas"],
      size=22, color=ACENTO)
texto(s, 0.9, 5.4, 11.5, 0.9,
      ["SI3014 · Redes Neuronales y Aprendizaje Profundo · Universidad EAFIT · 2026-2",
       "Grupo 1 – EritroFase: Juan José Escobar, Emmanuel Hernández, Mateo Lopera"],
      size=16, color=GRIS)
s.notes_slide.notes_text_frame.text = (
    "Responsable: Juan José Escobar.\n\n"
    "Buenas tardes. Somos el Grupo 1, con el proyecto EritroFase. Presentamos la corrección del fondo "
    "en imágenes de fase de eritrocitos mediante una red neuronal informada por la física. En los próximos "
    "quince minutos vamos a revisar el problema, los datos, el modelo y los resultados. Adelantamos que el "
    "modelo no cumplió el criterio de éxito en la prueba, y que explicar por qué es la parte más útil del trabajo.")

# ---------------------------------------------------------------- 2. Problema
s = nueva(2, "El fondo sesga la medida de la célula", "Juan José Escobar",
          "La microscopía holográfica digital da una imagen de fase: el desfase que introduce la célula respecto "
          "al medio, proporcional a su espesor (Proposición 1). Todas las medidas cuantitativas, como altura, "
          "volumen y masa seca, se calculan respecto al nivel del fondo. Si el fondo está inclinado o curvo, la "
          "medida sale sesgada. El problema es estimar ese fondo y restarlo sin tocar los valores dentro de la "
          "célula. Unos treinta segundos para esta diapositiva; pasamos a los objetivos.")
caja(s, 0.6, 1.65, 12.1, 2.6, [
    "La imagen de fase cuantitativa mide el desfase que introduce cada célula respecto al medio.",
    "Las medidas de altura, volumen y masa seca se toman respecto al nivel del fondo.",
    "Un fondo inclinado o curvo sesga esas medidas.",
    "Objetivo del problema: estimar el fondo y restarlo sin alterar los valores dentro de la célula.",
], size=19)
imagen(s, FIG / "E1" / "fondo_lento_ejemplo.png", 0.9, 4.45, 11.5, 2.4)

# ---------------------------------------------------------------- 3. Objetivos
s = nueva(3, "Objetivos", "Juan José Escobar",
          "El objetivo general es estimar y compensar el fondo preservando la célula, con una red informada por la "
          "física basada en representaciones implícitas. Los tres específicos corresponden a las etapas del "
          "proyecto: fundamentos y restricciones, caracterización y selección de datos, y validación del fondo "
          "con métricas fijadas antes de evaluar. Son los criterios con los que cerramos la presentación.")
rectangulo(s, 0.6, 1.7, 12.1, 1.7, relleno=CLARO)
texto(s, 0.85, 1.78, 11.6, 1.55, [
    "Objetivo general",
    "Estimar y compensar el fondo de imágenes de fase de eritrocitos, preservando los valores de la célula, "
    "con una red neuronal informada por la física y basada en representaciones implícitas.",
], size=18, color=TEXTO)
s.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
s.shapes[-1].text_frame.paragraphs[0].runs[0].font.color.rgb = ACENTO
caja(s, 0.6, 3.6, 12.1, 3.1, [
    "Específico 1. Fundamentar la representación neuronal y las restricciones físicas (SIREN, características "
    "de Fourier y modelo de formación de la imagen).",
    "Específico 2. Caracterizar el dataset y construir una partición de entrenamiento, validación, prueba y "
    "generalización sin fuga de información.",
    "Específico 3. Validar el fondo y la preservación de la célula frente a líneas base, con métricas y "
    "criterio de éxito fijados antes de entrenar.",
], size=18, espacio=12, vinetas=False)

# ---------------------------------------------------------------- 4. Datos y partición
s = nueva(4, "Datos y partición sin fuga de información", "Juan José Escobar",
          "El dataset tiene 192 adquisiciones; cada una aporta cuatro campos y cada campo, dieciséis recortes de "
          "256 por 256 píxeles, para 12 288 recortes. Los recortes de una misma adquisición no son independientes: "
          "son la misma escena rotada. Por eso la partición original, que comparte 47,5 % de los recortes entre "
          "carpetas, se descartó. Usamos la adquisición como unidad: doce, cinco y cinco imágenes para desarrollo "
          "el 29 de enero y ocho de generalización el 28 de enero. Los umbrales y los hiperparámetros se eligieron "
          "con validación, y la prueba se usó una sola vez.")
caja(s, 0.6, 1.7, 6.1, 4.9, [
    "192 adquisiciones, 768 campos y 12 288 recortes de 256 × 256 px.",
    "La partición original comparte 5 837 recortes (47,5 %) entre carpetas: hay fuga.",
    "Unidad de independencia: la adquisición, no el recorte.",
    "Umbrales e hiperparámetros se eligen con validación; la prueba se usa una sola vez.",
], size=18, espacio=14)
tabla(s, 7.1, 1.9, 5.6, [
    ["Conjunto", "Imágenes", "Adquisición"],
    ["Entrenamiento", "12", "29/01/2021 (12 adq.)"],
    ["Validación", "5", "29/01/2021 (5 adq.)"],
    ["Prueba", "5", "29/01/2021 (5 adq.)"],
    ["Generalización", "8", "28/01/2021 (3 adq.)"],
], anchos=[2.2, 1.2, 2.6], size=14, alto_fila=0.55)
texto(s, 7.1, 4.8, 5.6, 1.2, [
    "Desarrollo 12/5/5: 22 adquisiciones distintas para 22 imágenes, sin recortes compartidos."],
    size=14, color=GRIS)

# ---------------------------------------------------------------- 5. Preprocesamiento
s = nueva(5, "Preprocesamiento", "Juan José Escobar",
          "La fase viene envuelta en el intervalo de menos pi a pi, así que la desenvolvemos con unwrap_phase. Es "
          "exacto en el caso sintético y tarda unos diez milisegundos por imagen; SPINR fue unas 2 650 veces más "
          "lento y dejó errores de 2 pi. Las métricas se expresan en niveles equivalentes, sin normalizar. Las "
          "máscaras de referencia célula y fondo se construyen sin el modelo, con umbral de 4 sigma, y solo sirven "
          "para evaluar, nunca para ajustar.")
caja(s, 0.6, 1.7, 6.0, 5.0, [
    "Desenvolvimiento con unwrap_phase: exacto en el caso sintético y unos 10 ms por imagen.",
    "SPINR fue unas 2 650 veces más lento y dejó errores de 2π.",
    "Signo: la célula aparece con fase positiva sobre un fondo oscuro.",
    "Unidades: niveles equivalentes sin normalizar, en las que se expresan todas las métricas.",
    "Máscaras de referencia (umbral 4σ, margen de 6 px): solo para evaluar, no para ajustar.",
], size=17, espacio=12)
imagen(s, FIG / "E1" / "preprocesamiento_ejemplos.png", 6.9, 2.0, 5.9, 4.4)

# ---------------------------------------------------------------- 6. Fundamentos
s = nueva(6, "Fundamentos: representación implícita y pérdida física", "Juan José Escobar",
          "Una SIREN representa el fondo como función continua de las coordenadas; con frecuencia inicial uno, el "
          "resultado es un fondo de baja frecuencia. Las características de Fourier son la alternativa de la "
          "literatura. La física entra por la pérdida: el primer término recorta el 30 % de los residuos más "
          "grandes (esparsidad, S2), el segundo favorece el signo positivo de la célula (S3), el tercero fija el "
          "nivel de referencia (S4), y la suavidad S1 viene de la arquitectura de baja frecuencia.")
caja(s, 0.6, 1.7, 6.2, 5.0, [
    "SIREN: red con activación senoidal que representa el fondo como función continua de (x, y).",
    "Características de Fourier: alternativa de la literatura para representar la misma función.",
    "Modelo de formación: la fase medida es fondo lento más la fase de la célula (Proposición 1).",
    "Suavidad (S1): se logra con la arquitectura de baja frecuencia.",
    "Esparsidad (S2), signo (S3) y referencia de nivel (S4): van en la pérdida.",
], size=17, espacio=12)
rectangulo(s, 7.2, 2.0, 5.5, 3.6, relleno=CLARO)
texto(s, 7.45, 2.15, 5.1, 0.5, ["Pérdida del modelo A"], size=16, color=ACENTO, bold=True)
texto(s, 7.45, 2.7, 5.1, 1.2, [
    "r = u − b_θ    ·    K = ⌊0,7 N⌋",
    "L = L_rec + 0,8 [ media(ReLU(−r)²) + 0,01 · media(ReLU(r)) ] + 0,1 · mediana(r)²",
], size=15, color=TEXTO)
texto(s, 7.45, 4.25, 5.1, 1.2, [
    "L_rec: media de los K valores |r| más pequeños (S2).",
    "Segundo término: signo de la célula (S3). Tercero: referencia de nivel (S4).",
], size=13, color=GRIS)

# ---------------------------------------------------------------- 7. Arquitectura
s = nueva(7, "Arquitectura del modelo A", "Emmanuel Hernández",
          "El modelo recibe coordenadas normalizadas y produce un nivel de fondo. Tiene una capa senoidal de "
          "entrada de dos a 128 neuronas, dos capas ocultas de 128 y una salida lineal: 33 537 parámetros "
          "entrenables. Los pesos se reinician para cada imagen, y el ajuste no usa máscaras. Lo escogimos por su "
          "compromiso entre fondo y altura de la célula, y porque no requiere una segunda red. Esta arquitectura "
          "es la que se evalúa en prueba.")
cajas = [
    ("Coordenadas", "(x, y) ∈ [−1, 1]²", "N × 2"),
    ("Capa senoidal", "2 → 128", "ω₀ = 1"),
    ("Capa oculta", "128 → 128", "senoidal"),
    ("Capa oculta", "128 → 128", "senoidal"),
    ("Salida lineal", "128 → 1", "nivel de fondo"),
    ("Corrección", "u − b_θ(x, y)", "imagen sin fondo"),
]
ancho, gap, x0, y0 = 1.8, 0.28, 0.6, 1.85
for i, (a, b, c) in enumerate(cajas):
    x = x0 + i * (ancho + gap)
    r_ = rectangulo(s, x, y0, ancho, 1.55, relleno=(ACENTO if i == 5 else CLARO),
                    borde=ACENTO)
    rotulo_forma(r_, [a, b, c], size=13, color=(BLANCO if i == 5 else TEXTO))
    if i < len(cajas) - 1:
        fl = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + ancho + 0.04), Inches(y0 + 0.62),
                                Inches(gap - 0.08), Inches(0.3))
        fl.fill.solid()
        fl.fill.fore_color.rgb = GRIS
        fl.line.fill.background()
caja(s, 0.6, 3.9, 12.1, 2.9, [
    "33 537 parámetros entrenables por imagen; los pesos se reinician en cada ajuste.",
    "El ajuste no usa máscaras: solo se emplean para evaluar.",
    "Pérdida de la diapositiva 6 y optimizador Adam, con tasa 10⁻³ y 1 000 iteraciones fijas.",
    "Elegida por el compromiso entre fondo y altura de la célula, sin una segunda red.",
], size=18, espacio=12)

# ---------------------------------------------------------------- 8. Alternativas
s = nueva(8, "Alternativas exploradas y por qué A", "Emmanuel Hernández",
          "Exploramos cinco esquemas sobre las cinco imágenes de validación. A: red implícita por imagen. C: "
          "coeficientes de una base amortizada. D: dos ramas para fondo y célula, con colapso de la rama celular. "
          "E: pesos espaciales aprendidos con mínimos cuadrados, que preserva mejor pero deja más fondo. AE combina "
          "E con una red implícita. C y D solo tienen una semilla, así que no son concluyentes. Elegimos A: iguala "
          "a AE en fondo con una sola red, y la búsqueda se hizo sobre validación, lo que puede favorecer esas "
          "imágenes.")
tabla(s, 0.6, 1.7, 12.1, [
    ["Esquema", "Idea", "M1 fondo (niveles)", "|Δh| (%)", "Semillas"],
    ["A (propuesto)", "Red implícita por imagen", "6,18 ± 0,18", "4,16 ± 0,56", "3"],
    ["C", "Coeficientes de base amortizada", "19,52", "0,76", "1"],
    ["D", "Dos ramas: fondo y célula", "26,02", "0,03", "1"],
    ["E", "Pesos aprendidos + mínimos cuadrados", "8,63 ± 0,87", "1,81 ± 0,23", "3"],
    ["AE", "Pesos de E para orientar una red", "6,16 ± 0,09", "4,97 ± 0,57", "3"],
], anchos=[1.9, 4.2, 2.2, 2.0, 1.4], size=15, alto_fila=0.56, resaltar={1})
caja(s, 0.6, 5.35, 12.1, 1.5, [
    "E preserva mejor la célula (|Δh| 1,81 %), pero deja más fondo (M1 8,63 niveles).",
    "Se elige A: M1 comparable a AE, con una sola red y menor cambio de altura que AE.",
], size=16, espacio=8)
texto(s, 0.6, 6.45, 12.1, 0.4, ["Validación: medias ± desviaciones de las medianas entre semillas (C y D: una semilla)."],
      size=12, color=GRIS)

# ---------------------------------------------------------------- 9. Entrenamiento
s = nueva(9, "Entrenamiento, equipo y tiempos", "Emmanuel Hernández",
          "Cada imagen se ajusta por separado, desde sus propios pesos iniciales, con Adam a tasa 10⁻³ "
          "durante mil iteraciones fijas y sin parada temprana. Probamos tres semillas, 42, 2024 y 12345; la "
          "semilla elegida es 2024, con la regla de validación fijada antes de mirar la prueba (menor M1 mediano en validación, 5,98 niveles). El equipo fue una RTX 3060 de 12 "
          "gigabytes con PyTorch 2.14 y CUDA 12.6. Un ajuste completo cuesta unos siete segundos por imagen, y la "
          "inferencia, tres milésimas de segundo. Son mediciones de validación y no una comparación con otras "
          "arquitecturas.")
caja(s, 0.6, 1.7, 12.1, 2.6, [
    "Un ajuste independiente por imagen; pérdida de la diapositiva 6; Adam con tasa 10⁻³ y 1 000 iteraciones.",
    "Semillas 42, 2024 y 12345; se eligió la 2024 con la regla de validación, fijada antes de mirar la prueba.",
    "Equipo: GPU NVIDIA RTX 3060 (12 GB), PyTorch 2.14.0 con CUDA 12.6, Python 3.12.",
    "Tiempo medio por imagen (validación): ajuste 7,47 s; inferencia 0,003 s.",
], size=16, espacio=8)
imagen(s, FIG / "E4" / "curvas_A.png", 0.9, 4.45, 11.5, 2.45)

# ---------------------------------------------------------------- 10. Resultados en prueba
s = nueva(10, "Resultados en prueba frente a las líneas base", "Emmanuel Hernández",
          "Este es el criterio que fijamos antes de entrenar. La preservación de A, 4,81 %, cumple el umbral de "
          "cinco por ciento. Pero la mejor línea base admisible es Zernike, con 6,49 niveles de fondo residual y "
          "2,94 % de cambio de altura; A deja 7,10 niveles. Por tanto, el criterio de éxito no se cumple. SIREN "
          "deja el fondo más limpio, con 4,22 niveles, pero absorbe la célula: 12,27 %. Por eso no es admisible.")
tabla(s, 0.6, 1.7, 6.4, [
    ["Método (prueba, mediana)", "RMS fondo (niveles)", "|Δh| (%)"],
    ["Pistón", "18,18", "0,00"],
    ["Polinomio grado 3", "10,37", "1,19"],
    ["Zernike (guía)", "6,49", "2,94"],
    ["SIREN ω₀ = 1 (guía)", "4,22", "12,27"],
    ["A (propuesto)", "7,10", "4,81"],
], anchos=[3.0, 2.0, 1.4], size=15, alto_fila=0.55, resaltar={5})
imagen(s, FIG / "G10" / "comparacion_prueba.png", 7.2, 1.7, 5.6, 2.6)
caja(s, 0.6, 5.2, 12.1, 1.7, [
    "Criterio 1, preservación (|Δh| ≤ 5 %): A cumple, con 4,81 %.",
    "Criterio 2, fondo: A no supera a Zernike, la mejor línea base admisible (7,10 frente a 6,49 niveles).",
], size=16, espacio=8)

# ---------------------------------------------------------------- 11. Ejemplos de corrección en prueba
s = nueva(11, "Ejemplos de corrección en prueba", "Emmanuel Hernández",
          "Cada fila es una de las cinco imágenes de prueba. Las columnas son la entrada, el fondo que estima la red, "
          "la imagen corregida y la misma imagen en negro, donde se ve la célula sin fondo. El fondo estimado es suave, "
          "porque la red tiene baja frecuencia. Estas figuras muestran qué hace el modelo; el resumen numérico está en "
          "la diapositiva 10.")
imagen(s, FIG / "E5" / "mapas_prueba.png", 0.6, 1.6, 5.3, 5.3)
caja(s, 6.1, 1.7, 6.6, 5.0, [
    "Columnas: entrada, fondo estimado, imagen corregida y visualización en negro.",
    "El fondo estimado es suave: la baja frecuencia de la red limita sus variaciones.",
    "Cinco imágenes de prueba, una por fila.",
], size=18, espacio=14)

# ---------------------------------------------------------------- 12. Perfiles transversales en prueba
s = nueva(12, "Perfiles transversales en prueba", "Emmanuel Hernández",
          "Cada panel es un corte horizontal de una imagen de prueba. La curva azul es la fase preprocesada y la "
          "naranja la corregida; la zona sombreada es la célula. En la mayoría de los cortes el perfil corregido "
          "conserva la forma de la célula y baja su nivel. El cambio de altura en mediana es 4,81 por ciento. No hay "
          "fondo verdadero, así que los perfiles son relativos, en niveles equivalentes.")
imagen(s, FIG / "E5" / "perfiles_prueba.png", 0.6, 1.6, 5.3, 5.3)
caja(s, 6.1, 1.7, 6.6, 5.0, [
    "Azul: fase preprocesada. Naranja: fase corregida. Zona sombreada: célula.",
    "En la mayoría de los cortes el perfil corregido conserva la forma de la célula y baja su nivel.",
    "Cambio de altura en mediana: 4,81 %. No hay fondo verdadero: los perfiles son relativos.",
], size=18, espacio=14)

# ---------------------------------------------------------------- 12. Consistencia por imagen
s = nueva(13, "Consistencia por imagen en prueba", "Mateo Lopera",
          "Las cinco imágenes de prueba muestran que el resultado no es uniforme. En RMS, A supera a Zernike solo en "
          "la quinta imagen, con 4,71 frente a 5,65 niveles. En preservación, la supera solo en la tercera, con 2,88 "
          "frente a 4,81 por ciento. Ninguna imagen mejora las dos métricas a la vez, así que el criterio conjunto "
          "de la diapositiva 10 se cumple en cero de cinco. Con cinco imágenes no podemos afirmar que A sea peor en "
          "general; lo que sí podemos afirmar es que no demostró ser mejor.")
tabla(s, 0.6, 1.7, 12.1, [
    ["Imagen (hora · archivo)", "RMS A", "RMS Zernike", "|Δh| A (%)", "|Δh| Zernike (%)"],
    ["12:39:26 · 4-6", "6,92", "6,49", "5,77", "2,19"],
    ["12:55:21 · 1-5", "7,10", "6,52", "3,92", "0,82"],
    ["13:00:40 · 2-16", "9,36", "5,78", "2,88", "4,81"],
    ["13:06:04 · 1-5", "8,63", "7,22", "7,66", "2,94"],
    ["13:06:43 · 3-12", "4,71", "5,65", "4,81", "2,96"],
], anchos=[3.0, 1.8, 2.2, 2.2, 2.6], size=16, alto_fila=0.52, resaltar={3})
caja(s, 0.6, 4.95, 12.1, 1.9, [
    "A supera a Zernike en RMS en 1 de 5 imágenes y en |Δh| en 1 de 5; en ninguna en ambas (0 de 5).",
    "Con cinco imágenes, la variación entre ellas impide concluir sobre el orden general.",
], size=17, espacio=10)
texto(s, 0.6, 6.5, 12.1, 0.4, ["Imágenes del 29/01/2021; RMS en niveles equivalentes."], size=12, color=GRIS)

# ---------------------------------------------------------------- 14. Generalización a otro día
s = nueva(14, "Generalización a otro día de adquisición", "Mateo Lopera",
          "Aplicamos el mismo procedimiento, sin cambiar arquitectura, pérdida, iteraciones ni semilla, a ocho imágenes "
          "de tres adquisiciones del 28 de enero. La tabla compara los cinco métodos en los dos días. La preservación "
          "de A se mantiene dentro del umbral, con una mediana de 2,07 por ciento. Su fondo residual sube de 7,10 a "
          "11,79 niveles, con una dispersión de 7,26 niveles. Zernike sigue por delante de A en fondo en los dos días. "
          "Con tres adquisiciones no podemos separar el efecto del día de la calidad de la imagen, y la conclusión es "
          "limitada.")
tabla(s, 0.6, 1.7, 12.1, [
    ["Método", "M1 prueba", "M1 otro día", "|Δh| prueba (%)", "|Δh| otro día (%)"],
    ["Pistón", "18,18", "20,53", "0,00", "0,00"],
    ["Polinomio grado 3", "10,37", "9,29", "1,19", "2,01"],
    ["Zernike (guía)", "6,49", "5,98", "2,94", "4,60"],
    ["SIREN ω₀ = 1 (guía)", "4,22", "3,78", "12,27", "12,52"],
    ["A (propuesto)", "7,10", "11,79", "4,81", "2,07"],
], anchos=[3.4, 1.8, 1.9, 2.5, 2.5], size=14, alto_fila=0.4, resaltar={5})
caja(s, 0.6, 4.6, 6.2, 2.4, [
    "A no supera a Zernike en M1 en ningún día (7,10 frente a 6,49; 11,79 frente a 5,98).",
    "En |Δh| se invierte: peor que Zernike en prueba (4,81 frente a 2,94 %), mejor en otro día (2,07 frente a 4,60 %).",
    "Ocho imágenes, tres adquisiciones y dispersión de 7,26 niveles: la conclusión es limitada.",
], size=15, espacio=8)
imagen(s, FIG / "G10" / "ejemplo_otro_dia.png", 6.9, 4.7, 5.8, 1.6)
texto(s, 6.9, 6.4, 5.8, 0.4, ["Ejemplo de corrección en el otro día de adquisición."], size=12, color=GRIS)

# ---------------------------------------------------------------- 15. Compromiso fondo–célula en prueba
s = nueva(15, "El compromiso entre fondo y célula en prueba", "Mateo Lopera",
          "Esta figura muestra el conjunto de prueba, con A incluido. El eje horizontal es el cambio de altura de la "
          "célula: a la izquierda se preserva mejor. El eje vertical es el fondo residual: abajo está mejor. En fondo, A "
          "queda entre Zernike y el polinomio de grado tres. En preservación, queda por detrás de Zernike y del "
          "polinomio, así que Zernike lo supera en ambas métricas. En validación, A tenía una ligera ventaja de fondo "
          "sobre Zernike, y en prueba esa ventaja no se repitió.")
caja(s, 0.6, 1.7, 4.6, 5.1, [
    "Horizontal, |Δh|: a la izquierda, mejor preservación de la célula.",
    "Vertical, M1: abajo, mejor fondo residual.",
    "Pistón: preserva, pero deja el fondo.",
    "SIREN: quita el fondo, pero absorbe la célula (12 %).",
    "A queda entre Zernike y el polinomio en fondo, pero Zernike lo supera en ambas métricas.",
], size=15, espacio=10)
imagen(s, FIG / "E5" / "compromiso_fondo_celula_prueba.png", 5.4, 1.65, 7.4, 5.2)

# ---------------------------------------------------------------- 15. Conclusiones
s = nueva(16, "Conclusiones: el criterio de éxito no se cumplió", "Mateo Lopera",
          "Concluimos que el modelo no superó a la mejor línea base admisible en prueba. A preserva la célula y "
          "supera al pistón y al polinomio de grado tres, pero no a Zernike. El compromiso fondo–célula explica "
          "buena parte: quien quita más fondo también puede absorber célula, como la SIREN, y E preserva mejor pero "
          "deja más fondo. Además, cinco imágenes de validación y cinco de prueba son muy pocas. Lo aprendido: fijar "
          "el criterio antes de entrenar evitó declarar un éxito que los datos no sostienen, y detectar la fuga de la "
          "partición original obligó a rediseñar la evaluación.")
caja(s, 0.6, 1.7, 5.9, 4.1, [
    "Partición sin fuga y caracterización del dataset: 47,5 % de recortes compartidos en la original.",
    "A preserva la célula en prueba: |Δh| 4,81 %, dentro del umbral.",
    "En fondo, A supera al pistón y al polinomio de grado 3, pero no a Zernike (7,10 frente a 6,49 niveles).",
], size=17, espacio=12)
rectangulo(s, 6.9, 1.7, 5.8, 4.1, relleno=CLARO)
texto(s, 7.15, 1.85, 5.3, 0.55, ["Por qué no se cumplió"], size=17, color=ACENTO, bold=True)
caja(s, 7.15, 2.4, 5.4, 3.3, [
    "Compromiso fondo–célula: quitar más fondo también absorbe célula (SIREN).",
    "E preserva mejor (1,81 %) pero deja más fondo (8,63 niveles).",
    "Pocas imágenes: cinco de validación y cinco de prueba.",
], size=16, espacio=10)
texto(s, 0.6, 6.05, 12.1, 0.8, [
    "Lo aprendido: el criterio fijado antes de entrenar impidió declarar un éxito que los datos no sostienen."],
    size=16, color=ACENTO, bold=True)

# ---------------------------------------------------------------- 17. Limitaciones y trabajo futuro
s = nueva(17, "Limitaciones y trabajo futuro", "Mateo Lopera",
          "Cerramos con las limitaciones. No hay fondo verdadero, así que las máscaras y los anillos son referencias "
          "aproximadas. Son pocas adquisiciones, y las líneas base del código guía no tuvieron la misma búsqueda de "
          "hiperparámetros. Como trabajo futuro, proponemos más adquisiciones y diversidad, contrastar con referencias "
          "físicas y estudiar la sensibilidad a máscaras, resolución y ruido. Gracias; quedamos atentos a preguntas.")
caja(s, 0.6, 1.7, 6.0, 4.6, [
    "No hay fondo verdadero: máscaras y anillos son referencias aproximadas.",
    "Pocas adquisiciones: cinco imágenes de validación y cinco de prueba.",
    "Las líneas base del código guía no tuvieron búsqueda equivalente de hiperparámetros.",
    "La diferencia del otro día mezcla día de adquisición y calidad de imagen.",
], size=17, espacio=12)
caja(s, 6.9, 1.7, 5.8, 4.6, [
    "Ampliar adquisiciones y su diversidad, con una imagen por adquisición.",
    "Contrastar con referencias físicas independientes.",
    "Estudiar la sensibilidad a máscaras, resolución y ruido.",
], size=17, espacio=12)

# ---------------------------------------------------------------- 18. Referencias
s = nueva(18, "Referencias", "Mateo Lopera",
          "Estas son todas las referencias del informe, en orden de citación. El código guía del curso implementa las "
          "líneas base Zernike y SIREN.")
texto(s, 0.6, 1.7, 12.1, 5.2, [
    "[1] Castañeda, R., Trujillo, C. y Doblas, A. (2024). A human erythrocytes hologram dataset for learning-based model training. Data in Brief, 54, 110424.",
    "[2] Herráez, M. A., Burton, D. R., Lalor, M. J. y Gdeisat, M. A. (2002). Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path. Applied Optics, 41(35), 7437–7444.",
    "[3] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B. y Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. arXiv:2006.09661.",
    "[4] Tancik, M., et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. arXiv:2006.10739.",
    "[5] Tang, J., et al. (2023). Phase aberration compensation via a self-supervised sparse constraint network in digital holographic microscopy. Optics and Lasers in Engineering, 168, 107671.",
    "[6] Huang, Z. y Cao, L. (2023). Phase aberration separation for holographic microscopy by alternating direction sparse optimization. Optics Express, 31(8), 12520–12533.",
    "[7] Xu, D., Cao, L. y Sun, H. (2025). Adaptive aberration-corrected quantitative phase microscopy via physics-informed deep learning. Advanced Photonics, 7(6), 066013.",
    "[8] Código guía del curso (compensate_background, compensate_background_zernike), incorporado en el notebook EritroFase.",
], size=14, color=TEXTO)

# ---------------------------------------------------------------- Metadatos
cp = prs.core_properties
cp.author = "Grupo 1 – EritroFase"
cp.last_modified_by = "Grupo 1 – EritroFase"
cp.title = "Corrección del fondo en imágenes de fase de eritrocitos"
cp.subject = "Sustentación del proyecto final, SI3014, Universidad EAFIT 2026-2"
cp.keywords = "eritrocitos; imagen de fase; redes neuronales implícitas; SIREN"
cp.comments = ""
cp.category = "Presentación académica"
cp.revision = 1
ahora = datetime(2026, 11, 11, 9, 0, 0)
cp.created = ahora
cp.modified = ahora

APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Office PowerPoint</Application><PresentationFormat>Presentación en pantalla (16:9)</PresentationFormat><Slides>{n}</Slides><Notes>{n}</Notes><HiddenSlides>0</HiddenSlides><Company>Universidad EAFIT</Company></Properties>"""


def _corregir_app_xml(ruta):
    """La plantilla por defecto deja formato 4:3 y 0 diapositivas en docProps/app.xml."""
    import shutil
    import zipfile
    tmp = ruta.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(ruta) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            datos = zin.read(item.filename)
            if item.filename == "docProps/app.xml":
                datos = APP_XML.format(n=len(prs.slides)).encode("utf-8")
            zout.writestr(item, datos)
    shutil.move(str(tmp), str(ruta))


SALIDA.parent.mkdir(parents=True, exist_ok=True)
prs.save(SALIDA)
_corregir_app_xml(SALIDA)
print(f"Guardado: {SALIDA}")
print(f"Diapositivas: {len(prs.slides)}")
