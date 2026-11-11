"""Construye entregables/EritroFase_Informe.pdf a partir de docs/informe/informe.md.

Subconjunto de Markdown: títulos (#, ##), párrafos, listas con guion, tablas con
barras verticales, figuras ![pie](ruta){w=8cm} y matemáticas en $...$ (subconjunto LaTeX).
"""
import os
import re
import sys
from pathlib import Path

from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

BASE = Path(__file__).resolve().parent
MD = BASE / "informe.md"
OUT = BASE.parents[1] / "entregables" / "EritroFase_Informe.pdf"
MAX_PAGES = 3
METADATOS = {
    "/Title": "EritroFase: corrección del fondo en imágenes de fase de eritrocitos "
              "mediante redes neuronales informadas por la física",
    "/Author": "Grupo 1 – EritroFase",
    "/Subject": "Informe final · SI3014 Redes Neuronales y Aprendizaje Profundo",
    "/Creator": "Grupo 1 – EritroFase",
    "/Producer": "Grupo 1 – EritroFase",
}

FUENTES = "C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("TNR", f"{FUENTES}/times.ttf"))
pdfmetrics.registerFont(TTFont("TNR-B", f"{FUENTES}/timesbd.ttf"))
pdfmetrics.registerFont(TTFont("TNR-I", f"{FUENTES}/timesi.ttf"))
pdfmetrics.registerFont(TTFont("TNR-BI", f"{FUENTES}/timesbi.ttf"))
registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

ESTILO = {
    "cuerpo": ParagraphStyle("cuerpo", fontName="TNR", fontSize=10, leading=12.2,
                             alignment=TA_JUSTIFY, spaceAfter=3.5),
    "titulo": ParagraphStyle("titulo", fontName="TNR-B", fontSize=14, leading=16.5,
                             alignment=TA_CENTER, spaceAfter=4),
    "meta": ParagraphStyle("meta", fontName="TNR", fontSize=9.2, leading=11,
                           alignment=TA_CENTER, spaceAfter=1.5),
    "h2": ParagraphStyle("h2", fontName="TNR-B", fontSize=11.2, leading=13,
                         spaceBefore=6, spaceAfter=2.5),
    "lista": ParagraphStyle("lista", parent=None, fontName="TNR", fontSize=10,
                            leading=12.2, alignment=TA_JUSTIFY, leftIndent=12,
                            bulletIndent=2, spaceAfter=1.5),
    "ref": ParagraphStyle("ref", fontName="TNR", fontSize=8.6, leading=10.2,
                          leftIndent=14, firstLineIndent=-14, spaceAfter=1),
    "pie_figura": ParagraphStyle("pie_figura", fontName="TNR", fontSize=8.6,
                                 leading=10.2, alignment=TA_JUSTIFY, spaceAfter=5),
    "celda": ParagraphStyle("celda", fontName="TNR", fontSize=8.8, leading=10.3),
    "celda_cab": ParagraphStyle("celda_cab", fontName="TNR-B", fontSize=8.8, leading=10.3),
}

GRIEGO = {"varphi": "φ", "pi": "π", "Delta": "Δ", "omega": "ω", "mu": "μ",
          "sigma": "σ", "rho": "ρ"}
SIMBOLOS = {"le": "≤", "ge": "≥", "lvert": "|", "rvert": "|", "approx": "≈",
            "times": "×", "in": "∈", "cdot": "·", "ldots": "…", "operatorname": "",
            "text": "", "mathcal": "", "left": "", "right": "", "min": "mín"}


def escapar(texto):
    return texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def matematica(tex):
    """Convierte el subconjunto LaTeX usado en el informe a marcado de reportlab."""
    s = tex
    s = re.sub(r"\\t?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\hat\s*\{?(\w)\}?", r"\1ˆ", s)
    s = s.replace("\\mathcal U", "U").replace("\\%", "%")
    s = re.sub(r"\\Delta\s+h", "Δh", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"<super>\1</super>", s)
    s = re.sub(r"\^(\w)", r"<super>\1</super>", s)
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"_(\w)", r"<sub>\1</sub>", s)
    s = re.sub(r"\\([,;!])", " ", s)

    def comando(m):
        nombre = m.group(1)
        return GRIEGO.get(nombre, SIMBOLOS.get(nombre, nombre))

    s = re.sub(r"\\([A-Za-z]+)", comando, s)
    s = s.replace("{", "").replace("}", "")
    return s


def en_linea(texto):
    """Texto con negrita, cursiva, matemáticas y código en marcado de reportlab."""
    partes = re.split(r"(\$[^$]+\$)", texto)
    salida = []
    for parte in partes:
        if parte.startswith("$") and parte.endswith("$") and len(parte) > 2:
            salida.append(matematica(parte[1:-1]))
        else:
            t = escapar(parte.replace("`", ""))
            t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
            t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
            salida.append(t)
    return "".join(salida)


def leer_bloques(md):
    """Devuelve una lista de bloques (tipo, datos) a partir del Markdown."""
    bloques, parrafo, tabla = [], [], []
    visto_h2 = False

    def vaciar_parrafo():
        if parrafo:
            tipo = "meta" if not visto_h2 else "p"
            bloques.append((tipo, " ".join(parrafo)))
            parrafo.clear()

    def vaciar_tabla():
        if tabla:
            filas = []
            for linea in tabla:
                celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
                filas.append(celdas)
            alineaciones = [c.strip() for c in filas[1]]
            cabecera, cuerpo = filas[0], filas[2:]
            bloques.append(("tabla", (cabecera, cuerpo, alineaciones)))
            tabla.clear()

    for linea in md.splitlines():
        s = linea.strip()
        if s.startswith("|"):
            vaciar_parrafo()
            tabla.append(s)
            continue
        vaciar_tabla()
        if not s:
            vaciar_parrafo()
        elif s.startswith("# "):
            vaciar_parrafo()
            bloques.append(("titulo", s[2:]))
        elif s.startswith("## "):
            vaciar_parrafo()
            visto_h2 = True
            bloques.append(("h2", s[3:]))
        elif s.startswith("- "):
            vaciar_parrafo()
            bloques.append(("lista", s[2:]))
        elif re.match(r"!\[(.*)\]\((.*?)\)(\{w=([\d.]+)cm\})?$", s):
            vaciar_parrafo()
            m = re.match(r"!\[(.*)\]\((.*?)\)(\{w=([\d.]+)cm\})?$", s)
            ancho = float(m.group(4)) if m.group(4) else 8.0
            bloques.append(("figura", (m.group(1), m.group(2), ancho)))
        elif re.match(r"\[\d+\] ", s):
            vaciar_parrafo()
            bloques.append(("ref", s))
        else:
            parrafo.append(s)
    vaciar_parrafo()
    vaciar_tabla()
    return bloques


def tabla_flowable(cabecera, cuerpo, alineaciones, ancho_total):
    n = len(cabecera)
    ancho_primera = 5.0 * cm
    ancho_resto = (ancho_total - ancho_primera) / (n - 1)
    anchos = [ancho_primera] + [ancho_resto] * (n - 1)

    def celda(texto, estilo, alin):
        estilo_alin = ParagraphStyle(f"{estilo.name}-{alin}", parent=estilo,
                                     alignment={"l": 0, "c": 1, "r": 2}[alin])
        return Paragraph(en_linea(texto), estilo_alin)

    datos = [[celda(c, ESTILO["celda_cab"], "l" if i == 0 else "c")
              for i, c in enumerate(cabecera)]]
    for fila in cuerpo:
        datos.append([celda(c, ESTILO["celda"], "l" if i == 0 else _alin(alineaciones[i]))
                      for i, c in enumerate(fila)])
    t = Table(datos, colWidths=anchos, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, "black"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, "black"),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, "black"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _alin(marca):
    if marca.startswith(":") and marca.endswith(":"):
        return "c"
    if marca.endswith(":"):
        return "r"
    return "l"


def figura_flowable(pie, ruta, ancho_cm):
    ruta_abs = (BASE / ruta).resolve()
    with PILImage.open(ruta_abs) as im:
        w, h = im.size
    ancho = ancho_cm * cm
    alto = ancho * h / w
    return [Image(str(ruta_abs), width=ancho, height=alto),
            Paragraph(en_linea(pie), ESTILO["pie_figura"])]


def construir_historia(md, ancho_util):
    historia = []
    for tipo, datos in leer_bloques(md):
        if tipo == "titulo":
            historia.append(Paragraph(en_linea(datos), ESTILO["titulo"]))
        elif tipo == "meta":
            historia.append(Paragraph(en_linea(datos), ESTILO["meta"]))
        elif tipo == "h2":
            historia.append(Paragraph(en_linea(datos), ESTILO["h2"]))
        elif tipo == "p":
            historia.append(Paragraph(en_linea(datos), ESTILO["cuerpo"]))
        elif tipo == "lista":
            historia.append(Paragraph(en_linea(datos), ESTILO["lista"], bulletText="•"))
        elif tipo == "ref":
            historia.append(Paragraph(en_linea(datos), ESTILO["ref"]))
        elif tipo == "tabla":
            cab, cuerpo, alin = datos
            historia.append(Spacer(1, 2))
            historia.append(tabla_flowable(cab, cuerpo, alin, ancho_util))
            historia.append(Spacer(1, 4))
        elif tipo == "figura":
            pie, ruta, ancho = datos
            historia.append(Spacer(1, 2))
            historia.extend(figura_flowable(pie, ruta, ancho))
    return historia


def fijar_metadatos(ruta):
    lector = PdfReader(str(ruta))
    escritor = PdfWriter(clone_from=lector)
    escritor.add_metadata(METADATOS)
    temporal = ruta.with_suffix(".tmp.pdf")
    with open(temporal, "wb") as f:
        escritor.write(f)
    os.replace(temporal, ruta)


def pie_pagina(canvas, doc):
    canvas.saveState()
    canvas.setFont("TNR", 8)
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"EritroFase · Informe final · {doc.page}")
    canvas.restoreState()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = MD.read_text(encoding="utf-8")
    margen = 2.0 * cm
    ancho_util = A4[0] - 2 * margen
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=margen, rightMargin=margen,
                            topMargin=1.6 * cm, bottomMargin=1.9 * cm,
                            title=METADATOS["/Title"], author=METADATOS["/Author"],
                            subject=METADATOS["/Subject"], creator=METADATOS["/Creator"])
    doc.build(construir_historia(md, ancho_util), onFirstPage=pie_pagina,
              onLaterPages=pie_pagina)
    fijar_metadatos(OUT)
    paginas = len(PdfReader(str(OUT)).pages)
    print(f"PDF: {OUT}")
    print(f"Páginas: {paginas}")
    if paginas > MAX_PAGES:
        print(f"ERROR: el informe excede {MAX_PAGES} páginas", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
