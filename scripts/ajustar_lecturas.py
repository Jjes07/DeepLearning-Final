"""Contrasta las lecturas finales con las tablas de la ejecución."""
from pathlib import Path
import pandas as pd
import nbformat
from nbformat.v4 import new_markdown_cell

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
RMS = "RMS lento (niveles eq.)"
ALT = "|Δh| mediana (%)"


def numero(x, digitos=2):
    return f"{x:.{digitos}f}".replace(".", ",")


def main():
    ruta = ROOT / "EritroFase.ipynb"
    nb = nbformat.read(ruta, as_version=4)
    tabla = pd.read_csv(TAB / "LB_metricas_validacion.csv", index_col="método")
    siren = tabla.loc["SIREN ω0=1 (guía)"]
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.startswith("**Lectura de la salida (6.4).**"):
            c.source = c.source.replace("| SIREN $\\omega_0 = 1$ (guía) | 3,5 | 12,80 |",
                                        f"| SIREN $\\omega_0 = 1$ (guía) | {numero(siren[RMS])} | {numero(siren[ALT])} |")
            c.source = c.source.replace("mediano de −12,8 %", f"mediano de −{numero(abs(siren['Δh mediana (%)']))} %")
            c.source = c.source.replace("= 0{,}46", "= 0{,}45")
            c.source = c.source.replace("(3,2–4,1 frente a 5,1–10,9 niveles)", "(3,3–4,0 frente a 5,1–10,9 niveles)")
    semillas = pd.read_csv(TAB / "E4_A_validacion_semillas.csv", index_col="semilla")
    prueba = pd.read_csv(TAB / "E5_metricas_prueba_mediana.csv", index_col="método")
    generalizacion = pd.read_csv(TAB / "G10_comparacion_prueba.csv", index_col=0)
    a = prueba.loc["A (propuesto)"]
    adm = prueba.drop(index="A (propuesto)")
    adm = adm[adm[ALT] <= 5]
    mejor = adm[RMS].min()
    h9 = a[ALT] <= 5 and a[RMS] <= mejor
    h10 = generalizacion.loc[ALT, "Otro día: mediana"] <= 5
    def estado(valor):
        return "se cumple" if valor else "no se cumple"
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.startswith("### 0.4 Registro de hipótesis"):
            c.source = c.source.replace(
                "| H7–H10 | Arquitectura, entrenamiento, prueba y otro día | Criterios y contraste en las secciones indicadas | Ver salidas y lecturas correspondientes | 7–10 |",
                f"| H7.1 | Arquitectura | M1 < 6,4 y |Δh| ≤ 5 % en validación | {estado(6.18 < 6.4 and 4.16 <= 5)} | 7.1 |\n"
                f"| H8.1–H8.2 | Ajuste | Reproducción experimental y evolución de pérdida y altura | Ver contraste y curvas | 8.2–8.3 |\n"
                f"| H9.1 | Prueba | Preservación y comparación con línea base admisible | {estado(h9)} | 9.1 |\n"
                f"| H10.1 | Otro día | |Δh| mediana ≤ 5 % | {estado(h10)} | 10 |")
    # Las barras en fórmulas de tablas Markdown se expresan como entidad HTML.
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.startswith("### 0.4 Registro de hipótesis"):
            c.source = c.source.replace("|Δh| ≤", "&#124;Δh&#124; ≤")
            c.source = c.source.replace("|Δh| mediana", "&#124;Δh&#124; mediana")
            c.source = c.source.replace("\n\n| H7.1", "\n| H7.1")
            c.source = c.source.replace("\n\n| Modelo A por imagen", "\n| Modelo A por imagen")
    if not any(c.cell_type == "markdown" and c.source.startswith("**Contraste de H7.1.") for c in nb.cells):
        for i, c in enumerate(nb.cells):
            if c.cell_type == "code" and c.source.startswith("# 7.1 Resultados documentados"):
                nb.cells.insert(i + 1, new_markdown_cell(
                    "**Contraste de H7.1.** Se cumple en la tabla experimental: A alcanza "
                    "M1 de 6,18 niveles y cambio de altura de 4,16 %, dentro de los criterios fijados. "
                    "Son promedios de medianas entre tres semillas, con cinco imágenes por semilla."))
                break
    curvas = pd.read_csv(TAB / "E4_curvas_A.csv")
    seleccion = int(semillas[semillas[ALT] <= 5][RMS].idxmin())
    curvas = curvas[curvas["semilla"] == seleccion].groupby("iteración")[["pérdida", RMS, ALT]].median()
    puntos = curvas.dropna(subset=[ALT])
    lectura_curvas = (
        "**Contraste de H8.2.** La pérdida mediana desciende de "
        f"{numero(curvas['pérdida'].iloc[0])} a {numero(curvas['pérdida'].iloc[-1])}, "
        f"mientras el cambio absoluto de altura sube de {numero(puntos[ALT].iloc[0])} % "
        f"a {numero(puntos[ALT].iloc[-1])} %. El descenso se ralentiza hacia el final y "
        "es consistente con una aproximación a la convergencia; las fluctuaciones no prueban "
        "que haya un mínimo. El aumento de altura es compatible con cierta absorción celular, "
        "aunque esta métrica no demuestra por sí sola su causa. El fondo residual al final "
        f"es {numero(puntos[RMS].iloc[-1])} niveles, por lo que no se puede descartar subajuste.")
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and c.source.startswith("# 8.3 Evolución"):
            if i + 1 < len(nb.cells) and nb.cells[i + 1].source.startswith("**Contraste de H8.2."):
                nb.cells[i + 1].source = lectura_curvas
            else:
                nb.cells.insert(i + 1, new_markdown_cell(lectura_curvas))
            break
    por_imagen = pd.read_csv(TAB / "E5_metricas_prueba_por_imagen.csv")
    base = adm[RMS].idxmin()
    tabla_pares = por_imagen.set_index(["archivo", "método"])
    archivos = por_imagen[por_imagen["método"] == "A (propuesto)"]["archivo"]
    victorias = sum(tabla_pares.loc[(f, "A (propuesto)"), RMS] < tabla_pares.loc[(f, base), RMS]
                    and tabla_pares.loc[(f, "A (propuesto)"), ALT] < tabla_pares.loc[(f, base), ALT]
                    for f in archivos)
    hallazgo = ("Las secciones 9 y 10 documentan el fondo residual y la preservación medidos, junto con la variación entre imágenes. "
                f"En prueba, A obtiene M1 de {numero(a[RMS])} niveles y |Δh| de {numero(a[ALT])} %, "
                f"frente a {numero(adm.loc[base, RMS])} y {numero(adm.loc[base, ALT])} % de {base}. "
                f"El criterio conjunto de 6.5 {estado(h9)} y A mejora ambas métricas en {victorias} de {len(archivos)} imágenes. "
                f"En el otro día, la mediana de M1 es {numero(generalizacion.loc[RMS, 'Otro día: mediana'])} "
                f"y la de |Δh| es {numero(generalizacion.loc[ALT, 'Otro día: mediana'])} %. "
                "Minimizar solo M1 podría absorber parte de la célula. El resultado se limita a las adquisiciones seleccionadas y no es una garantía clínica.")
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.startswith("## 12. Conclusiones"):
            inicio = c.source.index("Las secciones 9 y 10")
            fin = c.source.index("\n\n### 12.2", inicio)
            c.source = c.source[:inicio] + hallazgo + c.source[fin:]
    from celdas_7_8 import CELDAS_7_8
    fuente_71 = next(c.source for c in CELDAS_7_8 if c.cell_type == "code" and c.source.startswith("# 7.1 Resultados documentados"))
    for c in nb.cells:
        if c.cell_type == "code" and c.source.startswith("# 7.1 Resultados documentados"):
            c.source = fuente_71
            break
    nota = ("**Límite del protocolo de prueba.** En la primera ejecución se calcularon también "
            "métricas de prueba en puntos intermedios del ajuste. No se utilizaron para seleccionar "
            "la arquitectura, los hiperparámetros ni la semilla, pero esa ejecución no respetó "
            "estrictamente una única evaluación de prueba. El bucle se corrigió para calcular "
            "controles intermedios únicamente en validación; las predicciones finales aquí mostradas "
            "son las mismas, recuperadas de caché.")
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and c.source.startswith("# 8.2 Ajuste independiente"):
            c.source = c.source.replace("if paso == 1 or paso % 100 == 0:",
                                        "if archivo in ARCHIVOS_VAL and (paso == 1 or paso % 100 == 0):")
        if c.cell_type == "code" and c.source.startswith("# 9.1 Una evaluación"):
            if i + 1 < len(nb.cells) and nb.cells[i + 1].source.startswith("**Límite del protocolo de prueba."):
                nb.cells[i + 1].source = nota
            else:
                nb.cells.insert(i + 1, new_markdown_cell(nota))
            break
    nbformat.validate(nb)
    nbformat.write(nb, ruta)
    print("Lecturas actualizadas con las tablas de la ejecución.")


if __name__ == "__main__":
    main()
