"""Inserta en EritroFase.ipynb las celdas de código nuevas:
- 9.1b: figura de compromiso fondo / célula en prueba, con A (estrella).
- 10.1b: líneas base en el otro día de adquisición y tabla prueba vs otro día.
Las celdas de lectura (markdown) se añaden en un paso posterior, con las cifras ya calculadas.
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "EritroFase.ipynb"

CODIGO_9_1B = '''# 9.1b Figura: compromiso entre fondo residual y preservación de la célula en prueba (incluye A)
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_yscale("log")
for j, m in enumerate(ORDEN_PRUEBA):
    t = EVAL_PRUEBA[EVAL_PRUEBA["método"] == m]
    es_a = m == "A (propuesto)"
    circular = m == GRADO_LB
    ax.scatter(t[ADH], t[L], color=f"C{j}", alpha=0.45, s=36)
    etiqueta = m + (" (comparte ajuste con la referencia)" if circular else "")
    ax.scatter(RESUMEN_PRUEBA.loc[m, ADH], RESUMEN_PRUEBA.loc[m, L], color=f"C{j}",
               s=320 if es_a else 200, edgecolor="k", zorder=3, label=etiqueta,
               marker="*" if es_a else ("D" if circular else "o"))
ax.axvline(5, ls="--", color="gray", lw=1.2)
ax.text(5.2, ax.get_ylim()[1] * 0.92, "|Δh| = 5 %", color="gray", fontsize=11, va="top")
ax.set_xlabel("|Δh| mediana de la imagen (%)   ← mejor preservación de la célula", fontsize=12)
ax.set_ylabel("RMS del fondo lento residual (niveles eq., escala log)   ← mejor fondo", fontsize=12)
ax.tick_params(labelsize=11)
ax.set_title("Conjunto de prueba (5 imágenes): puntos pequeños = imágenes; marcadores grandes = medianas", fontsize=12)
ax.legend(fontsize=11, loc="upper right")
fig.tight_layout(); plt.show()
guardar_figura(fig, "E5", "compromiso_fondo_celula_prueba")
'''

CODIGO_10_1B = '''# 10.1b Líneas base en el otro día de adquisición: mismas imágenes y métricas que en prueba, con A incluido
METODOS_GEN = ["Pistón", GRADO_LB, "Zernike (guía)", "SIREN ω0=1 (guía)"]
ORDEN_GEN = METODOS_GEN + ["A (propuesto)"]
LB_GEN = {f: en_cache(f"G10_LB_{f[:-4]}", lambda f=f: _lineas_base_en_imagen(f)) for f in ARCHIVOS_GEN}
filas_gen = []
for f in ARCHIVOS_GEN:
    u = PREPROCESADAS[f]
    for metodo in METODOS_GEN:
        filas_gen.append({"archivo": f, "método": metodo,
                          **metricas_archivo(u - LB_GEN[f]["fondos"][metodo], f)})
    filas_gen.append({"archivo": f, "método": "A (propuesto)",
                      **metricas_archivo(u - AJUSTES_GEN[f]["fondo"], f)})
EVAL_GEN_TODOS = pd.DataFrame(filas_gen)
guardar_tabla(EVAL_GEN_TODOS, "G10_lineas_base_generalizacion_por_imagen")
G10_LINEAS_BASE = EVAL_GEN_TODOS.groupby("método")[[L, ADH]].median().loc[ORDEN_GEN]
guardar_tabla(G10_LINEAS_BASE, "G10_lineas_base_generalizacion", index=True)
PRUEBA_VS_GEN = pd.DataFrame({"M1 prueba": RESUMEN_PRUEBA[L], "M1 otro día": G10_LINEAS_BASE[L],
                              "|Δh| prueba (%)": RESUMEN_PRUEBA[ADH],
                              "|Δh| otro día (%)": G10_LINEAS_BASE[ADH]}).loc[ORDEN_GEN]
for columna in ["M1 prueba", "M1 otro día", "|Δh| prueba (%)", "|Δh| otro día (%)"]:
    PRUEBA_VS_GEN[f"orden {columna}"] = PRUEBA_VS_GEN[columna].rank(method="min").astype(int)
guardar_tabla(PRUEBA_VS_GEN, "G10_prueba_vs_generalizacion_por_metodo", index=True)
print("Mediana por método en el otro día de adquisición (8 imágenes; |Δh| usa solo las imágenes con células evaluables):")
display(G10_LINEAS_BASE.round(2))
print("Prueba frente a otro día. El orden va de 1 (mejor) al último; en M1 y |Δh| menor es mejor:")
display(PRUEBA_VS_GEN.round(2))
'''


def insertar_despues(nb, id_ancla: str, celda) -> None:
    posiciones = [i for i, c in enumerate(nb.cells) if c.get("id") == id_ancla]
    assert len(posiciones) == 1, f"ancla {id_ancla} no encontrada de forma única"
    nb.cells.insert(posiciones[0] + 1, celda)


def main() -> None:
    nb = nbformat.read(RUTA, as_version=4)
    assert not any(c.source.startswith("# 9.1b") for c in nb.cells), "La celda 9.1b ya existe"
    assert not any(c.source.startswith("# 10.1b") for c in nb.cells), "La celda 10.1b ya existe"
    # 9.1b tras el cierre de 9.1 (celda 'Límite del protocolo de prueba')
    insertar_despues(nb, "746740c1", new_code_cell(CODIGO_9_1B))
    # 10.1b tras la celda 10.1 de aplicación del modelo
    insertar_despues(nb, "9bddd2a1", new_code_cell(CODIGO_10_1B))
    nbformat.validate(nb)
    nbformat.write(nb, RUTA)
    print(f"Celdas totales: {len(nb.cells)}")


if __name__ == "__main__":
    main()
