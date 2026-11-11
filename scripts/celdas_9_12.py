"""Celdas de evaluación y discusión para incorporar con nbformat."""
from nbformat.v4 import new_markdown_cell as md, new_code_cell as code

CELDAS_9_12 = [
md(r'''## 9. Resultados en el conjunto de prueba  *(lineamiento C · etapa E5)*

### 9.1 Comparación con las líneas base

**H9.1 (antes de evaluar).** El procedimiento seleccionado en validación conserva la altura con mediana de $|\Delta h| \leq 5\,\%$ y obtiene M1 no mayor que la mejor línea base admisible en prueba. Además, se contará en cuántas imágenes reduce simultáneamente M1 y $|\Delta h|$ frente a esa línea base (6.5). Se usan las mismas máscaras y métricas para todos los métodos. La semilla y los hiperparámetros se fijaron en validación; no se seleccionan con prueba. El ajuste de A se hace por imagen, sin máscaras; es una evaluación del procedimiento, no una inferencia con pesos universales.'''),
code(r'''# 9.1 Una evaluación con el procedimiento ya fijado
ARCHIVOS_PRUEBA = SELECCION.loc[SELECCION.conjunto == "prueba", "archivo"].tolist()
assert len(ARCHIVOS_PRUEBA) == 5
METODOS_PRUEBA = ["Pistón", GRADO_LB, "Zernike (guía)", "SIREN ω0=1 (guía)"]
AJUSTES_PRUEBA = {f: en_cache(f"E5_A_{SEMILLA_ELEGIDA}_{f[:-4]}",
                               lambda f=f: ajustar_imagen(f, SEMILLA_ELEGIDA)) for f in ARCHIVOS_PRUEBA}
LB_PRUEBA = {f: en_cache(f"E5_LB_{f[:-4]}", lambda f=f: _lineas_base_en_imagen(f)) for f in ARCHIVOS_PRUEBA}
filas_prueba = []
for f in ARCHIVOS_PRUEBA:
    u = PREPROCESADAS[f]
    for metodo in METODOS_PRUEBA:
        filas_prueba.append({"archivo": f, "método": metodo,
                             **metricas_archivo(u - LB_PRUEBA[f]["fondos"][metodo], f)})
    filas_prueba.append({"archivo": f, "método": "A (propuesto)",
                         **metricas_archivo(u - AJUSTES_PRUEBA[f]["fondo"], f)})
EVAL_PRUEBA = pd.DataFrame(filas_prueba)
ORDEN_PRUEBA = METODOS_PRUEBA + ["A (propuesto)"]
RESUMEN_PRUEBA = EVAL_PRUEBA.groupby("método")[COLS_M].median().loc[ORDEN_PRUEBA]
DISPERSION_PRUEBA = EVAL_PRUEBA.groupby("método")[COLS_M].std().loc[ORDEN_PRUEBA]
guardar_tabla(EVAL_PRUEBA, "E5_metricas_prueba_por_imagen")
guardar_tabla(RESUMEN_PRUEBA, "E5_metricas_prueba_mediana", index=True)
guardar_tabla(DISPERSION_PRUEBA, "E5_metricas_prueba_desviacion", index=True)
print("Mediana por método en prueba; la desviación muestral entre imágenes se muestra aparte:")
display(RESUMEN_PRUEBA.round(2)); display(DISPERSION_PRUEBA.round(2))
admisibles = [m for m in METODOS_PRUEBA if RESUMEN_PRUEBA.loc[m, ADH] <= 5]
assert admisibles, "No hay una línea base admisible para aplicar el criterio 6.5"
LB_CRITERIO = min(admisibles, key=lambda m: RESUMEN_PRUEBA.loc[m, L])
comp = EVAL_PRUEBA.set_index(["archivo", "método"])
VICTORIAS = sum((comp.loc[(f, "A (propuesto)"), L] < comp.loc[(f, LB_CRITERIO), L]) and
                (comp.loc[(f, "A (propuesto)"), ADH] < comp.loc[(f, LB_CRITERIO), ADH])
                for f in ARCHIVOS_PRUEBA)
EXITO_PRUEBA = bool(RESUMEN_PRUEBA.loc["A (propuesto)", ADH] <= 5 and
                    RESUMEN_PRUEBA.loc["A (propuesto)", L] <= RESUMEN_PRUEBA.loc[LB_CRITERIO, L])
print(f"Lectura de la salida (9.1). La referencia admisible es {LB_CRITERIO}: "
      f"M1 = {RESUMEN_PRUEBA.loc[LB_CRITERIO, L]:.2f} niveles y |Δh| = {RESUMEN_PRUEBA.loc[LB_CRITERIO, ADH]:.2f} %. "
      f"A obtiene {RESUMEN_PRUEBA.loc['A (propuesto)', L]:.2f} niveles y "
      f"{RESUMEN_PRUEBA.loc['A (propuesto)', ADH]:.2f} %. "
      f"El criterio de medianas {'se cumple' if EXITO_PRUEBA else 'no se cumple'}; "
      f"A mejora ambas métricas en {VICTORIAS} de {len(ARCHIVOS_PRUEBA)} imágenes.")'''),
md(r'''### 9.2 Perfiles transversales

Se usa una fila que atraviesa la máscara celular de cada imagen; la coordenada horizontal es la misma antes y después. La máscara se usa aquí solo para elegir y señalar el corte, no durante el ajuste. La comparación muestra perfiles relativos en niveles equivalentes, no un error respecto a una verdad de fondo desconocida.'''),
code(r'''# 9.2 Perfiles sobre los mismos píxeles
fig, ejes = plt.subplots(len(ARCHIVOS_PRUEBA), 1, figsize=(10, 2.7 * len(ARCHIVOS_PRUEBA)), squeeze=False)
for ax, f in zip(ejes[:, 0], ARCHIVOS_PRUEBA):
    u = PREPROCESADAS[f]; cel, _ = MASCARAS[f]
    fila = int(np.argmax(cel.sum(axis=1)))
    x = np.arange(u.shape[1]); v = u - AJUSTES_PRUEBA[f]["fondo"]
    ax.plot(x, u[fila], label="Preprocesada", alpha=0.8)
    ax.plot(x, v[fila], label="Corregida", alpha=0.9)
    ax.fill_between(x, 0, 1, where=cel[fila], transform=ax.get_xaxis_transform(),
                    color="tab:orange", alpha=0.14, label="Máscara celular")
    ax.set(title=f"{f}: corte y = {fila} píxeles", xlabel="x (píxeles)", ylabel="Fase (niveles eq.)")
    ax.legend(loc="best")
fig.tight_layout(); plt.show(); guardar_figura(fig, "E5", "perfiles_prueba")
print("Lectura de la salida (9.2). Cada par de curvas corresponde a los mismos píxeles; "
      "la franja señala la referencia celular. La diferencia entre curvas incluye el fondo estimado "
      "y no permite por sí sola establecer el error sin un fondo verdadero.")'''),
md(r'''### 9.3 Entrada, fondo y corrección

La última columna es una copia visual de la corrección, con la máscara de referencia y negro fuera de ella. Las métricas y perfiles se calculan sobre el mapa corregido completo, sin recorte.'''),
code(r'''# 9.3 La máscara se aplica solo a una copia de visualización
fig, axes = plt.subplots(len(ARCHIVOS_PRUEBA), 4, figsize=(14, 3.1 * len(ARCHIVOS_PRUEBA)), squeeze=False)
for i, f in enumerate(ARCHIVOS_PRUEBA):
    u = PREPROCESADAS[f]; b = AJUSTES_PRUEBA[f]["fondo"]; v = u - b
    copia_visual = np.ma.masked_where(~MASCARAS[f][0], v)
    mapas = [u, b, v, copia_visual]
    escala = plt.get_cmap("gray").copy()
    escala.set_bad("black")
    for j, (a, titulo) in enumerate(zip(mapas, ["Entrada", "Fondo estimado", "Corregida", "Visualización en negro"])):
        limites = {"vmin": float(np.min(v)), "vmax": float(np.max(v))} if j in (2, 3) else {}
        im = axes[i, j].imshow(a, cmap=escala, **limites)
        axes[i, j].set(title=f"{f}\n{titulo}", xlabel="x (píxeles)", ylabel="y (píxeles)")
        cb = fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
        cb.set_label("Niveles equivalentes")
fig.tight_layout(); plt.show(); guardar_figura(fig, "E5", "mapas_prueba")
print("Lectura de la salida (9.3). El fondo negro es una representación con máscara; "
      "no modifica el mapa cuantitativo. La inspección visual complementa las métricas locales.")'''),
md(r'''### 9.4 Errores, fortalezas y límites

El caso con mayor M1 y el de mayor cambio absoluto de altura no tienen por qué coincidir. La comparación por imagen permite distinguir fondo residual de posible absorción celular. No hay fondo verdadero para medir el error absoluto del mapa de fondo; las máscaras y anillos son referencias aproximadas y algunas imágenes contienen pocas células evaluables.'''),
code(r'''# 9.4 Casos extremos observados
casos = EVAL_PRUEBA[EVAL_PRUEBA["método"] == "A (propuesto)"].set_index("archivo")
guardar_tabla(casos.reset_index(), "E5_casos_modelo")
display(casos[[L, DH, ADH, "células evaluadas"]].round(2))
peor_fondo, peor_altura = casos[L].idxmax(), casos[ADH].idxmax()
print(f"Lectura de la salida (9.4). El mayor fondo lento corresponde a {peor_fondo} "
      f"({casos.loc[peor_fondo, L]:.2f} niveles); el mayor cambio absoluto de altura a {peor_altura} "
      f"({casos.loc[peor_altura, ADH]:.2f} %). Estos resultados son descriptivos para las imágenes evaluadas.")'''),
md(r'''## 10. Generalización a otro día de adquisición  *(lineamiento D)*

**H10.1 (antes de evaluar).** El mismo procedimiento, con arquitectura, pérdida, iteraciones y semilla fijadas, mantendrá la mediana de $|\Delta h|$ en el umbral de 6.5. Se compara M1 y preservación con prueba, sin cambiar parámetros. Como A se optimiza por imagen, «sin cambiar» se refiere al procedimiento y sus hiperparámetros: se ajustan pesos nuevos a cada imagen sin utilizar sus máscaras.

Las imágenes del 28/01 pertenecen a otro día y se eligieron por defectos visibles, incluidas manchas de envolvimiento en el fondo y células envueltas (3.6–3.7). Por ello, una diferencia con prueba puede reflejar simultáneamente día, calidad y selección, y no identifica una causa única.'''),
code(r'''# 10.1 Aplicación con los mismos hiperparámetros
ARCHIVOS_GEN = SELECCION.loc[SELECCION.conjunto == "generalizacion", "archivo"].tolist()
assert len(ARCHIVOS_GEN) == 8
AJUSTES_GEN = {f: en_cache(f"G10_A_{SEMILLA_ELEGIDA}_{f[:-4]}",
                           lambda f=f: ajustar_imagen(f, SEMILLA_ELEGIDA)) for f in ARCHIVOS_GEN}
EVAL_GEN = pd.DataFrame([{"archivo": f, **metricas_archivo(PREPROCESADAS[f] - AJUSTES_GEN[f]["fondo"], f)}
                         for f in ARCHIVOS_GEN])
guardar_tabla(EVAL_GEN, "G10_metricas_por_imagen")
modelo_prueba = EVAL_PRUEBA[EVAL_PRUEBA["método"] == "A (propuesto)"]
COMPARACION_GEN = pd.DataFrame({"Prueba: mediana": modelo_prueba[COLS_M].median(),
                                "Prueba: desviación": modelo_prueba[COLS_M].std(),
                                "Otro día: mediana": EVAL_GEN[COLS_M].median(),
                                "Otro día: desviación": EVAL_GEN[COLS_M].std()})
guardar_tabla(COMPARACION_GEN, "G10_comparacion_prueba", index=True)
print("Medianas y desviaciones muestrales entre imágenes:")
display(COMPARACION_GEN.round(2)); display(EVAL_GEN[["archivo", L, DH, ADH, "células evaluadas"]].round(2))
print(f"Lectura de la salida (10). M1 pasa de {COMPARACION_GEN.loc[L, 'Prueba: mediana']:.2f} a "
      f"{COMPARACION_GEN.loc[L, 'Otro día: mediana']:.2f} niveles; |Δh| pasa de "
      f"{COMPARACION_GEN.loc[ADH, 'Prueba: mediana']:.2f} a {COMPARACION_GEN.loc[ADH, 'Otro día: mediana']:.2f} %. "
      f"H10.1 {'se cumple' if COMPARACION_GEN.loc[ADH, 'Otro día: mediana'] <= 5 else 'se rechaza'} "
      "para esta muestra. El cambio es consistente con diferencias de adquisición y calidad, sin demostrar causalidad.")'''),
code(r'''# 10.2 Comparación y ejemplo del otro día
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metrica, unidad in zip(axes, [L, ADH], ["M1 (niveles equivalentes)", "|Δh| (%)"]):
    ax.boxplot([modelo_prueba[metrica].dropna(), EVAL_GEN[metrica].dropna()], tick_labels=["Prueba", "Otro día"])
    ax.set(title=f"Comparación de {unidad}", ylabel=unidad, xlabel="Conjunto")
fig.tight_layout(); plt.show(); guardar_figura(fig, "G10", "comparacion_prueba")
f = ARCHIVOS_GEN[0]; u = PREPROCESADAS[f]; b = AJUSTES_GEN[f]["fondo"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, mapa, titulo in zip(axes, [u, b, u-b], ["Entrada", "Fondo estimado", "Corregida"]):
    im = ax.imshow(mapa, cmap="gray"); ax.set(title=f"{f}: {titulo}", xlabel="x (píxeles)", ylabel="y (píxeles)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Niveles equivalentes")
fig.tight_layout(); plt.show(); guardar_figura(fig, "G10", "ejemplo_otro_dia")
print("Lectura de la salida (10.2). Las distribuciones muestran variabilidad entre imágenes; "
      "el ejemplo ilustra el procedimiento y no representa por sí solo todo el conjunto.")'''),
md(r'''## 11. Compromiso entre desempeño, complejidad y tiempo

Las métricas de las líneas base y A son medianas de prueba. Los tiempos de A y AE proceden de la medición controlada de 8.4 en las imágenes de validación; AE se incluye como referencia exploratoria de validación: sus métricas son medias de las medianas por semilla, según la tabla 7.1; no se evaluó en prueba. Los tiempos de las líneas base corresponden al cálculo completo en prueba y se muestran por separado: no constituyen una comparación controlada con A o AE. Los coeficientes analíticos no son parámetros entrenables de una red.'''),
code(r'''# 11.1 Complejidad y tiempos por imagen
complejidad = {"Pistón": "0 entrenables; 1 coeficiente", GRADO_LB: "0 entrenables; coeficientes polinomiales",
               "Zernike (guía)": "0 entrenables; coeficientes de Zernike", "SIREN ω0=1 (guía)": "no contabilizados"}
filas_coste = []
for m in METODOS_PRUEBA:
    filas_coste.append({"método": m, "M1 (niveles eq.)": RESUMEN_PRUEBA.loc[m, L],
        "|Δh| (%)": RESUMEN_PRUEBA.loc[m, ADH], "conjunto métricas": "prueba",
        "complejidad": complejidad[m], "ajuste (s/imagen)": np.nan, "inferencia (s/imagen)": np.nan,
        "mapa E (s/imagen)": np.nan,
        "cálculo completo (s/imagen)": np.median([LB_PRUEBA[f]["tiempos"][m] for f in ARCHIVOS_PRUEBA]),
        "conjunto tiempo": "prueba; sin control comparable"})
ae_validacion = alternativas.loc[alternativas["Esquema"] == "AE"].iloc[0]
for metodo, medida in TIEMPOS_CONTROLADOS.iterrows():
    es_a = metodo == "A"
    filas_coste.append({"método": "A (propuesto)" if es_a else "AE (exploratorio)",
        "M1 (niveles eq.)": RESUMEN_PRUEBA.loc["A (propuesto)", L] if es_a else ae_validacion["M1"],
        "|Δh| (%)": RESUMEN_PRUEBA.loc["A (propuesto)", ADH] if es_a else ae_validacion["|Δh| (%)"],
        "conjunto métricas": "prueba; mediana" if es_a else "validación; media de medianas por semilla",
        "complejidad": str(medida["parámetros entrenables"]),
        "ajuste (s/imagen)": medida["ajuste (s/imagen)"],
        "inferencia (s/imagen)": medida["inferencia (s/imagen)"],
        "mapa E (s/imagen)": medida["mapa E (s/imagen)"],
        "cálculo completo (s/imagen)": medida["total (s/imagen)"],
        "conjunto tiempo": "validación; medición 8.4"})
COSTE = pd.DataFrame(filas_coste).set_index("método")
guardar_tabla(COSTE, "E5_compromiso_coste", index=True); display(COSTE.round(4))
print("Lectura de la salida (11). El ajuste por imagen forma parte del costo operativo de A y AE. "
      "Los tiempos controlados se midieron en validación; las métricas de prueba son independientes de ellos. "
      "Las celdas vacías indican cantidades no medidas o no aplicables.")'''),
md(r'''## 12. Conclusiones  *(lineamiento E)*

### 12.1 Hallazgos

Las secciones 9 y 10 documentan el fondo residual y la preservación medidos, junto con la variación entre imágenes. La interpretación depende tanto de las dos métricas como de los perfiles: minimizar solo M1 podría absorber parte de la célula. El resultado de prueba es una evidencia limitada a las adquisiciones seleccionadas, no una garantía clínica.

### 12.2 Cumplimiento de los objetivos

| Objetivo de 1.4 | Evidencia | Estado |
|---|---|---|
| General: estimar y compensar el fondo preservando la célula | Arquitectura y ajuste (7–8), métricas, perfiles y visualización (9–10) | Evaluado; el cumplimiento cuantitativo se contrasta en 9.1 |
| Específico 1: fundamentos y restricciones | SIREN y supuestos físicos (4, 7–8) | Abordado |
| Específico 2: caracterizar y seleccionar datos | Adquisiciones, selección y partición (3) | Abordado |
| Específico 3: validar fondo y perfiles | Métricas, referencias, caso sintético y perfiles (6, 9–10) | Evaluado con las limitaciones de referencia indicadas |

### 12.3 Limitaciones y trabajo futuro

Las máscaras son aproximadas, no existe un fondo verdadero para las imágenes reales y hay pocas adquisiciones evaluadas. La selección de hiperparámetros usó validación; las líneas base del código guía no recibieron una búsqueda equivalente. La diferencia entre conjuntos de prueba y otro día mezcla calidad de imagen y adquisición. Conviene ampliar el número y diversidad de adquisiciones, contrastar con referencias físicas y estudiar la sensibilidad a máscaras, resolución y ruido. Los esquemas E y AE, evaluados en validación, son direcciones prometedoras para investigar pesos espaciales y sus límites antes de evaluar de nuevo en datos independientes.''')
]
