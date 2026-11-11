"""Celdas de arquitectura y ajuste para integrar con nbformat."""
from nbformat.v4 import new_markdown_cell as md, new_code_cell as code

CELDAS_7_8 = [
md(r'''## 7. Arquitectura del modelo

### 7.1 Alternativas evaluadas y decisión

**H7.1 (antes de comparar).** Una representación de baja frecuencia ajustada por imagen con restricciones de signo y referencia puede alcanzar M1 menor que 6,4 niveles y conservar $|\Delta h|\leq5\,\%$ en validación. Cada observación es la mediana de las cinco imágenes; para A, E y AE se muestran media y desviación muestral de esas medianas entre tres semillas. C y D se ensayaron solamente con la semilla 42.

La tabla resume `experimentos/INFORME_EXPERIMENTOS.md`: A representa el fondo mediante una red implícita por imagen; C predice coeficientes de una base de forma amortizada; D separa fondo y célula en dos ramas; E aprende pesos para mínimos cuadrados; AE usa esos pesos para orientar un ajuste implícito. El informe registra 59 configuraciones o ejecuciones de A y 2 de A+, 3 de C, 10 de D, 6 de E y 4 variantes de AE; estos conteos no equivalen a arquitecturas independientes.'''),
code(r'''# 7.1 Resultados documentados
alternativas = pd.DataFrame([
    ("A", 3, 6.18, 0.18, 4.16, 0.56), ("C", 1, 19.52, np.nan, 0.76, np.nan),
    ("D", 1, 26.02, np.nan, 0.03, np.nan), ("E", 3, 8.63, 0.87, 1.81, 0.23),
    ("AE", 3, 6.16, 0.09, 4.97, 0.57),
], columns=["Esquema", "Semillas", "M1", "desv. M1", "|Δh| (%)", "desv. |Δh| (%)"])
guardar_tabla(alternativas, "M7_alternativas_validacion")
display(alternativas)
busqueda = pd.DataFrame([("A", 59), ("A+", 2), ("C", 3), ("D", 10), ("E", 6), ("AE", 4)],
                        columns=["Esquema", "Configuraciones o ejecuciones registradas"])
guardar_tabla(busqueda, "M7_busqueda_experimental")
display(busqueda)
print("Lectura de la salida (7.1). A y AE presentan valores de M1 próximos; A conserva mejor la altura media. E conserva altura, pero deja más fondo lento. C y D son resultados de una sola semilla.")'''),
md(r'''La pérdida de posición por promedio global es una explicación plausible de las dificultades de C; en D se observó colapso de la rama de célula. Las implementaciones de esa ronda también presentaron mezcla de imágenes en el lote y errores de broadcasting, y D empleó la mediana del lote: los resultados no aíslan únicamente el efecto de la arquitectura. E aportó una estimación de pesos espaciales aprendidos y AE combinó esos pesos con una red implícita.

Se adopta A por su compromiso entre fondo y altura y porque no requiere una segunda red. La ventaja de tiempo debe comprobarse con medición controlada. El informe también seleccionó AE bajo otra regla; aquí se prioriza preservación y sencillez. La búsqueda se hizo sobre cinco imágenes de validación, por lo que la selección puede favorecer esas imágenes.

### 7.2 Modelo final

> **Definición 16 (fondo implícito).** Para coordenadas normalizadas $(x,y)\in[-1,1]^2$, una red $b_\theta(x,y)$ predice un nivel equivalente de fondo. La imagen corregida es $u-b_\theta$. Los pesos $\theta$ se ajustan de nuevo para cada imagen.

La entrada tiene dimensión $N\times2$ y la salida $N\times1$, con $N=H W$. Una capa senoidal de entrada y dos ocultas de 128 neuronas transforman las coordenadas; la capa final es lineal. Se parte de SIREN (Sitzmann et al., 2020, *Implicit Neural Representations with Periodic Activation Functions*) y del código guía del proyecto. Se usa frecuencia inicial y oculta $\omega_0=1$ y se incorpora la pérdida física de 8.1.'''),
code(r'''# 7.2 Diagrama de arquitectura
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(12, 2.8))
etiquetas = ["Coordenadas\nN × 2", "Seno de entrada\n2 → 128", "Seno oculto 1\n128 → 128", "Seno oculto 2\n128 → 128", "Salida lineal\n128 → 1"]
for i, etiqueta in enumerate(etiquetas):
    caja = FancyBboxPatch((i * 2.15, .35), 1.8, .8, boxstyle="round,pad=0.1", facecolor="aliceblue", edgecolor="steelblue")
    ax.add_patch(caja); ax.text(i * 2.15 + .9, .75, etiqueta, ha="center", va="center")
    if i: ax.annotate("", xy=(i * 2.15, .75), xytext=(i * 2.15 - .35, .75), arrowprops={"arrowstyle": "->"})
ax.set(xlim=(-.2, 10.5), ylim=(0, 1.5), title="SIREN de baja frecuencia para el fondo por imagen")
ax.axis("off"); fig.tight_layout(); plt.show(); guardar_figura(fig, "E3", "arquitectura_A")
print("Lectura de la salida (7.2). Cada fila de coordenadas produce un valor de fondo; el número de filas es el número de píxeles.")'''),
md(r'''## 8. Entrenamiento y selección

### 8.1 Pérdida e hiperparámetros

> **Definición 17 (pérdida de A).** Sea $r=u-b_\theta$ y $K=\lfloor0{,}7N\rfloor$. La pérdida es
> $$L=\underbrace{\frac1K\sum_{i=1}^{K}|r|_{(i)}}_{L_{\rm rec}} +0{,}8\left[\operatorname{media}(\operatorname{ReLU}(-r)^2)+0{,}01\operatorname{media}(\operatorname{ReLU}(r))\right]+0{,}1\operatorname{mediana}(r)^2.$$
> $|r|_{(i)}$ son los valores absolutos ordenados de menor a mayor. Es la esparsidad recortada de la Definición 9.

El primer término implementa S2, el segundo favorece el signo de S3 y el tercero fija una referencia de nivel relacionada con S4. La suavidad S1 se favorece mediante la arquitectura de baja frecuencia, no con un laplaciano explícito. Se usa Adam con tasa $10^{-3}$ durante 1000 iteraciones fijas; no hay parada temprana. El ajuste no utiliza máscaras; estas se emplean después para evaluación.'''),
code(r'''# 8.1 Arquitectura y pérdida autocontenidas
if DEVICE.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.75, 0)

class CapaSeno(torch.nn.Module):
    def __init__(self, entradas, salidas, primera=False, omega=1.0):
        super().__init__(); self.omega = omega
        self.lineal = torch.nn.Linear(entradas, salidas)
        limite = 1 / entradas if primera else np.sqrt(6 / entradas) / omega
        with torch.no_grad(): self.lineal.weight.uniform_(-limite, limite)
    def forward(self, x):
        return torch.sin(self.omega * self.lineal(x))

class FondoSIREN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(CapaSeno(2, 128, True), CapaSeno(128, 128), CapaSeno(128, 128))
        self.salida = torch.nn.Linear(128, 1)
    def forward(self, coords):
        return self.salida(self.net(coords)).squeeze(-1)

def perdida_A(residuo):
    ordenados = torch.sort(torch.abs(residuo)).values
    rec = ordenados[:max(1, int(.7 * residuo.numel()))].mean()
    signo = torch.relu(-residuo).square().mean() + .01 * torch.relu(residuo).mean()
    referencia = residuo.median().square()
    return rec + .8 * signo + .1 * referencia

CONFIG_A = {"modelo": "A_cand_signo0.8_ref0.1_rec0.7", "omega_0": 1.0, "neuronas": 128,
            "capas_ocultas": 2, "iteraciones": 1000, "optimizador": "Adam", "lr": 0.001,
            "rho": 0.7, "signo": 0.8, "signo_neg": 1.0, "signo_pos": 0.01,
            "referencia": 0.1, "laplaciano": 0.0, "semillas": [42, 2024, 12345]}
RUN_A = RUNS_DIR / "A_rec07_signo08_ref01"
RUN_A.mkdir(parents=True, exist_ok=True)
(RUN_A / "config.json").write_text(json.dumps(CONFIG_A, ensure_ascii=False, indent=2), encoding="utf-8")
NUM_PARAMETROS_A = sum(p.numel() for p in FondoSIREN().parameters() if p.requires_grad)
assert NUM_PARAMETROS_A == 33537
print("Parámetros entrenables:", NUM_PARAMETROS_A)
display(pd.DataFrame([CONFIG_A]))
print("Lectura de la salida (8.1). La red tiene", NUM_PARAMETROS_A, "parámetros entrenables por imagen; la configuración se guardó en", RUN_A / "config.json")'''),
md(r'''### 8.2 Ajuste por imagen y semillas

**H8.1 (antes de ejecutar).** La configuración elegida reproducirá aproximadamente la media experimental de M1 y $|\Delta h|$ indicada en 7.1 y mantendrá $|\Delta h|\leq5\,\%$ para cada una de las tres semillas. Las cinco imágenes de validación se ajustan por separado, cada una desde sus propios pesos iniciales. No hay pesos compartidos entrenados sobre las doce imágenes de entrenamiento: la generalización es la del procedimiento con hiperparámetros fijados en validación. Para la evaluación posterior se selecciona la semilla admisible de menor M1 mediano en validación; si ninguna fuera admisible, se elige la de menor $|\Delta h|$.'''),
code(r'''# 8.2 Ajuste independiente por imagen
# La caché nueva permanece dentro de la carpeta de corridas de esta rama.
CACHE_DIR = RUNS_DIR / "cache_nuevas"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def sincronizar_A():
    if DEVICE.type == "cuda": torch.cuda.synchronize()

def ajustar_imagen(archivo, semilla):
    u = PREPROCESADAS[archivo]
    h, w = u.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=DEVICE), torch.linspace(-1, 1, w, device=DEVICE), indexing="ij")
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    objetivo = torch.tensor(u, dtype=torch.float32, device=DEVICE).flatten()
    fijar_semillas(semilla)
    modelo = FondoSIREN().to(DEVICE)
    fijar_semillas(semilla)
    optim = torch.optim.Adam(modelo.parameters(), lr=CONFIG_A["lr"])
    curva = []
    sincronizar_A(); inicio = time.perf_counter()
    for paso in range(1, CONFIG_A["iteraciones"] + 1):
        optim.zero_grad()
        prediccion = modelo(coords)
        perdida = perdida_A(objetivo - prediccion)
        perdida.backward(); optim.step()
        fila = {"iteración": paso, "pérdida": float(perdida.detach()), L: np.nan, ADH: np.nan}
        if archivo in ARCHIVOS_VAL and (paso == 1 or paso % 100 == 0):
            with torch.no_grad(): fondo_paso = modelo(coords).reshape(h, w).cpu().numpy()
            metrica = metricas_archivo(u - fondo_paso, archivo)
            fila.update({L: metrica[L], ADH: metrica[ADH]})
        curva.append(fila)
    sincronizar_A(); tiempo_ajuste = time.perf_counter() - inicio
    sincronizar_A(); inicio = time.perf_counter()
    with torch.no_grad(): fondo = modelo(coords).reshape(h, w)
    sincronizar_A(); tiempo_inferencia = time.perf_counter() - inicio
    fondo = fondo.cpu().numpy()
    del modelo, optim, coords, objetivo
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return {"fondo": fondo, "curva": pd.DataFrame(curva), "tiempo_ajuste": tiempo_ajuste, "tiempo_inferencia": tiempo_inferencia}

AJUSTES_VAL = {(s, f): en_cache(f"E4_A_v1_{s}_{f[:-4]}", lambda s=s, f=f: ajustar_imagen(f, s))
               for s in CONFIG_A["semillas"] for f in ARCHIVOS_VAL}
filas = []
for (s, f), resultado in AJUSTES_VAL.items():
    filas.append({"semilla": s, "archivo": f, **metricas_archivo(PREPROCESADAS[f] - resultado["fondo"], f)})
EVAL_A_VAL = pd.DataFrame(filas)
SEMILLAS_A = EVAL_A_VAL.groupby("semilla")[[L, ADH]].median()
guardar_tabla(EVAL_A_VAL, "E4_A_validacion_por_imagen")
guardar_tabla(SEMILLAS_A, "E4_A_validacion_semillas", index=True)
display(SEMILLAS_A.round(3))
aptas = SEMILLAS_A[SEMILLAS_A[ADH] <= 5]
SEMILLA_ELEGIDA = int(aptas[L].idxmin() if len(aptas) else SEMILLAS_A[ADH].idxmin())
print(f"Lectura de la salida (8.2). Media ± desviación muestral entre semillas: M1 = {SEMILLAS_A[L].mean():.2f} ± {SEMILLAS_A[L].std():.2f} niveles; |Δh| = {SEMILLAS_A[ADH].mean():.2f} ± {SEMILLAS_A[ADH].std():.2f} %. Se elige la semilla {SEMILLA_ELEGIDA} en validación. "
      f"La cota de altura {'se cumple' if (SEMILLAS_A[ADH] <= 5).all() else 'no se cumple'} en todas las semillas.")'''),
md(r'''### 8.3 Curvas de ajuste y preservación

**H8.2 (antes de comparar).** La pérdida descenderá durante el ajuste, pero una pérdida menor no garantiza conservar las células: si el fondo estimado absorbe su señal, $|\Delta h|$ puede aumentar. Como cada imagen se ajusta independientemente, no existe una pérdida de validación separada para esos mismos pesos. Las métricas en los puntos de control se calculan después de cada actualización y la pérdida mostrada corresponde al paso anterior a ella.'''),
code(r'''# 8.3 Evolución para la semilla elegida
CURVAS_A = pd.concat([r["curva"].assign(archivo=f, semilla=s) for (s, f), r in AJUSTES_VAL.items()], ignore_index=True)
guardar_tabla(CURVAS_A, "E4_curvas_A")
curva_med = CURVAS_A[CURVAS_A.semilla == SEMILLA_ELEGIDA].groupby("iteración")[["pérdida", L, ADH]].median()
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, titulo, unidad in zip(axes, ["pérdida", L, ADH], ["Pérdida de ajuste", "Fondo lento residual", "Cambio de altura"], ["Pérdida física", "M1 (niveles eq.)", "|Δh| (%)"]):
    serie = curva_med[col].dropna()
    ax.plot(serie.index, serie, marker="o" if col != "pérdida" else None, ms=3)
    ax.set(title=titulo, xlabel="Iteración", ylabel=unidad)
axes[2].axhline(5, color="gray", ls="--", label="Umbral 5 %"); axes[2].legend()
fig.suptitle(f"Mediana de las cinco imágenes de validación, semilla {SEMILLA_ELEGIDA}")
fig.tight_layout(); plt.show(); guardar_figura(fig, "E4", "curvas_A")
display(curva_med.round(3))
print(f"Lectura de la salida (8.3). La pérdida mediana pasa de {curva_med['pérdida'].iloc[0]:.2f} a {curva_med['pérdida'].iloc[-1]:.2f}; "
      f"|Δh| pasa de {curva_med[ADH].iloc[0]:.2f} % a {curva_med[ADH].iloc[-1]:.2f} %. "
      "La evolución de altura permite vigilar posible absorción celular; no hay un conjunto de validación independiente durante cada ajuste.")'''),
md(r'''### 8.4 Selección, equipo y tiempos

La semilla se fijó con la regla de 8.2, antes de examinar prueba. El equipo y sus versiones se registran en 2.3. El ajuste por imagen es parte del costo operativo; el tiempo de una pasada final no sustituye ese costo. Las mediciones del ajuste usadas para las curvas incluyen puntos de control de métricas y no son una comparación justa con otras arquitecturas.'''),
]
