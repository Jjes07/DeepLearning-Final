"""Celdas para incorporar la comparación temporal al notebook."""
from nbformat.v4 import new_markdown_cell as md, new_code_cell as code

CELDAS_TIEMPOS = [
md(r'''**Medición de A y AE en el mismo equipo.** La comparación se ejecuta de forma independiente con `scripts/medir_tiempos.py`, utilizando las cinco imágenes de validación y la semilla 42 para ambos métodos. Se alterna el orden, se calienta la GPU y se sincroniza antes y después de cada intervalo. En AE se mide por separado la pasada de la U-Net que produce los pesos, además de las 1.000 iteraciones de la SIREN y su salida final. El entrenamiento previo de la U-Net no forma parte del costo por imagen. Las lecturas de actividad de GPU antes y después son controles puntuales y no demuestran ausencia continua de otras cargas. Los tiempos no incluyen la carga inicial del archivo de pesos ni el cálculo de métricas; los resultados dependen del equipo y de su carga.'''),
code(r'''# 8.4 Resultados medidos con el mismo protocolo
ruta_tiempos = RUNS_DIR / "comparacion_tiempos"
with (ruta_tiempos / "config.json").open(encoding="utf-8") as archivo:
    configuracion_tiempos = json.load(archivo)
TIEMPOS_POR_IMAGEN = pd.read_csv(ruta_tiempos / "por_imagen.csv")
TIEMPOS_CONTROLADOS = pd.read_csv(ruta_tiempos / "resumen.csv").set_index("método")
assert configuracion_tiempos["semilla"] == 42
assert TIEMPOS_POR_IMAGEN.groupby("método").size().eq(5).all()
guardar_tabla(TIEMPOS_POR_IMAGEN, "E4_tiempos_controlados_por_imagen")
guardar_tabla(TIEMPOS_CONTROLADOS, "E4_tiempos_controlados", index=True)
display(TIEMPOS_CONTROLADOS.round(4))
print("Actividad puntual de GPU (%):", configuracion_tiempos["actividad_gpu_porcentaje"])
razon = TIEMPOS_CONTROLADOS.loc["AE", "total (s/imagen)"] / TIEMPOS_CONTROLADOS.loc["A", "total (s/imagen)"]
print(f"Lectura de la salida (8.4). En esta medición, la mediana del costo por imagen es "
      f"{TIEMPOS_CONTROLADOS.loc['A', 'total (s/imagen)']:.2f} s para A y "
      f"{TIEMPOS_CONTROLADOS.loc['AE', 'total (s/imagen)']:.2f} s para AE; "
      f"la razón AE/A es {razon:.2f}. La U-Net añade {configuracion_tiempos['parametros_unet']:,} parámetros "
      "y su entrenamiento previo no está incluido. Son tiempos de la semilla 42 y dependen de la carga del equipo.")''')
]
