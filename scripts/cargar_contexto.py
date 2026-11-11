"""Carga el contexto de hasta la sección 6 del notebook EritroFase.ipynb."""

import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Windows: forzar UTF-8 en stdout/stderr para caracteres del notebook.
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")  # evita ventanas gráficas al ejecutar como script


def _display_noop(obj, **kwargs):
    """Reemplazo ligero de IPython.display.display fuera de un notebook."""
    pass


try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = _display_noop


def cargar_contexto(hasta_seccion: int = 6) -> dict:
    """Ejecuta las celdas de código del notebook hasta el final de la sección indicada.

    Por defecto se detiene al terminar la sección 6 (celda cuyo primer comentario
    empieza por ``# 6.6``). Fuerza ``RECALCULAR = False`` para reutilizar la caché.

    Returns
    -------
    dict
        Espacio de nombres resultante de ejecutar las celdas.
    """
    root = Path(__file__).resolve().parent.parent
    nb_path = root / "EritroFase.ipynb"
    if not nb_path.exists():
        raise FileNotFoundError(f"No se encontró el notebook: {nb_path}")

    import os

    cwd_previo = os.getcwd()
    os.chdir(root)  # las celdas usan Path.cwd() como raíz del proyecto

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Patrón que marca el final: primer comentario de la última subsección
    # de la sección indicada (p. ej. # 6.6 para la sección 6).
    patron_fin = re.compile(rf"^#\s*{hasta_seccion}\.{hasta_seccion}\b")

    celdas_a_ejecutar = []
    for celda in nb["cells"]:
        if celda["cell_type"] != "code":
            continue
        src = "".join(celda["source"])
        if not src.strip():
            continue
        celdas_a_ejecutar.append(src)
        primera = src.strip().split("\n", 1)[0]
        if patron_fin.match(primera):
            break

    if not celdas_a_ejecutar:
        raise ValueError("No se encontraron celdas de código para ejecutar.")

    # Forzar RECALCULAR = False en la celda de configuración (2.1).
    codigo_2_1 = celdas_a_ejecutar[0]
    if "RECALCULAR =" in codigo_2_1 and "# 2.1" in codigo_2_1:
        celdas_a_ejecutar[0] = re.sub(
            r"RECALCULAR\s*=\s*(True|False)", "RECALCULAR = False", codigo_2_1
        )

    ns = {"display": display, "__name__": "__main__"}

    t0 = time.perf_counter()
    for i, src in enumerate(celdas_a_ejecutar, start=1):
        exec(compile(src, f"<celda-{i}>", "exec"), ns)
    t1 = time.perf_counter()

    # Variables mínimas que debe dejar disponibles el contexto.
    requeridas = [
        "PREPROCESADAS", "MASCARAS", "SELECCION", "IMAGENES",
        "metricas_archivo", "metricas_imagen", "CELULAS_EVAL",
        "SIGMA_RUIDO", "LINEAS_BASE", "RESUMEN_LB_VAL",
        "DEVICE", "fijar_semillas", "SEED",
    ]
    faltantes = [v for v in requeridas if v not in ns]
    if faltantes:
        raise RuntimeError(
            f"Variables faltantes tras ejecutar el contexto: {', '.join(faltantes)}"
        )

    ns["_tiempo_carga_contexto"] = t1 - t0
    os.chdir(cwd_previo)
    return ns


def _tamanio(obj):
    """Devuelve una descripción aproximada del tamaño de un objeto."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return f"{obj.shape} {obj.dtype}"
    except Exception:
        pass
    try:
        n = len(obj)
        tipo = type(obj).__name__
        return f"{tipo} con {n} elementos"
    except Exception:
        return type(obj).__name__


if __name__ == "__main__":
    t0 = time.perf_counter()
    ctx = cargar_contexto(hasta_seccion=6)
    t1 = time.perf_counter()

    print(f"Contexto cargado en {t1 - t0:.2f} s")
    print(f"Tiempo interno de ejecución de celdas: {ctx['_tiempo_carga_contexto']:.2f} s")
    for nombre in [
        "PREPROCESADAS", "MASCARAS", "SELECCION", "IMAGENES",
        "metricas_archivo", "metricas_imagen", "CELULAS_EVAL",
        "SIGMA_RUIDO", "LINEAS_BASE", "RESUMEN_LB_VAL",
        "DEVICE", "fijar_semillas", "SEED",
    ]:
        print(f"  {nombre}: {_tamanio(ctx[nombre])}")
