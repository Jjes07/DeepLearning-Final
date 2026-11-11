"""Pipeline completo: A, C y C+ con guardado de resultados y curvas."""

import sys
import time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from correr import ajustar_fino_C, entrenar_A, entrenar_C, guardar_resultado, inferir_C
from data import cargar
from modelos import SirenBackgroundNetwork


def run_A(d, config, nombre):
    """Ejecuta esquema A en validación y guarda resultado."""
    val = d["validacion"]
    metricas = []
    tiempos = []
    for archivo in val:
        u = d["ctx"]["preprocesadas"][archivo]
        modelo = SirenBackgroundNetwork(**config["modelo_kwargs"])
        b, info = entrenar_A(u, modelo, config)
        m = d["metricas_archivo"](u - b, archivo)
        metricas.append(m)
        tiempos.append(info["tiempo_inferencia"])

    m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
    dh = [m["|Δh| mediana (%)"] for m in metricas]
    print(f"A {nombre}: M1={np.median(m1):.2f}, |Δh|={np.median(dh):.2f}")

    return guardar_resultado(
        None,
        {"descripcion": nombre, **config},
        metricas,
        {"tiempo_entrenamiento": sum(tiempos) * config.get("iteraciones", 1), "tiempo_inferencia": sum(tiempos)},
        "A",
    )


def run_C(d, config, nombre):
    """Ejecuta esquema C en validación y guarda resultado y curvas."""
    encoder, base, info = entrenar_C(d["entrenamiento"], d["validacion"], config, d["ctx"])
    val = d["validacion"]
    metricas = []
    tiempos = []
    for archivo in val:
        u = d["ctx"]["preprocesadas"][archivo]
        t0 = time.perf_counter()
        b = inferir_C(encoder, base, u)
        tiempos.append(time.perf_counter() - t0)
        m = d["metricas_archivo"](u - b, archivo)
        metricas.append(m)

    m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
    dh = [m["|Δh| mediana (%)"] for m in metricas]
    print(f"C {nombre}: M1={np.median(m1):.2f}, |Δh|={np.median(dh):.2f}")

    exp_dir = Path(__file__).resolve().parent
    exp_dir.mkdir(exist_ok=True)
    curvas_dir = exp_dir / "curvas"
    curvas_dir.mkdir(exist_ok=True)

    row_id = guardar_resultado(
        None,
        {"descripcion": nombre, **config},
        metricas,
        {"tiempo_entrenamiento": info["tiempo_entrenamiento"], "tiempo_inferencia": sum(tiempos)},
        "C",
    )
    info["curva"].to_csv(curvas_dir / f"{row_id}.csv", index=False)
    return row_id


def run_Cplus(d, config_c, config_finetune, nombre):
    """Ejecuta esquema C+ en validación y guarda resultado."""
    encoder, base, info_c = entrenar_C(d["entrenamiento"], d["validacion"], config_c, d["ctx"])
    val = d["validacion"]
    metricas = []
    tiempos = []
    for archivo in val:
        u = d["ctx"]["preprocesadas"][archivo]
        b, info_f = ajustar_fino_C(encoder, base, u, config_finetune, d["ctx"], archivo)
        m = d["metricas_archivo"](u - b, archivo)
        metricas.append(m)
        tiempos.append(info_f["tiempo_finetune"])

    m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
    dh = [m["|Δh| mediana (%)"] for m in metricas]
    print(f"C+ {nombre}: M1={np.median(m1):.2f}, |Δh|={np.median(dh):.2f}")

    return guardar_resultado(
        None,
        {"descripcion": nombre, "config_c": config_c, "config_finetune": config_finetune},
        metricas,
        {"tiempo_entrenamiento": info_c["tiempo_entrenamiento"], "tiempo_inferencia": sum(tiempos)},
        "C+",
    )


def main():
    d = cargar()
    d["ctx"]["fijar_semillas"](42)

    # A: configuraciones seleccionadas
    cfg_A = {
        "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
        "lr": 1e-3,
        "iteraciones": 2000,
        "pesos": {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0, "recorte": True, "rho_recorte": 0.7},
        "semilla": 42,
    }
    run_A(d, cfg_A, "A_SIREN_recorte07_2k")

    cfg_A2 = {
        "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
        "lr": 1e-3,
        "iteraciones": 2000,
        "pesos": {"esp": 1.0, "signo": 0.5, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0, "recorte": True, "rho_recorte": 0.7},
        "semilla": 42,
    }
    run_A(d, cfg_A2, "A_SIREN_signo05_recorte07_2k")

    # C: configuración base
    cfg_C = {
        "f_max": 8,
        "grado_poli": 2,
        "base_channels": 64,
        "lr": 1e-3,
        "epocas": 200,
        "batch_size": 8,
        "scheduler_step": 60,
        "pesos": {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0, "recorte": True, "rho_recorte": 0.7},
        "semilla": 42,
    }
    run_C(d, cfg_C, "C_base")


if __name__ == "__main__":
    main()
