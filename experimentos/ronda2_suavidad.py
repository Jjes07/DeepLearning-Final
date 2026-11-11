"""Prueba rápida de D con suavidad en el fondo."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from correr import entrenar_D, guardar_curva, guardar_resultado, inferir_D
from data import cargar


def _evaluar_D(modelo, imagenes_val, d):
    metricas = []
    tiempos_infer = []
    for archivo in imagenes_val:
        u = d["ctx"]["preprocesadas"][archivo]
        t0 = __import__("time").perf_counter()
        b = inferir_D(modelo, u)
        tiempos_infer.append(__import__("time").perf_counter() - t0)
        metricas.append(d["metricas_archivo"](u - b, archivo))
    return metricas, tiempos_infer


def main():
    d = cargar()
    ctx = d["ctx"]
    train = d["entrenamiento"]
    val = d["validacion"]
    ctx["fijar_semillas"](42)

    configs = [
        {"descripcion": "D_fmax5_lc0.001_suav0.01", "f_max": 5, "grado_poli": 2,
         "base_channels": 32, "lr": 1e-3, "epocas": 100, "batch_size": 8,
         "scheduler_step": 30, "pesos": {"lambda_c": 0.001, "lambda_tv": 0.0, "ref": 0.1, "suav": 0.01}},
        {"descripcion": "D_fmax5_lc0.01_suav0.01", "f_max": 5, "grado_poli": 2,
         "base_channels": 32, "lr": 1e-3, "epocas": 100, "batch_size": 8,
         "scheduler_step": 30, "pesos": {"lambda_c": 0.01, "lambda_tv": 0.0, "ref": 0.1, "suav": 0.01}},
        {"descripcion": "D_fmax5_lc0.001_suav0.001", "f_max": 5, "grado_poli": 2,
         "base_channels": 32, "lr": 1e-3, "epocas": 100, "batch_size": 8,
         "scheduler_step": 30, "pesos": {"lambda_c": 0.001, "lambda_tv": 0.0, "ref": 0.1, "suav": 0.001}},
        {"descripcion": "D_fmax5_lc0.01_suav0.001", "f_max": 5, "grado_poli": 2,
         "base_channels": 32, "lr": 1e-3, "epocas": 100, "batch_size": 8,
         "scheduler_step": 30, "pesos": {"lambda_c": 0.01, "lambda_tv": 0.0, "ref": 0.1, "suav": 0.001}},
    ]
    for cfg in configs:
        cfg["semilla"] = 42
        print(f"\nEntrenando D: {cfg['descripcion']}")
        modelo, info = entrenar_D(train, val, cfg, ctx)
        metricas, tiempos_infer = _evaluar_D(modelo, val, d)
        id_fila = guardar_resultado(None, cfg, metricas,
                                    {"tiempo_entrenamiento": info["tiempo_entrenamiento"],
                                     "tiempo_inferencia": sum(tiempos_infer)}, "D")
        guardar_curva(info["curva"], id_fila, "D")
        m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
        dh = [m["|Δh| mediana (%)"] for m in metricas]
        print(f"  M1={np.median(m1):.2f}, |Dh|={np.median(dh):.2f}, t_train={info['tiempo_entrenamiento']:.1f}s")


if __name__ == "__main__":
    main()
