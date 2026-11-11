"""Segunda ronda experimental: esquemas C y D en entrenamiento/validación."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from correr import entrenar_C, entrenar_D, guardar_curva, guardar_resultado, inferir_C, inferir_D
from data import cargar


def _evaluar_C(encoder, base, imagenes_val, d):
    metricas = []
    tiempos_infer = []
    for archivo in imagenes_val:
        u = d["ctx"]["preprocesadas"][archivo]
        t0 = __import__("time").perf_counter()
        b = inferir_C(encoder, base, u)
        tiempos_infer.append(__import__("time").perf_counter() - t0)
        metricas.append(d["metricas_archivo"](u - b, archivo))
    return metricas, tiempos_infer


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
    semilla = 42
    ctx["fijar_semillas"](semilla)

    # Esquema C: barrido corto de f_max con la pérdida ganadora de la ronda 1.
    configs_C = []
    for f_max in [3, 5, 8]:
        cfg = {
            "descripcion": f"C_fmax{f_max}_rec07_signo08_ref01",
            "f_max": f_max,
            "grado_poli": 2,
            "base_channels": 32,
            "lr": 1e-3,
            "epocas": 200,
            "batch_size": 16,
            "scheduler_step": 50,
            "pesos": {
                "esp": 1.0, "signo": 0.8, "signo_neg": 1.0, "signo_pos": 0.01,
                "suav": 0.0, "ref": 0.1, "recorte": True, "rho_recorte": 0.7,
            },
            "semilla": semilla,
        }
        configs_C.append(cfg)

    # Esquema D: barrido reducido a 6 configs (λ_c × f_max × λ_tv).
    configs_D = []
    for f_max in [5, 8]:
        for lambda_c in [0.01, 0.05, 0.1]:
            configs_D.append({
                "descripcion": f"D_fmax{f_max}_lc{lambda_c}_ltv0",
                "f_max": f_max,
                "grado_poli": 2,
                "base_channels": 32,
                "lr": 1e-3,
                "epocas": 300,
                "batch_size": 8,
                "scheduler_step": 75,
                "pesos": {
                    "lambda_c": lambda_c, "lambda_tv": 0.0, "ref": 0.1, "suav": 0.0,
                },
                "semilla": semilla,
            })

    resultados = []
    for cfg in configs_C:
        print(f"\nEntrenando C: {cfg['descripcion']}")
        encoder, base, info = entrenar_C(train, val, cfg, ctx)
        metricas, tiempos_infer = _evaluar_C(encoder, base, val, d)
        id_fila = guardar_resultado(None, cfg, metricas,
                                    {"tiempo_entrenamiento": info["tiempo_entrenamiento"],
                                     "tiempo_inferencia": sum(tiempos_infer)}, "C")
        guardar_curva(info["curva"], id_fila, "C")
        m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
        dh = [m["|Δh| mediana (%)"] for m in metricas]
        print(f"  M1={np.median(m1):.2f}, |Dh|={np.median(dh):.2f}, t_train={info['tiempo_entrenamiento']:.1f}s")
        resultados.append(("C", cfg["descripcion"], np.median(m1), np.median(dh)))

    for cfg in configs_D:
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
        resultados.append(("D", cfg["descripcion"], np.median(m1), np.median(dh)))

    print("\nResumen ronda 2 (semilla 42):")
    for esq, desc, m1, dh in resultados:
        print(f"{esq} {desc}: M1={m1:.2f}, |Dh|={dh:.2f}")


if __name__ == "__main__":
    main()
