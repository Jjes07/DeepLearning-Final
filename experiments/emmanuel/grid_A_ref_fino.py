"""Refinamiento alrededor de A con referencia prometedora."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from correr import entrenar_A, guardar_resultado
from data import cargar
from modelos import SirenBackgroundNetwork


def main():
    d = cargar()
    d["ctx"]["fijar_semillas"](42)
    val = d["validacion"]

    configs = [
        {"signo": 1.0, "ref": 0.05, "rho": 0.7, "iter": 1000},
        {"signo": 1.0, "ref": 0.2, "rho": 0.7, "iter": 1000},
        {"signo": 1.0, "ref": 0.1, "rho": 0.75, "iter": 1000},
        {"signo": 1.0, "ref": 0.1, "rho": 0.7, "iter": 1500},
        {"signo": 1.2, "ref": 0.1, "rho": 0.7, "iter": 1000},
        {"signo": 0.8, "ref": 0.1, "rho": 0.7, "iter": 1000},
    ]
    for cfg in configs:
        config = {
            "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
            "lr": 1e-3,
            "iteraciones": cfg["iter"],
            "semilla": 42,
            "pesos": {"esp": 1.0, "signo": cfg["signo"], "signo_neg": 1.0, "signo_pos": 0.01,
                      "suav": 0.0, "ref": cfg["ref"],
                      "recorte": True, "rho_recorte": cfg["rho"]},
        }
        metricas = []
        tiempos = []
        for archivo in val:
            u = d["ctx"]["preprocesadas"][archivo]
            b, info = entrenar_A(u, SirenBackgroundNetwork(**config["modelo_kwargs"]), config)
            m = d["metricas_archivo"](u - b, archivo)
            metricas.append(m)
            tiempos.append(info["tiempo_inferencia"])

        m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
        dh = [m["|Δh| mediana (%)"] for m in metricas]
        nombre = f"A_SIREN_signo{cfg['signo']}_ref{cfg['ref']}_rec{cfg['rho']}_{cfg['iter']}"
        print(f"{nombre}: M1={np.median(m1):.2f}, |Dh|={np.median(dh):.2f}")
        guardar_resultado(None, {"descripcion": nombre, **config}, metricas,
                          {"tiempo_entrenamiento": sum(tiempos) * config["iteraciones"], "tiempo_inferencia": sum(tiempos)}, "A")


if __name__ == "__main__":
    main()
