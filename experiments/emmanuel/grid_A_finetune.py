"""A + ajuste fino con signo fuerte para preservar células."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from correr import ajustar_fino_A, entrenar_A, guardar_resultado
from data import cargar
from modelos import SirenBackgroundNetwork


def main():
    d = cargar()
    d["ctx"]["fijar_semillas"](42)
    val = d["validacion"]

    configs = [
        {"signo_finetune": 2.0, "iter_finetune": 200},
        {"signo_finetune": 3.0, "iter_finetune": 200},
        {"signo_finetune": 2.0, "iter_finetune": 500},
        {"signo_finetune": 1.5, "iter_finetune": 500},
    ]
    for cfg in configs:
        config_base = {
            "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
            "lr": 1e-3,
            "iteraciones": 1000,
            "semilla": 42,
            "pesos": {"esp": 1.0, "signo": 0.5, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0,
                      "recorte": True, "rho_recorte": 0.7, "recorte_asim": True},
        }
        config_finetune = {
            "lr": 1e-4,
            "iteraciones": cfg["iter_finetune"],
            "semilla": 42,
            "pesos": {"esp": 1.0, "signo": cfg["signo_finetune"], "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0},
        }

        metricas = []
        tiempos = []
        for archivo in val:
            u = d["ctx"]["preprocesadas"][archivo]
            modelo = SirenBackgroundNetwork(**config_base["modelo_kwargs"])
            b0, info0 = entrenar_A(u, modelo, config_base)
            b = ajustar_fino_A(u, modelo, config_finetune)
            m = d["metricas_archivo"](u - b, archivo)
            metricas.append(m)
            tiempos.append(info0["tiempo_inferencia"] + cfg["iter_finetune"] * info0["tiempo_inferencia"] / config_base["iteraciones"])

        m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
        dh = [m["|Δh| mediana (%)"] for m in metricas]
        nombre = f"A_asim05_finetune_signo{cfg['signo_finetune']}_{cfg['iter_finetune']}"
        print(f"{nombre}: M1={np.median(m1):.2f}, |Δh|={np.median(dh):.2f}")
        guardar_resultado(None, {"descripcion": nombre, "config_base": config_base, "config_finetune": config_finetune},
                          metricas, {"tiempo_entrenamiento": sum(tiempos), "tiempo_inferencia": sum(tiempos)}, "A+")


if __name__ == "__main__":
    main()
