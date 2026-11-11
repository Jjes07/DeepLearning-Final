"""Evalúa la configuración ganadora con 3 semillas distintas."""

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
    val = d["validacion"]

    # Configuración ganadora robusta: A_SIREN_signo0.8_ref0.1_rec0.7_1000
    config_base = {
        "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
        "lr": 1e-3,
        "iteraciones": 1000,
        "pesos": {"esp": 1.0, "signo": 0.8, "signo_neg": 1.0, "signo_pos": 0.01,
                  "suav": 0.0, "ref": 0.1, "recorte": True, "rho_recorte": 0.7},
    }

    semillas = [42, 2024, 12345]
    resultados_semilla = []
    for semilla in semillas:
        d["ctx"]["fijar_semillas"](semilla)
        config = {**config_base, "semilla": semilla}
        metricas = []
        tiempos_train = []
        tiempos_infer = []
        for archivo in val:
            u = d["ctx"]["preprocesadas"][archivo]
            b, info = entrenar_A(u, SirenBackgroundNetwork(**config["modelo_kwargs"]), config)
            m = d["metricas_archivo"](u - b, archivo)
            metricas.append(m)
            tiempos_train.append(info["tiempo_entrenamiento"])
            tiempos_infer.append(info["tiempo_inferencia"])

        m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
        dh = [m["|Δh| mediana (%)"] for m in metricas]
        nombre = f"A_ganadora_final_semilla{semilla}"
        print(f"{nombre}: M1={np.median(m1):.2f}, |Dh|={np.median(dh):.2f}")
        guardar_resultado(None, {"descripcion": nombre, **config}, metricas,
                          {"tiempo_entrenamiento": sum(tiempos_train), "tiempo_inferencia": sum(tiempos_infer)}, "A")
        resultados_semilla.append((semilla, np.median(m1), np.median(dh)))

    m1s = [r[1] for r in resultados_semilla]
    dhs = [r[2] for r in resultados_semilla]
    print(f"\nMedia ± std M1: {np.mean(m1s):.2f} ± {np.std(m1s):.2f}")
    print(f"Media ± std |Dh|: {np.mean(dhs):.2f} ± {np.std(dhs):.2f}")


if __name__ == "__main__":
    main()
