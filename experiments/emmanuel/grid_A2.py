"""Búsqueda rápida de hiperparámetros para el esquema A con recorte."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from data import cargar
from correr import entrenar_A, guardar_resultado
from modelos import SirenBackgroundNetwork


def evaluar_A(d, config, nombre):
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
    print(f"{nombre}: M1 mediana={pd.Series(m1).median():.2f}, |Δh| mediana={pd.Series(dh).median():.2f}")
    for a, mm in zip(val, metricas):
        print(f"  {a}: M1={mm['RMS lento (niveles eq.)']:.2f}, |Δh|={mm['|Δh| mediana (%)']:.2f}")

    guardar_resultado(
        None,
        {"descripcion": nombre, **config},
        metricas,
        {"tiempo_entrenamiento": sum(tiempos) * config.get("iteraciones", 1), "tiempo_inferencia": sum(tiempos)},
        "A",
    )
    return metricas


def main():
    d = cargar()
    d["ctx"]["fijar_semillas"](42)

    base = {
        "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
        "lr": 1e-3,
        "iteraciones": 500,
        "semilla": 42,
    }

    configs = [
        {**base, "pesos": {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0}, "nombre": "A_SIREN_signo1"},
        {**base, "pesos": {"esp": 1.0, "signo": 2.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0}, "nombre": "A_SIREN_signo2"},
        {**base, "pesos": {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0, "recorte": True, "rho_recorte": 0.5}, "nombre": "A_SIREN_signo1_recorte05"},
        {**base, "pesos": {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0, "recorte": True, "rho_recorte": 0.7}, "nombre": "A_SIREN_signo1_recorte07"},
        {**base, "pesos": {"esp": 1.0, "signo": 2.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0, "recorte": True, "rho_recorte": 0.7}, "nombre": "A_SIREN_signo2_recorte07"},
        {**base, "pesos": {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.1, "suav": 0.0, "ref": 0.0, "recorte": True, "rho_recorte": 0.7}, "nombre": "A_SIREN_signo1_pos01_recorte07"},
    ]

    for cfg in configs:
        nombre = cfg.pop("nombre")
        evaluar_A(d, cfg, nombre)


if __name__ == "__main__":
    main()
