"""Búsqueda de hiperparámetros para el esquema A en validación."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from data import cargar
from correr import entrenar_A, guardar_resultado
from modelos import ImplicitFourierMLP, SirenBackgroundNetwork


def evaluar_A(d, modelo_fn, modelo_kwargs, config_base, nombre):
    val = d["validacion"]
    metricas = []
    tiempos = []
    for archivo in val:
        u = d["ctx"]["preprocesadas"][archivo]
        modelo = modelo_fn(**modelo_kwargs)
        b, info = entrenar_A(u, modelo, config_base)
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
        {"descripcion": nombre, **config_base},
        metricas,
        {"tiempo_entrenamiento": sum(tiempos) * config_base.get("iteraciones", 1), "tiempo_inferencia": sum(tiempos)},
        "A",
    )
    return metricas


def main():
    d = cargar()
    d["ctx"]["fijar_semillas"](42)

    configs = [
        {
            "modelo_fn": SirenBackgroundNetwork,
            "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
            "config": {
                "lr": 1e-3,
                "iteraciones": 1000,
                "pesos": {"esp": 1.0, "signo": 5.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0},
                "semilla": 42,
            },
            "nombre": "A_SIREN_w1_signo5",
        },
        {
            "modelo_fn": SirenBackgroundNetwork,
            "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
            "config": {
                "lr": 1e-3,
                "iteraciones": 1000,
                "pesos": {"esp": 1.0, "signo": 10.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0},
                "semilla": 42,
            },
            "nombre": "A_SIREN_w1_signo10",
        },
        {
            "modelo_fn": SirenBackgroundNetwork,
            "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
            "config": {
                "lr": 1e-3,
                "iteraciones": 1000,
                "pesos": {"esp": 1.0, "signo": 5.0, "signo_neg": 1.0, "signo_pos": 0.1, "suav": 0.0, "ref": 0.0},
                "semilla": 42,
            },
            "nombre": "A_SIREN_w1_signo5_pos01",
        },
        {
            "modelo_fn": SirenBackgroundNetwork,
            "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 0.5},
            "config": {
                "lr": 1e-3,
                "iteraciones": 1000,
                "pesos": {"esp": 1.0, "signo": 5.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0},
                "semilla": 42,
            },
            "nombre": "A_SIREN_w05_signo5",
        },
        {
            "modelo_fn": SirenBackgroundNetwork,
            "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
            "config": {
                "lr": 1e-3,
                "iteraciones": 1000,
                "pesos": {"esp": 1.0, "signo": 5.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.01, "ref": 0.0},
                "semilla": 42,
            },
            "nombre": "A_SIREN_w1_signo5_suav001",
        },
    ]

    for cfg in configs:
        evaluar_A(d, cfg["modelo_fn"], cfg["modelo_kwargs"], cfg["config"], cfg["nombre"])


if __name__ == "__main__":
    main()
