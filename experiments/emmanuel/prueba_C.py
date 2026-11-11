"""Prueba rápida del esquema C en entrenamiento/validación."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import cargar
from correr import entrenar_C


def main():
    d = cargar()
    d["ctx"]["fijar_semillas"](42)

    config = {
        "f_max": 4,
        "grado_poli": 2,
        "base_channels": 32,
        "lr": 1e-3,
        "epocas": 50,
        "pesos": {"esp": 1.0, "signo": 0.5, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0},
        "semilla": 42,
    }
    encoder, base, info = entrenar_C(d["entrenamiento"], d["validacion"], config, d["ctx"])
    print("Curva final:")
    print(info["curva"].tail())
    print(f"Tiempo entrenamiento: {info['tiempo_entrenamiento']:.2f} s")


if __name__ == "__main__":
    main()
