"""Prueba rápida del esquema C con batches y pocos parámetros."""

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

    configs = [
        {"f_max": 4, "grado_poli": 2, "base_channels": 32, "lr": 1e-3, "epocas": 50, "batch_size": 8, "scheduler_step": 30,
         "pesos": {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0, "recorte": True, "rho_recorte": 0.7}, "semilla": 42},
        {"f_max": 8, "grado_poli": 2, "base_channels": 64, "lr": 1e-3, "epocas": 50, "batch_size": 8, "scheduler_step": 30,
         "pesos": {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0, "recorte": True, "rho_recorte": 0.7}, "semilla": 42},
        {"f_max": 8, "grado_poli": 3, "base_channels": 64, "lr": 1e-3, "epocas": 50, "batch_size": 8, "scheduler_step": 30,
         "pesos": {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0, "recorte": True, "rho_recorte": 0.7}, "semilla": 42},
    ]
    for i, cfg in enumerate(configs):
        print(f"\nConfig C{i+1}: f_max={cfg['f_max']}, canales={cfg['base_channels']}, grado={cfg['grado_poli']}")
        encoder, base, info = entrenar_C(d["entrenamiento"], d["validacion"], cfg, d["ctx"])
        print(info["curva"].tail())
        print(f"Tiempo: {info['tiempo_entrenamiento']:.2f} s")


if __name__ == "__main__":
    main()
