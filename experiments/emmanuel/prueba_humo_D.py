"""Prueba de humo del esquema D: 2 épocas, imprime memoria máxima."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from correr import entrenar_D
from data import cargar


def main():
    d = cargar()
    ctx = d["ctx"]
    ctx["fijar_semillas"](42)

    cfg = {
        "descripcion": "D_humo_fmax5_lc0.05",
        "f_max": 5,
        "grado_poli": 2,
        "base_channels": 32,
        "lr": 1e-3,
        "epocas": 2,
        "batch_size": 8,
        "scheduler_step": 75,
        "pesos": {"lambda_c": 0.05, "lambda_tv": 0.0, "ref": 0.1, "suav": 0.0},
        "semilla": 42,
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    modelo, info = entrenar_D(d["entrenamiento"], d["validacion"], cfg, ctx)

    if torch.cuda.is_available():
        memoria_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Memoria máxima allocated: {memoria_gb:.2f} GB")
        if memoria_gb > 7.0:
            print("ADVERTENCIA: supera 7 GB; reducir base_channels a 16.")
        else:
            print("OK: memoria bajo 7 GB.")
    else:
        print("No hay GPU disponible.")


if __name__ == "__main__":
    main()
