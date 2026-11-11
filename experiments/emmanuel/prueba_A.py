"""Prueba rápida del esquema A en una imagen de entrenamiento."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import cargar
from correr import entrenar_A
from modelos import SirenBackgroundNetwork


def main():
    d = cargar()
    d["ctx"]["fijar_semillas"](42)
    d = cargar()
    ctx = d["ctx"]
    archivo = d["entrenamiento"][0]
    u = ctx["preprocesadas"][archivo]

    config = {
        "lr": 1e-3,
        "iteraciones": 500,
        "pesos": {"esp": 1.0, "signo": 0.1, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0},
        "semilla": 42,
    }
    modelo = SirenBackgroundNetwork(hidden_dim=128, n_layers=2, omega_0=1.0)
    b, info = entrenar_A(u, modelo, config)
    m = d["metricas_archivo"](u - b, archivo)
    print("Métricas en entrenamiento:")
    for k, v in m.items():
        clave = k.replace("Δ", "Delta").replace("|", "abs_")
        print(f"  {clave}: {v:.3f}" if isinstance(v, float) else f"  {clave}: {v}")
    print(f"Tiempo entrenamiento: {info['tiempo_entrenamiento']:.2f} s")


if __name__ == "__main__":
    main()
