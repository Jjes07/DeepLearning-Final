"""Entrena la ganadora y guarda curvas de pérdida por iteración."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from correr import entrenar_A
from data import cargar
from modelos import SirenBackgroundNetwork


def main():
    d = cargar()
    d["ctx"]["fijar_semillas"](42)
    val = d["validacion"]

    config = {
        "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
        "lr": 1e-3,
        "iteraciones": 1000,
        "semilla": 42,
        "pesos": {"esp": 1.0, "signo": 0.8, "signo_neg": 1.0, "signo_pos": 0.01,
                  "suav": 0.0, "ref": 0.1, "recorte": True, "rho_recorte": 0.7},
    }

    exp = Path(__file__).resolve().parent
    curvas_dir = exp / "curvas"
    curvas_dir.mkdir(exist_ok=True)

    for archivo in val:
        u = d["ctx"]["preprocesadas"][archivo]
        b, info = entrenar_A(u, SirenBackgroundNetwork(**config["modelo_kwargs"]), config)
        curva = info["curva"]
        curva["archivo"] = archivo
        fname = curvas_dir / f"ganadora_{Path(archivo).stem}.csv"
        curva.to_csv(fname, index=False)
        m = d["metricas_archivo"](u - b, archivo)
        print(f"{archivo}: M1={m['RMS lento (niveles eq.)']:.2f}, |Dh|={m['|Δh| mediana (%)']:.2f}")


if __name__ == "__main__":
    main()
