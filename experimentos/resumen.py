"""Muestra el resumen de resultados.csv ordenado por M1."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd


def main():
    csv = Path(__file__).resolve().parent / "resultados.csv"
    if not csv.exists():
        print("No hay resultados.csv aún.")
        return
    df = pd.read_csv(csv)
    cols = ["id", "descripcion", "esquema", "M1_val_mediana", "|Dh|_val_mediana", "tiempo_entrenamiento", "tiempo_inferencia"]
    print(df[cols].sort_values("M1_val_mediana").to_string(index=False))


if __name__ == "__main__":
    main()
