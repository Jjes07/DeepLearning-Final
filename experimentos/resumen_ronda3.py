"""Resume finalistas y representa los mapas de pesos de validación."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from correr import inferir_E
from data import cargar
from modelos import ModeloE

EXP = Path(__file__).resolve().parent


def main():
    filas = pd.read_csv(EXP / "resultados_ronda3.csv")
    grupos = filas.groupby("id").filter(lambda g: len(g) == 3).groupby("id")
    resumen = grupos.agg(M1_media=("M1", "mean"), M1_std=("M1", "std"),
                         dh_media=("dh", "mean"), dh_std=("dh", "std"),
                         tiempo=("tiempo_entrenamiento", "mean"))
    elegibles = resumen[resumen.dh_media <= 5]
    if elegibles.empty:
        print(resumen.to_string())
        print("Ninguna finalista satisface |Δh| ≤ 5 %.")
        # Representación diagnóstica de la menor pérdida de validación.
        ganador = filas.loc[filas.id.isin(resumen.index)].sort_values("loss_val").iloc[0].id
    else:
        ganador = elegibles.M1_media.idxmin()
    print(resumen.to_string())
    print("Modelo representado:", ganador)
    datos = cargar()
    guardado = torch.load(EXP / "pesos" / f"{ganador}_s42.pth", map_location="cpu", weights_only=False)
    cfg = guardado["config"]
    nombres = datos["validacion"]
    h, w = datos["ctx"]["preprocesadas"][nombres[0]].shape
    modelo = ModeloE(h, w, f_max=cfg["f_max"], grado_poli=3,
                     base_channels=cfg["base_channels"])
    modelo.load_state_dict(guardado["estado"])
    modelo.eval()
    fig, ejes = plt.subplots(len(nombres), 4, figsize=(14, 3.0 * len(nombres)), constrained_layout=True)
    titulos = ["Imagen", "Peso de fondo w", "Fondo estimado", "Imagen corregida"]
    for j, titulo in enumerate(titulos):
        ejes[0, j].set_title(titulo)
    for i, nombre in enumerate(nombres):
        u = datos["ctx"]["preprocesadas"][nombre]
        b, mapa = inferir_E(modelo, u)
        for j, img in enumerate((u, mapa, b, u - b)):
            opciones = {"cmap": "viridis", "vmin": 0, "vmax": 1} if j == 1 else {"cmap": "gray"}
            grafico = ejes[i, j].imshow(img, **opciones)
            ejes[i, j].axis("off")
            if j == 1:
                fig.colorbar(grafico, ax=ejes[i, j], fraction=0.046, pad=0.02)
        ejes[i, 0].set_ylabel(nombre, rotation=0)
        ejes[i, 0].text(0.01, 0.02, nombre, transform=ejes[i, 0].transAxes,
                         fontsize=6, color="white", bbox={"facecolor": "black", "alpha": 0.6})
    fig.suptitle(f"{ganador}, semilla 42: validación", fontsize=13)
    (EXP / "figuras").mkdir(exist_ok=True)
    fig.savefig(EXP / "figuras" / "E_pesos_validacion.png", dpi=140)
    plt.close(fig)
    curva = pd.read_csv(EXP / "curvas" / f"{ganador}_s42.csv")
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    ejes[0].plot(curva.epoca, curva.loss_train, label="Entrenamiento")
    ejes[0].plot(curva.epoca, curva.loss_val, label="Validación")
    ejes[0].set_ylabel("Pérdida")
    ejes[0].legend()
    ejes[1].plot(curva.epoca, curva.M1_val, marker=".", linestyle="none")
    ejes[1].set_ylabel("M1")
    ejes[2].plot(curva.epoca, curva.dh_val, marker=".", linestyle="none")
    ejes[2].set_ylabel("|Δh| (%)")
    for eje in ejes:
        eje.set_xlabel("Época")
        eje.grid(alpha=0.2)
    fig.suptitle(f"{ganador}, semilla 42")
    fig.savefig(EXP / "figuras" / "E_curvas_finalista.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
