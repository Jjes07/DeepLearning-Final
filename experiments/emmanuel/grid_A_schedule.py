"""A con schedule creciente de signo para balancear M1 y |Dh|."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import time

from correr import _device, guardar_resultado
from data import cargar, fijar_semillas
from modelos import SirenBackgroundNetwork, build_coordinate_grid
from perdidas import perdida_fisica


def entrenar_A_schedule(u, modelo, config):
    """Entrena A con signo que crece linealmente."""
    if config.get("semilla") is not None:
        fijar_semillas(config["semilla"])
    device = _device()
    H, W = u.shape
    coords = build_coordinate_grid(H, W, device=device)
    u_t = torch.tensor(u, dtype=torch.float32, device=device).flatten()
    modelo = modelo.to(device)
    optim = torch.optim.Adam(modelo.parameters(), lr=config["lr"])
    pesos = config["pesos"].copy()
    signo_inicial = config.get("signo_inicial", 0.0)
    signo_final = config.get("signo_final", 1.0)
    iteraciones = config["iteraciones"]
    curva = []
    t0 = time.perf_counter()
    for it in range(iteraciones):
        alpha = it / max(1, iteraciones - 1)
        pesos["signo"] = signo_inicial + alpha * (signo_final - signo_inicial)
        optim.zero_grad()
        b = modelo(coords)
        r = u_t - b
        loss, terminos = perdida_fisica(
            r, b, pesos, usar_signo=pesos.get("signo", 0) > 0,
            usar_suav=pesos.get("suav", 0) > 0, usar_ref=pesos.get("ref", 0) > 0,
            usar_recorte=pesos.get("recorte", False), rho_recorte=pesos.get("rho_recorte", 0.5),
            recorte_asim=pesos.get("recorte_asim", False)
        )
        loss.backward()
        optim.step()
        curva.append({"iter": it, "loss": loss.item(), "signo": pesos["signo"],
                      **{k: v.item() for k, v in terminos.items()}})
    t_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    with torch.no_grad():
        b = modelo(coords).reshape(H, W).cpu().numpy()
    t_infer = time.perf_counter() - t1
    return b, {"curva": curva, "tiempo_entrenamiento": t_train, "tiempo_inferencia": t_infer}


def main():
    d = cargar()
    d["ctx"]["fijar_semillas"](42)
    val = d["validacion"]

    configs = [
        {"signo_inicial": 0.0, "signo_final": 2.0, "iter": 1000, "rho": 0.7},
        {"signo_inicial": 0.0, "signo_final": 3.0, "iter": 1000, "rho": 0.7},
        {"signo_inicial": 0.2, "signo_final": 2.0, "iter": 1500, "rho": 0.7},
        {"signo_inicial": 0.0, "signo_final": 2.0, "iter": 1000, "rho": 0.75},
    ]
    for cfg in configs:
        config = {
            "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
            "lr": 1e-3,
            "iteraciones": cfg["iter"],
            "semilla": 42,
            "signo_inicial": cfg["signo_inicial"],
            "signo_final": cfg["signo_final"],
            "pesos": {"esp": 1.0, "signo": 0.0, "signo_neg": 1.0, "signo_pos": 0.01, "suav": 0.0, "ref": 0.0,
                      "recorte": True, "rho_recorte": cfg["rho"]},
        }
        metricas = []
        tiempos_train = []
        tiempos_infer = []
        for archivo in val:
            u = d["ctx"]["preprocesadas"][archivo]
            b, info = entrenar_A_schedule(u, SirenBackgroundNetwork(**config["modelo_kwargs"]), config)
            m = d["metricas_archivo"](u - b, archivo)
            metricas.append(m)
            tiempos_train.append(info["tiempo_entrenamiento"])
            tiempos_infer.append(info["tiempo_inferencia"])

        m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
        dh = [m["|Δh| mediana (%)"] for m in metricas]
        nombre = f"A_schedule{cfg['signo_inicial']}-{cfg['signo_final']}_rec{cfg['rho']}_{cfg['iter']}"
        print(f"{nombre}: M1={np.median(m1):.2f}, |Dh|={np.median(dh):.2f}")
        guardar_resultado(None, {"descripcion": nombre, **config}, metricas,
                          {"tiempo_entrenamiento": sum(tiempos_train), "tiempo_inferencia": sum(tiempos_infer)}, "A")


if __name__ == "__main__":
    main()
