"""Barrido y evaluación de la tercera ronda; selección por pérdida sin etiquetas."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from correr import entrenar_E, inferir_E
from data import cargar

RAIZ = Path(__file__).resolve().parent
SEMILLAS = (42, 2024, 12345)


def configuraciones():
    return [dict(id=f"E_f{f}_a{a}_c16", f_max=f, lambda_area=a,
                 base_channels=16, epocas=300, batch_size=4, lr=0.001)
            for f in (2, 4, 6) for a in (0.5, 2.0)]


def ejecutar(config, semilla, datos):
    ctx, val = datos["ctx"], datos["validacion"]
    cfg = {**config, "semilla": semilla}
    modelo, info = entrenar_E(datos["entrenamiento"], val, cfg, ctx)
    metricas, tiempos = [], []
    for nombre in val:
        u = ctx["preprocesadas"][nombre]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b, _ = inferir_E(modelo, u)
        torch.cuda.synchronize()
        tiempos.append(time.perf_counter() - t0)
        metricas.append(datos["metricas_archivo"](u - b, nombre))
    ruta = RAIZ / "pesos"
    ruta.mkdir(exist_ok=True)
    estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
    torch.save({"config": cfg, "estado": estado}, ruta / f"{cfg['id']}_s{semilla}.pth")
    (RAIZ / "curvas").mkdir(exist_ok=True)
    info["curva"].to_csv(RAIZ / "curvas" / f"{cfg['id']}_s{semilla}.csv", index=False)
    fila = {**cfg, "loss_val": info["loss_val"], "mejor_epoca": info["mejor_epoca"],
            "M1": float(np.median([m["RMS lento (niveles eq.)"] for m in metricas])),
            "dh": float(np.median([m["|Δh| mediana (%)"] for m in metricas])),
            "tiempo_entrenamiento": info["tiempo_entrenamiento"],
            "tiempo_inferencia": float(np.mean(tiempos)),
            "parametros": sum(p.numel() for p in modelo.parameters() if p.requires_grad)}
    for i, (nombre, m) in enumerate(zip(val, metricas)):
        fila[f"imagen_{i}"] = nombre
        fila[f"M1_{i}"] = m["RMS lento (niveles eq.)"]
        fila[f"dh_{i}"] = m["|Δh| mediana (%)"]
    del modelo, estado
    gc.collect()
    torch.cuda.empty_cache()
    return fila


def principal():
    parser = argparse.ArgumentParser()
    parser.add_argument("fase", choices=["barrido", "finalistas"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere GPU CUDA para la ronda.")
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    datos = cargar()
    archivo = RAIZ / "resultados_ronda3.csv"
    filas = pd.read_csv(archivo).to_dict("records") if archivo.exists() else []
    ya = {(str(x["id"]), int(x["semilla"])) for x in filas}
    configs = configuraciones()
    if args.fase == "barrido":
        tareas = [(c, 42) for c in configs]
    else:
        base = [f for f in filas if int(f["semilla"]) == 42]
        if len(base) != len(configs):
            raise RuntimeError("Completa el barrido antes de seleccionar finalistas.")
        elegidos = sorted(base, key=lambda x: x["loss_val"])[:2]
        ids = {f["id"] for f in elegidos}
        print("Finalistas por pérdida sin etiquetas:", sorted(ids), flush=True)
        tareas = [(c, s) for c in configs if c["id"] in ids for s in SEMILLAS]
    for cfg, semilla in tareas:
        if (cfg["id"], semilla) in ya:
            continue
        print(f"Inicio {cfg['id']} semilla {semilla}", flush=True)
        fila = ejecutar(cfg, semilla, datos)
        filas.append(fila)
        pd.DataFrame(filas).to_csv(archivo, index=False)
        print(json.dumps({k: fila[k] for k in ("id", "semilla", "loss_val", "M1", "dh",
                                               "tiempo_entrenamiento", "mejor_epoca")},
                         ensure_ascii=False), flush=True)
        print(f"Pico asignado: {torch.cuda.max_memory_allocated()/2**30:.3f} GiB", flush=True)


if __name__ == "__main__":
    principal()
