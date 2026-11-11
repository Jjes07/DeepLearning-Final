"""Evalúa las configuraciones finalistas de la ronda 2 con 3 semillas y guarda pesos."""

import sys
import time
import uuid
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from correr import (
    ajustar_fino_D,
    entrenar_C,
    entrenar_D,
    guardar_curva,
    guardar_resultado,
    inferir_C,
    inferir_D,
)
from data import cargar


def _guardar_pesos(obj, config_id, semilla, esquema):
    """Guarda state_dict(s) en experimentos/pesos/."""
    pesos_dir = Path(__file__).resolve().parent / "pesos"
    pesos_dir.mkdir(exist_ok=True)
    nombre = f"{esquema}_{config_id}_semilla{semilla}.pth"
    ruta = pesos_dir / nombre
    if esquema == "C":
        encoder, base = obj
        torch.save({"encoder": encoder.state_dict(), "base": base.state_dict()}, ruta)
    else:
        torch.save(obj.state_dict(), ruta)
    return ruta


def _evaluar_C(encoder, base, val, d):
    metricas = []
    tiempos = []
    for archivo in val:
        u = d["ctx"]["preprocesadas"][archivo]
        t0 = time.perf_counter()
        b = inferir_C(encoder, base, u)
        tiempos.append(time.perf_counter() - t0)
        metricas.append(d["metricas_archivo"](u - b, archivo))
    return metricas, tiempos


def _evaluar_D(modelo, val, d):
    metricas = []
    tiempos = []
    for archivo in val:
        u = d["ctx"]["preprocesadas"][archivo]
        t0 = time.perf_counter()
        b = inferir_D(modelo, u)
        tiempos.append(time.perf_counter() - t0)
        metricas.append(d["metricas_archivo"](u - b, archivo))
    return metricas, tiempos


def _evaluar_D_plus(modelo, val, d, cfg):
    metricas = []
    tiempos = []
    for archivo in val:
        u = d["ctx"]["preprocesadas"][archivo]
        t0 = time.perf_counter()
        b, info = ajustar_fino_D(modelo, u, cfg, cfg["pesos"])
        tiempos.append(time.perf_counter() - t0 + info["tiempo_finetune"])
        metricas.append(d["metricas_archivo"](u - b, archivo))
    return metricas, tiempos


def _correr_config(cfg, d, hacer_D_plus=False):
    ctx = d["ctx"]
    val = d["validacion"]
    semillas = [42, 2024, 12345]
    m1_sem, dh_sem = [], []
    print(f"\nFinalista: {cfg['descripcion']} ({cfg.get('esquema','?')})")
    for semilla in semillas:
        ctx["fijar_semillas"](semilla)
        cfg_s = {**cfg, "semilla": semilla}
        esquema = cfg_s.get("esquema", "C")
        if esquema == "C":
            encoder, base, info = entrenar_C(d["entrenamiento"], val, cfg_s, ctx)
            metricas, tiempos_infer = _evaluar_C(encoder, base, val, d)
            _guardar_pesos((encoder, base), cfg_s["descripcion"], semilla, "C")
        else:
            modelo, info = entrenar_D(d["entrenamiento"], val, cfg_s, ctx)
            metricas, tiempos_infer = _evaluar_D(modelo, val, d)
            _guardar_pesos(modelo, cfg_s["descripcion"], semilla, "D")

        id_fila = guardar_resultado(
            None, cfg_s, metricas,
            {"tiempo_entrenamiento": info["tiempo_entrenamiento"],
             "tiempo_inferencia": sum(tiempos_infer)},
            esquema
        )
        guardar_curva(info["curva"], id_fila, esquema)
        m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
        dh = [m["|Δh| mediana (%)"] for m in metricas]
        print(f"  semilla {semilla}: M1={np.median(m1):.2f}, |Dh|={np.median(dh):.2f}")
        m1_sem.append(np.median(m1))
        dh_sem.append(np.median(dh))

        if hacer_D_plus and esquema == "D":
            cfg_plus = {
                **cfg_s,
                "descripcion": f"{cfg_s['descripcion']}_plus",
                "lr_finetune": 1e-4,
                "iteraciones_finetune": 100,
            }
            metricas_plus, tiempos_plus = _evaluar_D_plus(modelo, val, d, cfg_plus)
            guardar_resultado(
                None, cfg_plus, metricas_plus,
                {"tiempo_entrenamiento": info["tiempo_entrenamiento"],
                 "tiempo_inferencia": sum(tiempos_plus)},
                "D+"
            )
            m1p = [m["RMS lento (niveles eq.)"] for m in metricas_plus]
            dhp = [m["|Δh| mediana (%)"] for m in metricas_plus]
            print(f"    D+ semilla {semilla}: M1={np.median(m1p):.2f}, |Dh|={np.median(dhp):.2f}")

    print(f"  -> media M1={np.mean(m1_sem):.2f}±{np.std(m1_sem):.2f}, |Dh|={np.mean(dh_sem):.2f}±{np.std(dh_sem):.2f}")
    return np.mean(m1_sem), np.std(m1_sem), np.mean(dh_sem), np.std(dh_sem)


def main():
    d = cargar()

    # Ejemplo de configuraciones finalistas; se editan tras el barrido.
    finalistas = [
        {
            "descripcion": "C_fmax5_rec07_signo08_ref01",
            "esquema": "C",
            "f_max": 5,
            "grado_poli": 2,
            "base_channels": 32,
            "lr": 1e-3,
            "epocas": 200,
            "batch_size": 16,
            "scheduler_step": 50,
            "pesos": {
                "esp": 1.0, "signo": 0.8, "signo_neg": 1.0, "signo_pos": 0.01,
                "suav": 0.0, "ref": 0.1, "recorte": True, "rho_recorte": 0.7,
            },
        },
        {
            "descripcion": "D_fmax5_lc0.05_ltv0",
            "esquema": "D",
            "f_max": 5,
            "grado_poli": 2,
            "base_channels": 32,
            "lr": 1e-3,
            "epocas": 300,
            "batch_size": 16,
            "scheduler_step": 75,
            "pesos": {"lambda_c": 0.05, "lambda_tv": 0.0, "ref": 0.1, "suav": 0.0},
        },
    ]

    for cfg in finalistas:
        _correr_config(cfg, d, hacer_D_plus=False)


if __name__ == "__main__":
    main()
