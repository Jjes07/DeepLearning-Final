"""Carga el cache de desarrollo y expone conjuntos de entrenamiento/validación."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metricas import metricas_archivo, metricas_imagen


def fijar_semillas(semilla: int = 42):
    """Fija semillas de Python, NumPy y PyTorch."""
    import os
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cargar():
    """Carga cache_dev.npz/pkl; si no existen los genera con crear_cache.py."""
    root = Path(__file__).resolve().parent.parent
    exp = root / "experimentos"
    npz_path = exp / "cache_dev.npz"
    pkl_path = exp / "cache_dev.pkl"

    if not npz_path.exists() or not pkl_path.exists():
        sys.path.insert(0, str(exp))
        import crear_cache

        crear_cache.main()

    npz = np.load(npz_path)
    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)

    archivos = npz["archivos"].tolist()
    indices = {a: i for i, a in enumerate(archivos)}
    pre = npz["preprocesadas"]
    cel = npz["mascaras_cel"]
    fon = npz["mascaras_fon"]
    sigma = npz["sigma_ruido"]

    preprocesadas = {a: pre[i] for i, a in enumerate(archivos)}
    mascaras = {a: (cel[i], fon[i]) for i, a in enumerate(archivos)}
    sigma_ruido = {a: float(sigma[i]) for i, a in enumerate(archivos)}

    seleccion = meta["SELECCION"]
    celulas_eval = meta["CELULAS_EVAL"]

    def _archivos(conjunto: str) -> list:
        return seleccion[seleccion.conjunto == conjunto].archivo.tolist()

    ctx = {
        "preprocesadas": preprocesadas,
        "mascaras": mascaras,
        "mascaras_cel": cel,
        "mascaras_fon": fon,
        "sigma_ruido": sigma_ruido,
        "celulas_eval": celulas_eval,
        "seleccion": seleccion,
        "archivos": archivos,
        "indices": indices,
        "SEED": meta["SEED"],
        "fijar_semillas": fijar_semillas,
    }

    return {
        "ctx": ctx,
        "entrenamiento": _archivos("entrenamiento"),
        "validacion": _archivos("validacion"),
        "prueba": _archivos("prueba"),
        "generalizacion": _archivos("generalizacion"),
        "metricas_archivo": lambda v, a: metricas_archivo(v, a, ctx),
        "metricas_imagen": metricas_imagen,
    }


def normalizar_por_imagen(u: np.ndarray) -> tuple:
    """Normaliza una imagen a media 0 y desviación 1 (usada por el modelo C)."""
    media = u.mean()
    std = u.std() + 1e-8
    return (u - media) / std, media, std
