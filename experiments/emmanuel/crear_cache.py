"""Genera experimentos/cache_dev.npz y .pkl a partir del notebook hasta la sección 6."""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cargar_contexto import cargar_contexto


def main():
    root = Path(__file__).resolve().parent.parent
    cache_dir = root / "experimentos"
    cache_dir.mkdir(exist_ok=True)
    npz_path = cache_dir / "cache_dev.npz"
    pkl_path = cache_dir / "cache_dev.pkl"

    print("Cargando contexto del notebook...")
    ctx = cargar_contexto(hasta_seccion=6)

    seleccion = ctx["SELECCION"]
    archivos = seleccion.archivo.tolist()
    n = len(archivos)
    h, w = ctx["PREPROCESADAS"][archivos[0]].shape

    pre = np.empty((n, h, w), dtype=np.float32)
    cel = np.empty((n, h, w), dtype=bool)
    fon = np.empty((n, h, w), dtype=bool)
    sigma = np.empty(n, dtype=np.float32)
    conjuntos = np.array([seleccion[seleccion.archivo == a].conjunto.iloc[0] for a in archivos])

    for i, a in enumerate(archivos):
        pre[i] = ctx["PREPROCESADAS"][a]
        cel[i], fon[i] = ctx["MASCARAS"][a]
        sigma[i] = ctx["SIGMA_RUIDO"][a]

    np.savez_compressed(
        npz_path,
        preprocesadas=pre,
        mascaras_cel=cel,
        mascaras_fon=fon,
        archivos=np.array(archivos),
        conjuntos=conjuntos,
        sigma_ruido=sigma,
    )

    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "CELULAS_EVAL": ctx["CELULAS_EVAL"],
                "SELECCION": seleccion,
                "SEED": ctx["SEED"],
            },
            f,
        )

    print(f"Cache guardado: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Metadata guardada: {pkl_path}")


if __name__ == "__main__":
    main()
