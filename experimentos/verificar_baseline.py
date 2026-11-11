"""Verifica que las métricas de Zernike y SIREN guía en validación coincidan con el notebook."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import cargar


def main():
    d = cargar()
    val = d["validacion"]

    # Se requiere el contexto completo solo para los fondos baseline del notebook.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from cargar_contexto import cargar_contexto

    print("Cargando contexto del notebook para obtener líneas base...")
    t0 = time.perf_counter()
    ctx = cargar_contexto(hasta_seccion=6)
    print(f"Contexto cargado en {time.perf_counter() - t0:.1f} s")

    resultados = []
    for archivo in val:
        u = d["ctx"]["preprocesadas"][archivo]
        for metodo in ["Zernike (guía)", "SIREN ω0=1 (guía)"]:
            b = ctx["LINEAS_BASE"][archivo]["fondos"][metodo]
            t0 = time.perf_counter()
            m = d["metricas_archivo"](u - b, archivo)
            t_infer = time.perf_counter() - t0
            resultados.append({"archivo": archivo, "método": metodo, **m, "tiempo_inferencia": t_infer})

    df = pd.DataFrame(resultados)
    resumen = df.groupby("método")[["RMS lento (niveles eq.)", "|Δh| mediana (%)", "Δh mediana (%)"]].median()
    print("\nMediana en validación:")
    print(resumen.round(2))

    esperado = {
        "Zernike (guía)": (6.4, 2.55),
        "SIREN ω0=1 (guía)": (3.5, 12.8),
    }
    print("\nComparación con valores esperados del notebook:")
    for metodo, (m1, m2) in esperado.items():
        r = resumen.loc[metodo]
        ok1 = np.isclose(r["RMS lento (niveles eq.)"], m1, atol=0.3)
        ok2 = np.isclose(r["|Δh| mediana (%)"], m2, atol=0.5)
        print(f"  {metodo}: M1={r['RMS lento (niveles eq.)']:.2f} vs {m1} ({'OK' if ok1 else 'DIFERENTE'}), "
              f"|Δh|={r['|Δh| mediana (%)']:.2f} vs {m2} ({'OK' if ok2 else 'DIFERENTE'})")

    df.to_csv(Path(__file__).resolve().parent / "baseline_validacion.csv", index=False)
    print("\nGuardado en experimentos/baseline_validacion.csv")


if __name__ == "__main__":
    main()
