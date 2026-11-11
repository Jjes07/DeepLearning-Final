"""Evalúa configuraciones prometedoras con 3 semillas para robustez."""

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from correr import entrenar_A, guardar_resultado
from data import cargar
from modelos import SirenBackgroundNetwork


def main():
    d = cargar()
    val = d["validacion"]

    candidatas = [
        ("A_cand_signo1.0_ref0.1_rec0.75", {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.01,
                                             "suav": 0.0, "ref": 0.1, "recorte": True, "rho_recorte": 0.75}),
        ("A_cand_signo1.0_ref0.1_rec0.7", {"esp": 1.0, "signo": 1.0, "signo_neg": 1.0, "signo_pos": 0.01,
                                             "suav": 0.0, "ref": 0.1, "recorte": True, "rho_recorte": 0.7}),
        ("A_cand_signo0.8_ref0.1_rec0.7", {"esp": 1.0, "signo": 0.8, "signo_neg": 1.0, "signo_pos": 0.01,
                                             "suav": 0.0, "ref": 0.1, "recorte": True, "rho_recorte": 0.7}),
        ("A_cand_signo1.5_ref0.5_rec0.7", {"esp": 1.0, "signo": 1.5, "signo_neg": 1.0, "signo_pos": 0.01,
                                             "suav": 0.0, "ref": 0.5, "recorte": True, "rho_recorte": 0.7}),
    ]

    semillas = [42, 2024, 12345]
    resumen = []
    for nombre_base, pesos in candidatas:
        config_base = {
            "modelo_kwargs": {"hidden_dim": 128, "n_layers": 2, "omega_0": 1.0},
            "lr": 1e-3,
            "iteraciones": 1000,
            "pesos": pesos,
        }
        m1_sem, dh_sem = [], []
        for semilla in semillas:
            d["ctx"]["fijar_semillas"](semilla)
            config = {**config_base, "semilla": semilla}
            metricas = []
            tiempos_train = []
            tiempos_infer = []
            for archivo in val:
                u = d["ctx"]["preprocesadas"][archivo]
                b, info = entrenar_A(u, SirenBackgroundNetwork(**config["modelo_kwargs"]), config)
                m = d["metricas_archivo"](u - b, archivo)
                metricas.append(m)
                tiempos_train.append(info["tiempo_entrenamiento"])
                tiempos_infer.append(info["tiempo_inferencia"])
            m1 = [m["RMS lento (niveles eq.)"] for m in metricas]
            dh = [m["|Δh| mediana (%)"] for m in metricas]
            nombre = f"{nombre_base}_semilla{semilla}"
            print(f"{nombre}: M1={np.median(m1):.2f}, |Dh|={np.median(dh):.2f}")
            guardar_resultado(None, {"descripcion": nombre, **config}, metricas,
                              {"tiempo_entrenamiento": sum(tiempos_train), "tiempo_inferencia": sum(tiempos_infer)}, "A")
            m1_sem.append(np.median(m1))
            dh_sem.append(np.median(dh))
        cumple_todas = all(m < 6.4 for m in m1_sem) and all(h <= 5.0 for h in dh_sem)
        resumen.append((nombre_base, np.mean(m1_sem), np.std(m1_sem), np.mean(dh_sem), np.std(dh_sem), cumple_todas))
        print(f"  -> media M1={np.mean(m1_sem):.2f}±{np.std(m1_sem):.2f}, |Dh|={np.mean(dh_sem):.2f}±{np.std(dh_sem):.2f}, cumple_todas={cumple_todas}\n")

    print("Resumen:")
    for nombre, m1_mean, m1_std, dh_mean, dh_std, cumple in resumen:
        print(f"{nombre}: M1={m1_mean:.2f}±{m1_std:.2f}, |Dh|={dh_mean:.2f}±{dh_std:.2f}, cumple_todas={cumple}")


if __name__ == "__main__":
    main()
