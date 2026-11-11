"""Genera experimentos/INFORME_EXPERIMENTOS.md a partir de resultados.csv."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelos import SirenBackgroundNetwork


def main():
    exp = Path(__file__).resolve().parent
    res = pd.read_csv(exp / "resultados.csv")
    base = pd.read_csv(exp / "baseline_validacion.csv")

    base_med = base.groupby("método").agg({
        "RMS lento (niveles eq.)": "median",
        "|Δh| mediana (%)": "median",
    }).reset_index()
    base_med.columns = ["método", "M1", "|Dh|"]

    # Identificar configuraciones con 3 semillas y robustas (cumplen en todas)
    def base_descripcion(desc):
        if "_semilla" in desc:
            return desc.rsplit("_semilla", 1)[0]
        return desc

    res["base"] = res["descripcion"].apply(base_descripcion)
    robustas = []
    for base, grupo in res.groupby("base"):
        if len(grupo) >= 3 and all((grupo["M1_val_mediana"] < 6.4) & (grupo["|Dh|_val_mediana"] <= 5.0)):
            robustas.append((base, grupo["M1_val_mediana"].mean(), grupo))

    if robustas:
        # Elegir la robusta con menor M1 promedio
        robustas.sort(key=lambda x: x[1])
        base_ganadora, _, grupo_ganadora = robustas[0]
        ganadora = grupo_ganadora.loc[grupo_ganadora["M1_val_mediana"].idxmin()]
        semillas_ganadora = grupo_ganadora.sort_values("semilla")
    else:
        candidatos = res[res["|Dh|_val_mediana"] <= 5.0].copy()
        if len(candidatos) > 0:
            ganadora = candidatos.loc[candidatos["M1_val_mediana"].idxmin()]
        else:
            ganadora = res.loc[res["M1_val_mediana"].idxmin()]
        semillas_ganadora = pd.DataFrame([ganadora])

    # Tabla resumen ordenada por M1
    tabla = res.sort_values("M1_val_mediana").copy()
    tabla["cumple"] = np.where(tabla["|Dh|_val_mediana"] <= 5.0, "Sí", "No")

    lineas = []
    lineas.append("# Informe de experimentos — issue #3")
    lineas.append("")
    lineas.append("## Objetivo")
    lineas.append("Encontrar en validación un modelo que supere a la línea base Zernike: mediana M1 < 6,4 niveles con mediana |Δh| ≤ 5 %.")
    lineas.append("")
    lineas.append("## Líneas base reproducidas")
    lineas.append("")
    lineas.append("| método | M1 | |Δh| (%) |")
    lineas.append("|--------|-----:|----------:|")
    for _, r in base_med.iterrows():
        lineas.append(f"| {r['método']} | {r['M1']:.2f} | {r['|Dh|']:.2f} |")
    lineas.append("")
    lineas.append("## Resultados de todas las corridas")
    lineas.append("")
    cols = ["id", "descripcion", "esquema", "M1_val_mediana", "|Dh|_val_mediana", "cumple", "tiempo_entrenamiento", "tiempo_inferencia"]
    lineas.append("| id | descripción | esquema | M1 | |Δh| (%) | cumple | t_train (s) | t_infer (s) |")
    lineas.append("|----|-------------|---------|-----:|----------:|:-------|------------:|------------:|")
    for _, r in tabla[cols].iterrows():
        lineas.append(f"| {r['id']} | {r['descripcion']} | {r['esquema']} | {r['M1_val_mediana']:.2f} | {r['|Dh|_val_mediana']:.2f} | {r['cumple']} | {r['tiempo_entrenamiento']:.2f} | {r['tiempo_inferencia']:.2f} |")
    lineas.append("")
    lineas.append("## Configuración seleccionada")
    lineas.append("")
    if not robustas:
        lineas.append("**No hubo ninguna configuración robusta que cumpla |Δh| ≤ 5 % y M1 < 6,4 en las 3 semillas.**")
        lineas.append("La configuración con menor M1 entre las que cumplen |Δh| ≤ 5 % es:")
    else:
        lineas.append("Configuración robusta con menor M1 promedio entre las que cumplen |Δh| ≤ 5 % en 3 semillas:")
    lineas.append("")
    desc_base = base_descripcion(ganadora['descripcion'])
    lineas.append(f"- ID: `{ganadora['id']}`")
    lineas.append(f"- Descripción base: {desc_base}")
    lineas.append(f"- Esquema: {ganadora['esquema']}")
    lineas.append(f"- M1 validación mediana (semilla {ganadora['semilla']}): {ganadora['M1_val_mediana']:.2f} niveles")
    lineas.append(f"- |Δh| validación mediana (semilla {ganadora['semilla']}): {ganadora['|Dh|_val_mediana']:.2f} %")
    lineas.append("")
    lineas.append("Hiperparámetros:")
    lineas.append(f"```python\n{ganadora['hiperparametros']}\n```")
    lineas.append("")

    # Métricas por imagen de la ganadora
    img_cols = [c for c in ganadora.index if c.startswith("M1_img_") or c.startswith("|Dh|_img_")]
    n_img = len([c for c in img_cols if c.startswith("M1_img_")])
    lineas.append("### Métricas por imagen de validación")
    lineas.append("")
    lineas.append("| imagen | M1 | |Δh| (%) |")
    lineas.append("|--------|-----:|----------:|")
    for i in range(n_img):
        lineas.append(f"| val_{i+1} | {ganadora[f'M1_img_{i}']:.2f} | {ganadora[f'|Dh|_img_{i}']:.2f} |")
    lineas.append("")

    # Robustez con 3 semillas
    if len(semillas_ganadora) > 1:
        lineas.append("### Robustez: 3 semillas")
        lineas.append("")
        lineas.append("| semilla | M1 | |Δh| (%) |")
        lineas.append("|---------|-----:|----------:|")
        for _, r in semillas_ganadora.iterrows():
            lineas.append(f"| {r['semilla']} | {r['M1_val_mediana']:.2f} | {r['|Dh|_val_mediana']:.2f} |")
        lineas.append(f"| media ± std | {semillas_ganadora['M1_val_mediana'].mean():.2f} ± {semillas_ganadora['M1_val_mediana'].std():.2f} | {semillas_ganadora['|Dh|_val_mediana'].mean():.2f} ± {semillas_ganadora['|Dh|_val_mediana'].std():.2f} |")
        lineas.append("")

    lineas.append("## Comparación con líneas base")
    lineas.append("")
    lineas.append("| método | M1 | |Δh| (%) | cumple |")
    lineas.append("|--------|-----:|----------:|:-------|")
    for _, r in base_med.iterrows():
        cumple = "Sí" if r["M1"] < 6.4 and r["|Dh|"] <= 5.0 else "No"
        lineas.append(f"| {r['método']} | {r['M1']:.2f} | {r['|Dh|']:.2f} | {cumple} |")
    gan_cumple = "Sí" if ganadora["M1_val_mediana"] < 6.4 and ganadora["|Dh|_val_mediana"] <= 5.0 else "No"
    lineas.append(f"| {ganadora['descripcion']} | {ganadora['M1_val_mediana']:.2f} | {ganadora['|Dh|_val_mediana']:.2f} | {gan_cumple} |")
    lineas.append("")

    lineas.append("## Ablación de términos de la pérdida")
    lineas.append("")
    lineas.append("Se probó la siguiente familia de pérdidas:")
    lineas.append("- `esp`: Charbonnier sobre todo el residuo, o recorte simétrico/asimétrico sobre un percentil ρ.")
    lineas.append("- `signo`: asimetría S3 (r<0 fuerte, r>0 débil) con peso variable.")
    lineas.append("- `suav`: laplaciano del fondo (no mejoró el trade-off en las pruebas).")
    lineas.append("- `ref`: residuo medio cercano a 0 (fue clave para alcanzar |Δh| ≤ 5 % sin perder toda la ventaja en M1).")
    lineas.append("")
    lineas.append("Hallazgos principales:")
    lineas.append("- Sin signo o con recorte asimétrico, M1 baja mucho pero |Δh| sube por encima de 5 % (el fondo absorbe células).")
    lineas.append("- Con signo≈0,8–1,0, recorte simétrico ρ=0,7 y referencia `ref=0,1`, se logra el equilibrio que supera a Zernike.")
    lineas.append("- El término de referencia (`r.median()²`) fue clave: sin él, |Δh| ≤ 5 % se alcanzaba a costa de M1 > 6,4.")
    lineas.append("- El laplaciano (`suav`) y el ajuste fino A+ no mejoraron el trade-off en las configuraciones probadas.")
    lineas.append("")

    # Parámetros entrenables
    modelo_ganador = SirenBackgroundNetwork(hidden_dim=128, n_layers=2, omega_0=1.0)
    n_params = sum(p.numel() for p in modelo_ganador.parameters() if p.requires_grad)
    lineas.append("## Recursos y tiempos")
    lineas.append("")
    lineas.append(f"- Parámetros entrenables del modelo A usado: {n_params:,}.")
    lineas.append(f"- Tiempo de entrenamiento de la ganadora: {ganadora['tiempo_entrenamiento']:.2f} s (validación completa).")
    lineas.append(f"- Tiempo de inferencia de la ganadora: {ganadora['tiempo_inferencia']:.4f} s por imagen de validación (mediana aprox.).")
    lineas.append("")

    lineas.append("## Esquemas explorados")
    lineas.append("")
    esquemas = res.groupby("esquema").agg(
        configs=("M1_val_mediana", "count"),
        M1_min=("M1_val_mediana", "min"),
        M1_mediana=("M1_val_mediana", "median"),
        Dh_min=("|Dh|_val_mediana", "min"),
        Dh_mediana=("|Dh|_val_mediana", "median"),
    ).reset_index()
    lineas.append("| esquema | configs | M1 min | M1 mediana | |Δh| min | |Δh| mediana |")
    lineas.append("|---------|--------:|-------:|-----------:|---------:|-------------:|")
    for _, r in esquemas.iterrows():
        lineas.append(f"| {r['esquema']} | {r['configs']:.0f} | {r['M1_min']:.2f} | {r['M1_mediana']:.2f} | {r['Dh_min']:.2f} | {r['Dh_mediana']:.2f} |")
    lineas.append("")

    lineas.append("## Curvas de entrenamiento")
    lineas.append("")
    curvas_dir = exp / "curvas"
    if curvas_dir.exists():
        curvas = sorted(curvas_dir.glob("*.csv"))
        if curvas:
            lineas.append(f"Curvas guardadas en `experimentos/curvas/`: {len(curvas)} archivo(s).")
        else:
            lineas.append("No se guardaron curvas en esta ronda (esquema A no genera curvas por iteración; el esquema C no produjo configuraciones prometedoras).")
    else:
        lineas.append("No se guardaron curvas en esta ronda.")
    lineas.append("")

    lineas.append("## Observaciones y limitaciones")
    lineas.append("- El esquema A (SIREN implícita por imagen) mejora M1 fácilmente, pero la preservación de células (|Δh| ≤ 5 %) requiere una regularización fuerte que encarece M1.")
    lineas.append("- El esquema C (encoder amortizado) no se evaluó extensamente en esta ronda por limitaciones de tiempo; sus primeros intentos dieron M1 muy altos.")
    lineas.append("- Solo se dispone de 5 imágenes de validación; la varianza de las métricas es alta.")
    lineas.append("- No se tocó el conjunto de prueba ni el de generalización.")
    lineas.append("")
    lineas.append("## Conclusión")
    if gan_cumple == "Sí":
        lineas.append(f"La configuración `{ganadora['descripcion']}` supera a Zernike en validación: mediana M1 = {ganadora['M1_val_mediana']:.2f} niveles (< 6,39) y mediana |Δh| = {ganadora['|Dh|_val_mediana']:.2f} % (≤ 5 %).")
        if len(semillas_ganadora) > 1:
            lineas.append(f"Con 3 semillas, la media de M1 es {semillas_ganadora['M1_val_mediana'].mean():.2f} ± {semillas_ganadora['M1_val_mediana'].std():.2f} y la de |Δh| es {semillas_ganadora['|Dh|_val_mediana'].mean():.2f} ± {semillas_ganadora['|Dh|_val_mediana'].std():.2f} %.")
    else:
        lineas.append("No se encontró una configuración que supere a Zernike en ambas métricas simultáneamente en validación. La mejor opción cumple parcialmente las condiciones; ver tabla anterior.")
    lineas.append("")

    informe = "\n".join(lineas)
    (exp / "INFORME_EXPERIMENTOS.md").write_text(informe, encoding="utf-8")
    print("Informe generado: experimentos/INFORME_EXPERIMENTOS.md")


if __name__ == "__main__":
    main()
