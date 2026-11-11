"""Ajuste AE y evaluación separada de las imágenes de validación."""

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import torch

from data import fijar_semillas, normalizar_por_imagen
from modelos import ModeloE, SirenBackgroundNetwork, build_coordinate_grid
from perdidas import charbonnier

EXP = Path(__file__).resolve().parent
SALIDA = EXP / "pesos" / "salidas_ae"
SEMILLAS = (42, 2024, 12345)
VARIANTES = ("w", "w05", "w025", "w_recorte")


def cargar_imagenes():
    # El archivo de referencia permite identificar solo los nombres de validación.
    with (EXP / "baseline_validacion.csv").open(encoding="utf-8", newline="") as f:
        nombres = list(dict.fromkeys(fila["archivo"] for fila in csv.DictReader(f)))
    with np.load(EXP / "cache_dev.npz") as datos:
        archivos = datos["archivos"].tolist()
        pre = datos["preprocesadas"]
        imagenes = {n: pre[archivos.index(n)].copy() for n in nombres}
    return nombres, imagenes


def mapa_e(imagen, semilla, device):
    ruta = EXP / "pesos" / f"E_f6_a0.5_c16_s{semilla}.pth"
    guardado = torch.load(ruta, map_location="cpu", weights_only=False)
    cfg = guardado["config"]
    if cfg["f_max"] != 6 or cfg["semilla"] != semilla:
        raise ValueError(f"Configuración inesperada: {ruta}")
    h, w = imagen.shape
    modelo = ModeloE(h, w, f_max=6, grado_poli=3, base_channels=cfg["base_channels"])
    modelo.load_state_dict(guardado["estado"])
    modelo = modelo.to(device).eval()
    un, _, _ = normalizar_por_imagen(imagen)
    with torch.no_grad():
        entrada = torch.tensor(un, dtype=torch.float32, device=device)[None, None]
        mapa = modelo.unet(entrada)[0, 0].cpu().numpy()
    del modelo, entrada, guardado
    gc.collect()
    torch.cuda.empty_cache()
    return mapa


def ajustar(imagen, mapa, variante, semilla, device, plazo):
    fijar_semillas(semilla)
    modelo = SirenBackgroundNetwork(hidden_dim=128, n_layers=2, omega_0=1.0).to(device)
    h, w = imagen.shape
    coords = build_coordinate_grid(h, w, device=device)
    u = torch.tensor(imagen, dtype=torch.float32, device=device).flatten()
    pesos = torch.tensor(mapa, dtype=torch.float32, device=device).flatten()
    if variante == "w05":
        pesos = pesos.pow(0.5)
    elif variante == "w025":
        pesos = pesos.pow(0.25)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    inicio = time.perf_counter()
    for paso in range(1000):
        if paso % 10 == 0 and time.monotonic() >= plazo:
            raise TimeoutError("Límite de cómputo alcanzado durante el ajuste")
        opt.zero_grad()
        fondo = modelo(coords)
        r = u - fondo
        if variante == "w_recorte":
            n = max(1, int(0.7 * r.numel()))
            indices = torch.topk(r.abs(), n, largest=False, sorted=False).indices
            ajuste = (pesos[indices] * charbonnier(r[indices])).sum() / pesos[indices].sum().clamp_min(1e-8)
        else:
            ajuste = (pesos * charbonnier(r)).sum() / pesos.sum().clamp_min(1e-8)
        signo = torch.relu(-r).square().mean() + 0.01 * torch.relu(r).mean()
        perdida = ajuste + 0.8 * signo + 0.1 * r.median().square()
        if not torch.isfinite(perdida):
            raise RuntimeError("Pérdida no finita")
        perdida.backward()
        opt.step()
    torch.cuda.synchronize()
    tiempo = time.perf_counter() - inicio
    with torch.no_grad():
        fondo = modelo(coords).reshape(h, w).cpu().numpy()
    del modelo, coords, u, pesos, opt, r, perdida
    gc.collect()
    torch.cuda.empty_cache()
    return fondo, tiempo


def ejecutar(limite):
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere CUDA")
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    nombres, imagenes = cargar_imagenes()
    SALIDA.mkdir(exist_ok=True)
    inicio = time.monotonic()
    plazo = inicio + min(limite, 2700)
    for variante in VARIANTES:
        for semilla in SEMILLAS:
            mapas = {}
            for i, nombre in enumerate(nombres):
                destino = SALIDA / f"{variante}_s{semilla}_i{i}.npz"
                if destino.exists():
                    print(f"Ya existe: {destino.name}", flush=True)
                    continue
                if time.monotonic() >= plazo:
                    print("Límite alcanzado; resultados parciales conservados.", flush=True)
                    return
                imagen = imagenes[nombre]
                if i not in mapas:
                    mapas[i] = mapa_e(imagen, semilla, "cuda")
                try:
                    fondo, tiempo = ajustar(imagen, mapas[i], variante, semilla, "cuda", plazo)
                except TimeoutError:
                    print("Límite alcanzado; resultados parciales conservados.", flush=True)
                    return
                np.savez_compressed(destino, nombre=nombre, fondo=fondo, w=mapas[i], segundos=tiempo)
                print(f"{variante} semilla {semilla} imagen {i+1}/{len(nombres)}: {tiempo:.1f} s", flush=True)


def evaluar():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from metricas import celulas_evaluables, metricas_imagen

    salida_csv = EXP / "resultados_ronda4.csv"
    if salida_csv.exists():
        raise FileExistsError(f"No se sobrescribe {salida_csv}")
    nombres, imagenes = cargar_imagenes()
    with np.load(EXP / "cache_dev.npz") as datos:
        archivos = datos["archivos"].tolist()
        indices = [archivos.index(nombre) for nombre in nombres]
        if any(datos["conjuntos"][i] != "validacion" for i in indices):
            raise ValueError("La lista contiene imágenes fuera de validación")
        celulas = datos["mascaras_cel"][indices]
        fondos = datos["mascaras_fon"][indices]
        sigmas = datos["sigma_ruido"][indices]
    evaluables = [celulas_evaluables(cel, fon) for cel, fon in zip(celulas, fondos)]
    filas = []
    for variante in VARIANTES:
        for semilla in SEMILLAS:
            medidas, tiempos = [], []
            for i, nombre in enumerate(nombres):
                ruta = SALIDA / f"{variante}_s{semilla}_i{i}.npz"
                if not ruta.exists():
                    print(f"Sin completar: {variante} semilla {semilla}", flush=True)
                    break
                with np.load(ruta) as salida:
                    if str(salida["nombre"]) != nombre:
                        raise ValueError(f"Orden inconsistente: {ruta}")
                    fondo = salida["fondo"]
                    tiempos.append(float(salida["segundos"]))
                u = imagenes[nombre]
                medidas.append(metricas_imagen(u - fondo, u, celulas[i], fondos[i],
                                                float(sigmas[i]), evaluables[i]))
            if len(medidas) != len(nombres):
                continue
            fila = {"variante": variante, "semilla": semilla,
                    "M1": float(np.median([m["RMS lento (niveles eq.)"] for m in medidas])),
                    "dh": float(np.median([m["|Δh| mediana (%)"] for m in medidas])),
                    "tiempo_medio": float(np.mean(tiempos))}
            for i, (nombre, m) in enumerate(zip(nombres, medidas)):
                fila.update({f"imagen_{i}": nombre, f"M1_{i}": m["RMS lento (niveles eq.)"],
                             f"dh_{i}": m["|Δh| mediana (%)"]})
            filas.append(fila)
            pd.DataFrame(filas).to_csv(salida_csv, index=False)
    if not filas:
        raise RuntimeError("No hay semillas completas")
    df = pd.DataFrame(filas)
    grupos = df.groupby("variante").filter(lambda x: len(x) == 3).groupby("variante")
    resumen = grupos.agg(M1_media=("M1", "mean"), M1_std=("M1", "std"),
                         dh_media=("dh", "mean"), dh_std=("dh", "std"),
                         tiempo=("tiempo_medio", "mean"))
    print(resumen.to_string(), flush=True)
    elegibles = resumen[resumen.dh_media <= 5]
    if elegibles.empty:
        print("Ninguna variante completa cumple la restricción.", flush=True)
        return
    mejor = elegibles.M1_media.idxmin()
    print(f"Mejor: {mejor}", flush=True)
    fig, ejes = plt.subplots(len(nombres), 4, figsize=(14, 3 * len(nombres)),
                             constrained_layout=True, squeeze=False)
    titulo_peso = "Peso de fondo w (sin recorte)" if mejor == "w_recorte" else "Peso de fondo"
    for j, titulo in enumerate(("Imagen", titulo_peso, "Fondo estimado", "Imagen corregida")):
        ejes[0, j].set_title(titulo)
    for i, nombre in enumerate(nombres):
        with np.load(SALIDA / f"{mejor}_s42_i{i}.npz") as salida:
            fondo, mapa = salida["fondo"], salida["w"]
        if mejor == "w05":
            mapa = mapa ** 0.5
        elif mejor == "w025":
            mapa = mapa ** 0.25
        u = imagenes[nombre]
        for j, imagen in enumerate((u, mapa, fondo, u - fondo)):
            opciones = {"cmap": "viridis", "vmin": 0, "vmax": 1} if j == 1 else {"cmap": "gray"}
            grafico = ejes[i, j].imshow(imagen, **opciones)
            ejes[i, j].axis("off")
            if j == 1:
                fig.colorbar(grafico, ax=ejes[i, j], fraction=0.046, pad=0.02)
        ejes[i, 0].text(0.01, 0.02, nombre, transform=ejes[i, 0].transAxes,
                        fontsize=6, color="white", bbox={"facecolor": "black", "alpha": 0.6})
    fig.suptitle(f"AE {mejor}, semilla 42: validación", fontsize=13)
    (EXP / "figuras").mkdir(exist_ok=True)
    fig.savefig(EXP / "figuras" / "AE_validacion.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fase", choices=("ajustar", "evaluar"))
    parser.add_argument("--limite", type=float, default=2650)
    args = parser.parse_args()
    ejecutar(args.limite) if args.fase == "ajustar" else evaluar()
