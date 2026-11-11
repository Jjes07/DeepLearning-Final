"""Entrenamiento e inferencia de los modelos A, C y C+."""

import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data import cargar, fijar_semillas, normalizar_por_imagen
from metricas import metricas_archivo
from modelos import BaseSuave, EncoderAmortizado, ImplicitFourierMLP, SirenBackgroundNetwork, build_coordinate_grid
from perdidas import perdida_fisica


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ajustar_fino_A(u, modelo, config):
    """Refina un modelo A ya entrenado con peso extra en signo."""
    if config.get("semilla") is not None:
        fijar_semillas(config["semilla"])
    device = _device()
    H, W = u.shape
    coords = build_coordinate_grid(H, W, device=device)
    u_t = torch.tensor(u, dtype=torch.float32, device=device).flatten()
    modelo = modelo.to(device)
    optim = torch.optim.Adam(modelo.parameters(), lr=config["lr"])
    pesos = config["pesos"]
    for it in range(config["iteraciones"]):
        optim.zero_grad()
        b = modelo(coords)
        r = u_t - b
        loss, _ = perdida_fisica(
            r, b, pesos, usar_signo=pesos.get("signo", 0) > 0,
            usar_suav=pesos.get("suav", 0) > 0, usar_ref=pesos.get("ref", 0) > 0,
            usar_recorte=pesos.get("recorte", False), rho_recorte=pesos.get("rho_recorte", 0.5),
            recorte_asim=pesos.get("recorte_asim", False)
        )
        loss.backward()
        optim.step()
    with torch.no_grad():
        b = modelo(coords).reshape(H, W).cpu().numpy()
    return b


def entrenar_A(u, modelo, config, ctx=None, archivo=None):
    """Entrena una red implícita por imagen (esquema A)."""
    if config.get("semilla") is not None:
        fijar_semillas(config["semilla"])
    device = _device()
    H, W = u.shape
    coords = build_coordinate_grid(H, W, device=device)
    u_t = torch.tensor(u, dtype=torch.float32, device=device).flatten()
    modelo = modelo.to(device)
    optim = torch.optim.Adam(modelo.parameters(), lr=config["lr"])

    pesos = config["pesos"]
    curva = []
    t0 = time.perf_counter()
    for it in range(config["iteraciones"]):
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
        curva.append({"iter": it, "loss": loss.item(), **{k: v.item() for k, v in terminos.items()}})

    t_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    with torch.no_grad():
        b = modelo(coords).reshape(H, W).cpu().numpy()
    t_infer = time.perf_counter() - t1
    return b, {"curva": pd.DataFrame(curva), "tiempo_entrenamiento": t_train, "tiempo_inferencia": t_infer}


def _aumentar_imagen(u):
    """Volteos y rotaciones 90° (8 variantes)."""
    imgs = [u]
    imgs.append(np.flipud(u))
    imgs.append(np.fliplr(u))
    imgs.append(np.rot90(u, k=1))
    imgs.append(np.rot90(u, k=2))
    imgs.append(np.rot90(u, k=3))
    imgs.append(np.flipud(np.rot90(u, k=1)))
    imgs.append(np.fliplr(np.rot90(u, k=1)))
    return imgs


def entrenar_C(imagenes_train, imagenes_val, config, ctx):
    """Entrena el encoder amortizado (esquema C) por batches."""
    if config.get("semilla") is not None:
        fijar_semillas(config["semilla"])
    device = _device()
    H, W = ctx["preprocesadas"][imagenes_train[0]].shape
    base = BaseSuave(H, W, f_max=config["f_max"], grado_poli=config["grado_poli"]).to(device)
    encoder = EncoderAmortizado(base.n_coef, base_channels=config["base_channels"]).to(device)

    # Tensor de entrenamiento con aumento (N, 1, H, W) normalizado
    train_list = []
    for archivo in imagenes_train:
        u = ctx["preprocesadas"][archivo]
        for ua in _aumentar_imagen(u):
            un, _, _ = normalizar_por_imagen(ua)
            train_list.append(un)
    train_tensor = torch.tensor(np.stack(train_list), dtype=torch.float32, device=device).unsqueeze(1)
    n_train = train_tensor.shape[0]
    batch_size = config.get("batch_size", min(8, n_train))

    optim = torch.optim.Adam(encoder.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=config.get("scheduler_step", 50), gamma=0.5)
    pesos = config["pesos"]

    curva = []
    t0 = time.perf_counter()
    for ep in range(config["epocas"]):
        encoder.train()
        perm = torch.randperm(n_train, device=device)
        losses = []
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            u_batch = train_tensor[idx]
            u_flat = u_batch.view(idx.shape[0], -1)
            optim.zero_grad()
            a = encoder(u_batch)
            b = base(a).view(idx.shape[0], -1)
            r = u_flat - b
            loss, _ = perdida_fisica(
                r, b, pesos, usar_signo=pesos.get("signo", 0) > 0,
                usar_suav=pesos.get("suav", 0) > 0, usar_ref=pesos.get("ref", 0) > 0,
                usar_recorte=pesos.get("recorte", False), rho_recorte=pesos.get("rho_recorte", 0.5),
                recorte_asim=pesos.get("recorte_asim", False)
            )
            loss.backward()
            optim.step()
            losses.append(loss.item())
        scheduler.step()

        # Validación
        val_m1, val_dh = [], []
        encoder.eval()
        with torch.no_grad():
            for archivo in imagenes_val:
                u = ctx["preprocesadas"][archivo]
                un, _, _ = normalizar_por_imagen(u)
                u_t = torch.tensor(un, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                a = encoder(u_t)
                b = base(a).reshape(H, W).cpu().numpy()
                media, std = u.mean(), u.std() + 1e-8
                b_real = b * std + media
                m = metricas_archivo(u - b_real, archivo, ctx)
                val_m1.append(m["RMS lento (niveles eq.)"])
                val_dh.append(m["|Δh| mediana (%)"])

        curva.append({
            "epoca": ep,
            "loss_train": float(np.mean(losses)),
            "M1_val": float(np.median(val_m1)),
            "|Δh|_val": float(np.median(val_dh)),
        })

    t_train = time.perf_counter() - t0
    return encoder, base, {"curva": pd.DataFrame(curva), "tiempo_entrenamiento": t_train}


def ajustar_fino_C(encoder, base, imagen, config, ctx, archivo):
    """Ajuste fino de coeficientes por imagen (esquema C+)."""
    device = _device()
    H, W = imagen.shape
    coords = build_coordinate_grid(H, W, device=device)
    un, media, std = normalizar_por_imagen(imagen)
    u_t = torch.tensor(un, dtype=torch.float32, device=device)

    encoder.eval()
    with torch.no_grad():
        a0 = encoder(u_t.unsqueeze(0).unsqueeze(0)).detach().clone()
    a = nn.Parameter(a0)
    optim = torch.optim.Adam([a], lr=config["lr_finetune"])
    pesos = config["pesos"]

    t0 = time.perf_counter()
    for it in range(config["iteraciones_finetune"]):
        optim.zero_grad()
        b = base(a).flatten()
        r = u_t.flatten() - b
        loss, _ = perdida_fisica(
            r, b, pesos, usar_signo=pesos.get("signo", 0) > 0,
            usar_suav=pesos.get("suav", 0) > 0, usar_ref=pesos.get("ref", 0) > 0,
            usar_recorte=pesos.get("recorte", False), rho_recorte=pesos.get("rho_recorte", 0.5),
            recorte_asim=pesos.get("recorte_asim", False)
        )
        loss.backward()
        optim.step()

    t_fine = time.perf_counter() - t0
    with torch.no_grad():
        b = base(a).reshape(H, W).cpu().numpy()
    b_real = b * std + media
    return b_real, {"tiempo_finetune": t_fine}


def inferir_C(encoder, base, imagen):
    """Predice fondo con modelo C sin ajuste fino."""
    device = _device()
    H, W = imagen.shape
    un, media, std = normalizar_por_imagen(imagen)
    u_t = torch.tensor(un, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    encoder.eval()
    with torch.no_grad():
        a = encoder(u_t)
        b = base(a).reshape(H, W).cpu().numpy()
    return b * std + media


def guardar_resultado(resultados, config, metricas_val, tiempos, esquema):
    """Guarda una fila en resultados.csv."""
    exp_dir = Path(__file__).resolve().parent
    csv_path = exp_dir / "resultados.csv"
    row = {
        "id": str(uuid.uuid4())[:8],
        "descripcion": config.get("descripcion", ""),
        "esquema": esquema,
        "hiperparametros": str(config),
        "semilla": config.get("semilla", 42),
        "M1_val_mediana": float(np.median([m["RMS lento (niveles eq.)"] for m in metricas_val])),
        "|Dh|_val_mediana": float(np.median([m["|Δh| mediana (%)"] for m in metricas_val])),
        "tiempo_entrenamiento": tiempos.get("tiempo_entrenamiento", np.nan),
        "tiempo_inferencia": tiempos.get("tiempo_inferencia", np.nan),
    }
    for i, m in enumerate(metricas_val):
        row[f"M1_img_{i}"] = m["RMS lento (niveles eq.)"]
        row[f"|Dh|_img_{i}"] = m["|Δh| mediana (%)"]

    df = pd.DataFrame([row])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    return row["id"]
