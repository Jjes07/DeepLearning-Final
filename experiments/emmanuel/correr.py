"""Entrenamiento e inferencia de los modelos A, C y C+."""

import gc
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import cargar, fijar_semillas, normalizar_por_imagen
from metricas import metricas_archivo
from modelos import (
    BaseSuave, EncoderAmortizado, EncoderCoordConv, ImplicitFourierMLP,
    ModeloDescomposicion, ModeloE, SirenBackgroundNetwork, build_coordinate_grid,
)
from perdidas import perdida_descomposicion, perdida_E, perdida_fisica


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _limpiar_memoria():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clip_batch_size(batch_size, maximo):
    return min(batch_size, maximo)


def _mejor_estado(modelo):
    return {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}


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


def _preparar_tensor_train(imagenes_train, ctx, device):
    """Normaliza y aumenta las imágenes de entrenamiento."""
    train_list = []
    for archivo in imagenes_train:
        u = ctx["preprocesadas"][archivo]
        for ua in _aumentar_imagen(u):
            un, _, _ = normalizar_por_imagen(ua)
            train_list.append(un)
    return torch.tensor(np.stack(train_list), dtype=torch.float32, device=device).unsqueeze(1)


def _val_loss_C(encoder, base, imagenes_val, pesos, device, ctx):
    """Pérdida de validación del esquema C sin usar máscaras."""
    encoder.eval()
    losses = []
    with torch.no_grad():
        for archivo in imagenes_val:
            u = ctx["preprocesadas"][archivo]
            un, _, _ = normalizar_por_imagen(u)
            u_t = torch.tensor(un, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            a = encoder(u_t)
            b = base(a).view(1, -1)
            r = u_t.view(1, -1) - b
            loss, _ = perdida_fisica(
                r, b, pesos, usar_signo=pesos.get("signo", 0) > 0,
                usar_suav=pesos.get("suav", 0) > 0, usar_ref=pesos.get("ref", 0) > 0,
                usar_recorte=pesos.get("recorte", False), rho_recorte=pesos.get("rho_recorte", 0.5),
                recorte_asim=pesos.get("recorte_asim", False)
            )
            losses.append(loss.item())
    return float(np.mean(losses))


def entrenar_C(imagenes_train, imagenes_val, config, ctx):
    """Entrena el encoder amortizado (esquema C) por batches."""
    if config.get("semilla") is not None:
        fijar_semillas(config["semilla"])
    device = _device()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
    H, W = ctx["preprocesadas"][imagenes_train[0]].shape
    base = BaseSuave(H, W, f_max=config["f_max"], grado_poli=config["grado_poli"]).to(device)
    encoder = EncoderAmortizado(base.n_coef, base_channels=config["base_channels"]).to(device)

    train_tensor = _preparar_tensor_train(imagenes_train, ctx, device)
    n_train = train_tensor.shape[0]
    batch_size = _clip_batch_size(config.get("batch_size", min(8, n_train)), 16)

    optim = torch.optim.Adam(encoder.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=config.get("scheduler_step", 50), gamma=0.5)
    pesos = config["pesos"]

    mejor_loss = float("inf")
    mejor_estado = None
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

        loss_val = _val_loss_C(encoder, base, imagenes_val, pesos, device, ctx)
        if loss_val < mejor_loss:
            mejor_loss = loss_val
            mejor_estado = {
                "encoder": _mejor_estado(encoder),
                "base": _mejor_estado(base),
                "epoca": ep,
            }

        # Métricas de validación solo cada 10 épocas para ahorrar tiempo.
        if ep % 10 == 0 or ep == config["epocas"] - 1:
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
                "loss_val": loss_val,
                "M1_val": float(np.median(val_m1)),
                "|Δh|_val": float(np.median(val_dh)),
            })
        else:
            curva.append({
                "epoca": ep,
                "loss_train": float(np.mean(losses)),
                "loss_val": loss_val,
                "M1_val": float("nan"),
                "|Δh|_val": float("nan"),
            })

    if mejor_estado is not None:
        encoder.load_state_dict(mejor_estado["encoder"])
        base.load_state_dict(mejor_estado["base"])

    t_train = time.perf_counter() - t0
    resultado = {"curva": pd.DataFrame(curva), "tiempo_entrenamiento": t_train,
                 "mejor_epoca": mejor_estado["epoca"] if mejor_estado else config["epocas"] - 1}
    del optim
    _limpiar_memoria()
    return encoder, base, resultado


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


def _val_loss_D(modelo, val_tensor, pesos):
    """Pérdida de validación del esquema D en escala normalizada."""
    modelo.eval()
    with torch.no_grad():
        b, c = modelo(val_tensor)
        loss, _ = perdida_descomposicion(val_tensor, b, c, pesos)
    return loss.item()


def entrenar_D(imagenes_train, imagenes_val, config, ctx):
    """Entrena el modelo de descomposición fondo + célula (esquema D)."""
    if config.get("semilla") is not None:
        fijar_semillas(config["semilla"])
    device = _device()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
    H, W = ctx["preprocesadas"][imagenes_train[0]].shape
    modelo = ModeloDescomposicion(
        H, W, f_max=config["f_max"], grado_poli=config["grado_poli"],
        base_channels=config["base_channels"]
    ).to(device)

    train_tensor = _preparar_tensor_train(imagenes_train, ctx, device)
    n_train = train_tensor.shape[0]
    batch_size = _clip_batch_size(config.get("batch_size", min(8, n_train)), 8)

    # Tensor de validación normalizado (sin aumento)
    val_norm, val_media, val_std = [], [], []
    for archivo in imagenes_val:
        u = ctx["preprocesadas"][archivo]
        un, media, std = normalizar_por_imagen(u)
        val_norm.append(un)
        val_media.append(media)
        val_std.append(std)
    val_tensor = torch.tensor(np.stack(val_norm), dtype=torch.float32, device=device).unsqueeze(1)

    optim = torch.optim.Adam(modelo.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=config.get("scheduler_step", 50), gamma=0.5)
    scaler = torch.amp.GradScaler("cuda")
    pesos = config["pesos"]

    mejor_loss = float("inf")
    mejor_estado = None
    curva = []
    t0 = time.perf_counter()
    for ep in range(config["epocas"]):
        modelo.train()
        perm = torch.randperm(n_train, device=device)
        losses = []
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            u_batch = train_tensor[idx]
            optim.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                b, c = modelo(u_batch)
                loss, _ = perdida_descomposicion(u_batch, b, c, pesos)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        loss_val = _val_loss_D(modelo, val_tensor, pesos)
        if loss_val < mejor_loss:
            mejor_loss = loss_val
            mejor_estado = {"modelo": _mejor_estado(modelo), "epoca": ep}

        # Métricas de validación solo cada 10 épocas para ahorrar tiempo.
        if ep % 10 == 0 or ep == config["epocas"] - 1:
            val_m1, val_dh = [], []
            modelo.eval()
            with torch.no_grad():
                b_val, _ = modelo(val_tensor)
                b_val = b_val.cpu().numpy()
                for j, archivo in enumerate(imagenes_val):
                    u = ctx["preprocesadas"][archivo]
                    b_real = b_val[j, 0] * val_std[j] + val_media[j]
                    m = metricas_archivo(u - b_real, archivo, ctx)
                    val_m1.append(m["RMS lento (niveles eq.)"])
                    val_dh.append(m["|Δh| mediana (%)"])
            curva.append({
                "epoca": ep,
                "loss_train": float(np.mean(losses)),
                "loss_val": loss_val,
                "M1_val": float(np.median(val_m1)),
                "|Δh|_val": float(np.median(val_dh)),
            })
        else:
            curva.append({
                "epoca": ep,
                "loss_train": float(np.mean(losses)),
                "loss_val": loss_val,
                "M1_val": float("nan"),
                "|Δh|_val": float("nan"),
            })

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado["modelo"])

    t_train = time.perf_counter() - t0
    resultado = {"curva": pd.DataFrame(curva), "tiempo_entrenamiento": t_train,
                 "mejor_epoca": mejor_estado["epoca"] if mejor_estado else config["epocas"] - 1}
    del optim
    _limpiar_memoria()
    return modelo, resultado


def inferir_D(modelo, imagen):
    """Predice fondo con el esquema D; la corrección usa solo el fondo."""
    device = _device()
    un, media, std = normalizar_por_imagen(imagen)
    u_t = torch.tensor(un, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    modelo.eval()
    with torch.no_grad():
        b, _ = modelo(u_t)
        b = b.reshape(imagen.shape).cpu().numpy()
    return b * std + media


def ajustar_fino_D(modelo, imagen, config, pesos):
    """Ajuste fino per imagen de los coeficientes del fondo con ĉ fija (D+)."""
    device = _device()
    un, media, std = normalizar_por_imagen(imagen)
    u_t = torch.tensor(un, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    modelo.eval()
    with torch.no_grad():
        c = modelo.unet(u_t).detach()
        feats = F.adaptive_avg_pool2d(modelo.backbone(u_t), 1).view(1, -1)
        a0 = modelo.fc_coef(feats).detach().clone()

    a = nn.Parameter(a0)
    optim = torch.optim.Adam([a], lr=config["lr_finetune"])

    t0 = time.perf_counter()
    for _ in range(config["iteraciones_finetune"]):
        optim.zero_grad()
        b = modelo.base(a)
        loss, _ = perdida_descomposicion(u_t, b, c, pesos)
        loss.backward()
        optim.step()

    t_fine = time.perf_counter() - t0
    with torch.no_grad():
        b = modelo.base(a).reshape(imagen.shape).cpu().numpy()
    return b * std + media, {"tiempo_finetune": t_fine}


def guardar_curva(curva_df, config_id, esquema):
    """Guarda la curva de entrenamiento en experimentos/curvas/<id>.csv."""
    exp_dir = Path(__file__).resolve().parent
    curvas_dir = exp_dir / "curvas"
    curvas_dir.mkdir(exist_ok=True)
    curva_df.to_csv(curvas_dir / f"{esquema}_{config_id}.csv", index=False)


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


def _ruido_sin_mascara(un):
    """Desviación robusta de diferencias entre vecinos, en escala normalizada."""
    dx = np.diff(un, axis=1).ravel()
    dy = np.diff(un, axis=0).ravel()
    dif = np.concatenate((dx, dy))
    return float(np.median(np.abs(dif - np.median(dif))) / (0.67448975 * np.sqrt(2)))


def entrenar_E(imagenes_train, imagenes_val, config, ctx):
    """Entrena E; la validación y el checkpoint no utilizan máscaras."""
    fijar_semillas(config["semilla"])
    device = _device()
    H, W = ctx["preprocesadas"][imagenes_train[0]].shape
    modelo = ModeloE(H, W, f_max=config["f_max"], grado_poli=3,
                     base_channels=config["base_channels"]).to(device)
    def preparar(nombres):
        xs, ks, medias, stds = [], [], [], []
        for nombre in nombres:
            un, media, std = normalizar_por_imagen(ctx["preprocesadas"][nombre])
            xs.append(un)
            ks.append(_ruido_sin_mascara(un))
            medias.append(media)
            stds.append(std)
        return (torch.tensor(np.stack(xs), dtype=torch.float32, device=device).unsqueeze(1),
                torch.tensor(ks, dtype=torch.float32, device=device).view(-1, 1, 1, 1),
                medias, stds)
    train, kt, _, _ = preparar(imagenes_train)
    val, kv, medias, stds = preparar(imagenes_val)
    opt = torch.optim.Adam(modelo.parameters(), lr=config.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    batch = min(config.get("batch_size", 4), 8)
    pesos = {"fondo": 1.0, "signo": 1.0, "area": config["lambda_area"], "tv_w": 0.001}
    mejor_loss, mejor_estado, mejor_epoca = float("inf"), None, -1
    curva = []
    t0 = time.perf_counter()
    for ep in range(config["epocas"]):
        modelo.train()
        perdidas = []
        for idx in torch.randperm(len(train), device=device).split(batch):
            imgs = []
            for imagen in train[idx]:
                imagen = torch.rot90(imagen, int(torch.randint(4, ()).item()), (-2, -1))
                if torch.rand(()).item() < 0.5:
                    imagen = imagen.flip(-1)
                imgs.append(imagen)
            ub = torch.stack(imgs)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                b, w = modelo(ub)
                loss, _ = perdida_E(ub, b, w.float(), pesos, kt[idx])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            perdidas.append(loss.item())
        scheduler.step()
        modelo.eval()
        val_losses, fondos = [], []
        with torch.no_grad():
            for i in range(0, len(val), batch):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    b, w = modelo(val[i:i+batch])
                    lv, _ = perdida_E(val[i:i+batch], b, w.float(), pesos, kv[i:i+batch])
                val_losses.extend([lv.item()] * len(b))
                if ep == 0 or (ep + 1) % 10 == 0 or ep == config["epocas"] - 1:
                    fondos.extend(b[:, 0].cpu().numpy())
        vl = float(np.mean(val_losses))
        if vl < mejor_loss:
            mejor_loss, mejor_estado, mejor_epoca = vl, _mejor_estado(modelo), ep
        fila = {"epoca": ep + 1, "loss_train": float(np.mean(perdidas)), "loss_val": vl,
                "M1_val": float("nan"), "dh_val": float("nan")}
        if fondos:
            ms = []
            for j, nombre in enumerate(imagenes_val):
                u = ctx["preprocesadas"][nombre]
                b_real = fondos[j] * stds[j] + medias[j]
                ms.append(metricas_archivo(u - b_real, nombre, ctx))
            fila["M1_val"] = float(np.median([m["RMS lento (niveles eq.)"] for m in ms]))
            fila["dh_val"] = float(np.median([m["|Δh| mediana (%)"] for m in ms]))
        curva.append(fila)
        if (ep + 1) % 50 == 0:
            print(f"  época {ep+1}: pérdida val={vl:.5f}", flush=True)
    modelo.load_state_dict(mejor_estado)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return modelo, {"curva": pd.DataFrame(curva), "mejor_epoca": mejor_epoca + 1,
                    "loss_val": mejor_loss, "tiempo_entrenamiento": time.perf_counter() - t0}


def inferir_E(modelo, imagen):
    """Una pasada, sin optimización por imagen."""
    un, media, std = normalizar_por_imagen(imagen)
    u = torch.tensor(un, dtype=torch.float32, device=next(modelo.parameters()).device)[None, None]
    modelo.eval()
    with torch.no_grad():
        b, w = modelo(u)
    return b[0, 0].cpu().numpy() * std + media, w[0, 0].cpu().numpy()
