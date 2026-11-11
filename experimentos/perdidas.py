"""Funciones de pérdida física para estimación de fondo."""

import torch
import torch.nn.functional as F


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x ** 2 + eps ** 2)


def laplaciano2d(b):
    """Laplaciano discreto de una imagen b (B, 1, H, W)."""
    pad = F.pad(b, (1, 1, 1, 1), mode="replicate")
    return (
        pad[:, :, 2:, 1:-1]
        + pad[:, :, :-2, 1:-1]
        + pad[:, :, 1:-1, 2:]
        + pad[:, :, 1:-1, :-2]
        - 4 * b
    )


def tv2d(x):
    """Total variation anisotrópica de x (B, 1, H, W)."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return torch.abs(dx).mean() + torch.abs(dy).mean()


def perdida_descomposicion(u, b, c, pesos):
    """
    u, b, c: tensores (B, 1, H, W) en escala normalizada.
    pesos: dict con 'lambda_c', 'lambda_tv', 'ref', 'suav'.
    """
    r = u - b - c
    terminos = {"esp": charbonnier(r).mean()}
    terminos["c_l1"] = c.mean()
    terminos["c_tv"] = tv2d(c)
    terminos["ref"] = r.median() ** 2
    if pesos.get("suav", 0) > 0:
        b2d = b.unsqueeze(1) if b.dim() == 3 else b
        terminos["suav"] = (laplaciano2d(b2d) ** 2).mean()
    else:
        terminos["suav"] = torch.tensor(0.0, device=u.device)

    total = (
        terminos["esp"]
        + pesos.get("lambda_c", 0.0) * terminos["c_l1"]
        + pesos.get("lambda_tv", 0.0) * terminos["c_tv"]
        + pesos.get("ref", 0.0) * terminos["ref"]
        + pesos.get("suav", 0.0) * terminos["suav"]
    )
    return total, terminos


def perdida_fisica(r, b, pesos, usar_signo=True, usar_suav=True, usar_ref=True, usar_recorte=False, rho_recorte=0.5, recorte_asim=False):
    """
    r: residuo u - b (tensor 1D o 2D).
    b: fondo predicho (tensor 1D o 2D).
    pesos: dict con 'esp', 'signo', 'suav', 'ref'.
    """
    terminos = {}
    if usar_recorte:
        n = r.numel()
        n_keep = max(1, int(rho_recorte * n))
        if recorte_asim:
            vals, _ = torch.sort(r)
            terminos["esp"] = vals[:n_keep].mean()
        else:
            vals, _ = torch.sort(torch.abs(r))
            terminos["esp"] = vals[:n_keep].mean()
    else:
        terminos["esp"] = charbonnier(r).mean()

    if usar_signo:
        neg = torch.relu(-r) ** 2
        pos = torch.relu(r)
        terminos["signo"] = pesos.get("signo_neg", 1.0) * neg.mean() + pesos.get("signo_pos", 0.01) * pos.mean()
    else:
        terminos["signo"] = torch.tensor(0.0, device=r.device)

    if usar_suav:
        if b.dim() == 1:
            h = w = int(b.numel() ** 0.5)
            b2d = b.view(1, 1, h, w)
        elif b.dim() == 2:
            b2d = b.unsqueeze(0).unsqueeze(0)
        else:
            b2d = b
        terminos["suav"] = (laplaciano2d(b2d) ** 2).mean()
    else:
        terminos["suav"] = torch.tensor(0.0, device=r.device)

    if usar_ref:
        terminos["ref"] = r.median() ** 2
    else:
        terminos["ref"] = torch.tensor(0.0, device=r.device)

    total = (
        pesos.get("esp", 1.0) * terminos["esp"]
        + pesos.get("signo", 0.0) * terminos["signo"]
        + pesos.get("suav", 0.0) * terminos["suav"]
        + pesos.get("ref", 0.0) * terminos["ref"]
    )
    return total, terminos


def perdida_E(u, b, w, pesos, kappa=1.0):
    """
    Pérdida sin etiquetas para el esquema E.
    u, b, w: tensores (B, 1, H, W) en escala normalizada por imagen.
    pesos: dict con 'fondo', 'signo', 'area', 'tv_w'.
    kappa: umbral del término de signo (~1 desviación del ruido normalizada).
    """
    r = u - b
    # Ajuste en la región que la red considera fondo.
    loss_fondo = ((w * charbonnier(r)).flatten(1).sum(1) / (w.flatten(1).sum(1) + 1e-8)).mean()

    # Signo S3: el fondo nunca debe quedar por encima de la imagen.
    loss_signo = torch.relu(-r - kappa).pow(2).mean()

    # Prior de área inferior: al menos ~60 % de píxeles deben ser fondo.
    mean_w = w.mean(dim=(1, 2, 3))
    loss_area = torch.relu(0.6 - mean_w).pow(2).mean()

    # Suavidad espacial opcional de w para regiones compactas.
    loss_tv_w = tv2d(w)

    terminos = {
        "fondo": loss_fondo,
        "signo": loss_signo,
        "area": loss_area,
        "tv_w": loss_tv_w,
    }

    total = (
        pesos.get("fondo", 1.0) * terminos["fondo"]
        + pesos.get("signo", 0.0) * terminos["signo"]
        + pesos.get("area", 0.0) * terminos["area"]
        + pesos.get("tv_w", 0.0) * terminos["tv_w"]
    )
    return total, terminos
