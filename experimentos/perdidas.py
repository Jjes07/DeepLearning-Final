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
