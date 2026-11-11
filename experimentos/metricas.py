"""Métricas M1–M4 de la sección 6 del notebook."""

import numpy as np
from scipy import ndimage as ndi

R_INT, R_EXT, AREA_MIN, SIGMA_SUAVE = 6, 16, 600, 8


def celulas_evaluables(cel: np.ndarray, fon: np.ndarray) -> list:
    """Componentes de célula evaluables con su anillo de fondo."""
    lab, n = ndi.label(cel)
    salida = []
    for j in range(1, n + 1):
        cj = lab == j
        if cj.sum() < AREA_MIN:
            continue
        ys, xs = np.nonzero(cj)
        if (
            ys.min() < R_EXT
            or xs.min() < R_EXT
            or ys.max() >= cel.shape[0] - R_EXT
            or xs.max() >= cel.shape[1] - R_EXT
        ):
            continue
        ext = ndi.binary_dilation(cj, iterations=R_EXT)
        anillo_completo = ext & ~ndi.binary_dilation(cj, iterations=R_INT)
        anillo = ext & fon
        if anillo.sum() >= 0.5 * anillo_completo.sum():
            salida.append((cj, anillo))
    return salida


def fondo_lento(v: np.ndarray, fon: np.ndarray, sigma: float = SIGMA_SUAVE):
    """Promedio gaussiano normalizado del residuo en la región de fondo."""
    num = ndi.gaussian_filter(np.where(fon, v, 0.0), sigma)
    den = ndi.gaussian_filter(fon.astype(float), sigma)
    valido = fon & (den > 0.5)
    return np.where(valido, num / np.maximum(den, 1e-12), np.nan), valido


def soporte(v: np.ndarray, sigma: float) -> np.ndarray:
    """Máscara de célula a partir del residuo corregido."""
    return ndi.binary_fill_holes(ndi.binary_opening(v > 4 * sigma, iterations=2))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Coeficiente Dice entre dos máscaras booleanas."""
    inter = (a & b).sum()
    return float(2 * inter / (a.sum() + b.sum())) if (a.sum() + b.sum()) > 0 else 0.0


def metricas_imagen(v, u, cel, fon, sigma, celulas) -> dict:
    """Métricas de la imagen corregida v frente a la preprocesada u."""
    lento, valido = fondo_lento(v, fon)
    vb = v[fon]
    dh, corr = [], []
    for cj, rj in celulas:
        h_u = u[cj].mean() - u[rj].mean()
        h_v = v[cj].mean() - v[rj].mean()
        dh.append((h_v - h_u) / h_u)
        corr.append(np.corrcoef(v[cj], u[cj])[0, 1])
    d = dice(soporte(v, sigma), cel)
    return {
        "RMS lento (niveles eq.)": float(np.sqrt(np.mean(lento[valido] ** 2))),
        "Δh mediana (%)": 100 * float(np.median(dh)) if dh else np.nan,
        "|Δh| mediana (%)": 100 * float(np.median(np.abs(dh))) if dh else np.nan,
        "μ_B": float(vb.mean()),
        "σ_B": float(vb.std()),
        "RMS_B/σ_ruido": float(np.sqrt(np.mean(vb ** 2)) / sigma),
        "Dice": d,
        "IoU": d / (2 - d),
        "correlación mediana": float(np.median(corr)) if corr else np.nan,
        "células evaluadas": len(celulas),
    }


def metricas_archivo(v: np.ndarray, archivo: str, ctx: dict) -> dict:
    """Métricas usando el contexto cargado desde el cache."""
    idx = ctx["indices"][archivo]
    cel = ctx["mascaras_cel"][idx]
    fon = ctx["mascaras_fon"][idx]
    u = ctx["preprocesadas"][archivo]
    sigma = ctx["sigma_ruido"][archivo]
    celulas = ctx["celulas_eval"][archivo]
    return metricas_imagen(v, u, cel, fon, sigma, celulas)
