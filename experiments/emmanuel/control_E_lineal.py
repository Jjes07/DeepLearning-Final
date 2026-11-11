"""Control amortizado con coordenadas explícitas y base suave."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import cargar, fijar_semillas, normalizar_por_imagen
from metricas import metricas_archivo
from modelos import BaseSuave, EncoderCoordConv
from perdidas import perdida_fisica


def _perdida_por_imagen(u, fondo, pesos):
    """Calcula recorte y mediana independientemente para cada imagen."""
    valores = []
    for imagen, estimado in zip(u, fondo):
        r = (imagen - estimado).flatten()
        b = estimado.flatten()
        perdida, _ = perdida_fisica(
            r, b, pesos, usar_signo=True, usar_suav=False, usar_ref=True,
            usar_recorte=True, rho_recorte=0.7,
        )
        valores.append(perdida)
    return torch.stack(valores).mean()


def entrenar_E_lineal(imagenes_train, imagenes_val, config, ctx):
    """Entrena CoordConv y selecciona el checkpoint por pérdida sin máscaras.

    Devuelve encoder, base y un diccionario con curva, pérdida, época y tiempo.
    La selección nunca emplea las métricas calculadas para observación.
    """
    if not imagenes_train or not imagenes_val:
        raise ValueError("Se requieren imágenes de entrenamiento y validación")
    fijar_semillas(config.get("semilla", 42))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
    h, w = ctx["preprocesadas"][imagenes_train[0]].shape
    base = BaseSuave(h, w, f_max=config.get("f_max", 4), grado_poli=3).to(device)
    encoder = EncoderCoordConv(base.n_coef, h, w, config.get("base_channels", 16)).to(device)

    def preparar(nombres):
        datos = []
        for nombre in nombres:
            imagen = ctx["preprocesadas"][nombre]
            if imagen.shape != (h, w):
                raise ValueError("Todas las imágenes deben tener la misma forma")
            datos.append(normalizar_por_imagen(imagen)[0])
        return torch.as_tensor(np.stack(datos), dtype=torch.float32, device=device).unsqueeze(1)

    train = preparar(imagenes_train)
    val = preparar(imagenes_val)
    batch = min(max(int(config.get("batch_size", 4)), 1), 8)
    pesos = {"esp": 1.0, "signo": 0.8, "signo_neg": 1.0,
             "signo_pos": 0.01, "ref": 0.1}
    optim = torch.optim.Adam(encoder.parameters(), lr=config.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.5)
    mejor_loss = float("inf")
    mejor_estado, mejor_epoca = None, None
    curva = []
    inicio = time.perf_counter()

    for ep in range(int(config.get("epocas", 300))):
        encoder.train()
        perdidas_train = []
        for indices in torch.randperm(len(train), device=device).split(batch):
            aumentadas = []
            for imagen in train[indices]:
                # Un giro de 90 grados intercambia dimensiones si la imagen no es cuadrada.
                giros = int(torch.randint(4 if h == w else 2, ()).item())
                imagen = torch.rot90(imagen, giros if h == w else giros * 2, (-2, -1))
                if torch.rand(()).item() < 0.5:
                    imagen = imagen.flip(-1)
                aumentadas.append(imagen)
            ub = torch.stack(aumentadas)
            optim.zero_grad(set_to_none=True)
            b = base(encoder(ub)).unsqueeze(1)
            loss = _perdida_por_imagen(ub, b, pesos)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Pérdida no finita en época {ep + 1}")
            loss.backward()
            optim.step()
            perdidas_train.append((loss.item(), len(indices)))
        scheduler.step()

        encoder.eval()
        perdidas_val = []
        with torch.no_grad():
            for i in range(0, len(val), batch):
                ub = val[i:i + batch]
                b = base(encoder(ub)).unsqueeze(1)
                perdidas_val.append((_perdida_por_imagen(ub, b, pesos).item(), len(ub)))
        lv = sum(v * n for v, n in perdidas_val) / len(val)
        if lv < mejor_loss:
            mejor_loss = lv
            mejor_epoca = ep + 1
            mejor_estado = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        fila = {
            "epoca": ep + 1,
            "loss_train": sum(v * n for v, n in perdidas_train) / len(train),
            "loss_val": lv,
            "M1_val": float("nan"),
            "dh_val": float("nan"),
        }
        if ep == 0 or (ep + 1) % 10 == 0 or ep + 1 == int(config.get("epocas", 300)):
            medidas = []
            for nombre in imagenes_val:
                imagen = ctx["preprocesadas"][nombre]
                estimado = inferir_E_lineal(encoder, base, imagen)
                medidas.append(metricas_archivo(imagen - estimado, nombre, ctx))
            fila["M1_val"] = float(np.median([m["RMS lento (niveles eq.)"] for m in medidas]))
            fila["dh_val"] = float(np.median([m["|Δh| mediana (%)"] for m in medidas]))
        curva.append(fila)

    if mejor_estado is None:
        raise ValueError("Debe entrenarse durante al menos una época")
    encoder.load_state_dict(mejor_estado)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return encoder, base, {
        "curva": pd.DataFrame(curva),
        "loss_val": mejor_loss,
        "mejor_epoca": mejor_epoca,
        "tiempo_entrenamiento": time.perf_counter() - inicio,
    }


def inferir_E_lineal(encoder, base, imagen):
    """Estima el fondo mediante una pasada, sin ajuste por imagen."""
    un, media, std = normalizar_por_imagen(imagen)
    device = next(encoder.parameters()).device
    u = torch.as_tensor(un, dtype=torch.float32, device=device)[None, None]
    encoder.eval()
    with torch.no_grad():
        fondo = base(encoder(u))[0].cpu().numpy()
    return fondo * std + media


def main():
    parser = argparse.ArgumentParser(description="Control E-lineal con CoordConv")
    parser.add_argument("--epocas", type=int, default=300)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--f-max", type=int, default=4)
    parser.add_argument("--canales", type=int, choices=(16, 32), default=16)
    parser.add_argument("--lote", type=int, default=4)
    args = parser.parse_args()
    config = {
        "epocas": args.epocas, "semilla": args.semilla, "f_max": args.f_max,
        "base_channels": args.canales, "batch_size": args.lote,
    }
    datos = cargar()
    encoder, base, info = entrenar_E_lineal(
        datos["entrenamiento"], datos["validacion"], config, datos["ctx"]
    )
    directorio = Path(__file__).resolve().parent
    (directorio / "pesos").mkdir(exist_ok=True)
    (directorio / "curvas").mkdir(exist_ok=True)
    nombre = f"E_lineal_f{args.f_max}_c{args.canales}_s{args.semilla}"
    torch.save({
        "config": config,
        "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
        "base": {k: v.detach().cpu() for k, v in base.state_dict().items()},
        "mejor_epoca": info["mejor_epoca"],
    }, directorio / "pesos" / f"{nombre}.pth")
    info["curva"].to_csv(directorio / "curvas" / f"{nombre}.csv", index=False)

    metricas = []
    tiempos = []
    for archivo in datos["validacion"]:
        imagen = datos["ctx"]["preprocesadas"][archivo]
        t0 = time.perf_counter()
        fondo = inferir_E_lineal(encoder, base, imagen)
        tiempos.append(time.perf_counter() - t0)
        metricas.append(datos["metricas_archivo"](imagen - fondo, archivo))
    salida = {
        "config": config, "loss_val": info["loss_val"],
        "mejor_epoca": info["mejor_epoca"],
        "tiempo_entrenamiento": info["tiempo_entrenamiento"],
        "tiempo_inferencia_medio": float(np.mean(tiempos)),
        "parametros": sum(p.numel() for p in encoder.parameters() if p.requires_grad),
        "M1_mediana": float(np.median([m["RMS lento (niveles eq.)"] for m in metricas])),
        "dh_mediana": float(np.median([m["|Δh| mediana (%)"] for m in metricas])),
    }
    print(json.dumps(salida, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
