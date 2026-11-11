"""Medición comparativa de A y AE sobre validación, sin procesos paralelos."""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experimentos'))
from data import fijar_semillas
from modelos import SirenBackgroundNetwork, UNetPesos, build_coordinate_grid

SEMILLA = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.75, 0)


def sincronizar():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def actividad_gpu():
    if DEVICE.type != 'cuda':
        return None
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=5, check=True)
        return int(p.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def ajustar(imagen, unet=None, iteraciones=1000):
    fijar_semillas(SEMILLA)
    h, w = imagen.shape
    coords = build_coordinate_grid(h, w, device=DEVICE)
    u = torch.tensor(imagen, dtype=torch.float32, device=DEVICE).flatten()
    modelo = SirenBackgroundNetwork(hidden_dim=128, n_layers=2, omega_0=1.0).to(DEVICE)
    optim = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    t_mapa = 0.0
    if unet is not None:
        norm = (imagen - imagen.mean()) / (imagen.std() + 1e-8)
        entrada = torch.tensor(norm, dtype=torch.float32, device=DEVICE)[None, None]
        sincronizar(); inicio = time.perf_counter()
        with torch.no_grad():
            pesos = unet(entrada).flatten().pow(0.25)
        sincronizar(); t_mapa = time.perf_counter() - inicio
    sincronizar(); inicio = time.perf_counter()
    for _ in range(iteraciones):
        optim.zero_grad()
        r = u - modelo(coords)
        if unet is None:
            n = max(1, int(0.7 * r.numel()))
            ajuste = torch.sort(r.abs()).values[:n].mean()
        else:
            ajuste = (pesos * torch.sqrt(r.square() + 1e-6)).sum() / pesos.sum().clamp_min(1e-8)
        signo = torch.relu(-r).square().mean() + 0.01 * torch.relu(r).mean()
        perdida = ajuste + 0.8 * signo + 0.1 * r.median().square()
        perdida.backward()
        optim.step()
    sincronizar(); t_ajuste = time.perf_counter() - inicio
    sincronizar(); inicio = time.perf_counter()
    with torch.no_grad():
        fondo = modelo(coords).reshape(h, w).cpu().numpy()
    sincronizar(); t_salida = time.perf_counter() - inicio
    del modelo, optim, coords, u, r, perdida, fondo
    if unet is not None:
        del pesos, entrada
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return t_mapa, t_ajuste, t_salida


def main():
    destino = ROOT / 'results' / 'runs' / 'comparacion_tiempos'
    destino.mkdir(parents=True, exist_ok=True)
    csv_val = pd.read_csv(ROOT / 'experimentos' / 'baseline_validacion.csv')
    archivos = list(dict.fromkeys(csv_val['archivo']))
    assert len(archivos) == 5
    with np.load(ROOT / 'experimentos' / 'cache_dev.npz') as datos:
        nombres = datos['archivos'].tolist()
        imagenes = {f: datos['preprocesadas'][nombres.index(f)].copy() for f in archivos}
        assert all(datos['conjuntos'][nombres.index(f)] == 'validacion' for f in archivos)
    ruta = ROOT / 'experimentos' / 'pesos' / f'E_f6_a0.5_c16_s{SEMILLA}.pth'
    ck = torch.load(ruta, map_location='cpu', weights_only=False)
    assert ck['config']['semilla'] == SEMILLA and ck['config']['f_max'] == 6
    unet = UNetPesos(ck['config']['base_channels'])
    unet.load_state_dict({k.removeprefix('unet.'): v for k, v in ck['estado'].items() if k.startswith('unet.')})
    unet = unet.to(DEVICE).eval()
    parametros_unet = sum(p.numel() for p in unet.parameters())
    parametros_siren = sum(p.numel() for p in SirenBackgroundNetwork().parameters())
    assert parametros_siren == 33537
    muestras = {'antes_calentamiento': actividad_gpu()}
    ajustar(imagenes[archivos[0]], iteraciones=5)
    ajustar(imagenes[archivos[0]], unet, iteraciones=5)
    muestras['antes_mediciones'] = actividad_gpu()
    filas = []
    for i, f in enumerate(archivos):
        for metodo in (['A', 'AE'] if i % 2 == 0 else ['AE', 'A']):
            mapa, ajuste, salida = ajustar(imagenes[f], unet if metodo == 'AE' else None)
            filas.append({'archivo': f, 'método': metodo, 'mapa E (s/imagen)': mapa,
                          'ajuste (s/imagen)': ajuste, 'inferencia (s/imagen)': salida,
                          'total (s/imagen)': mapa + ajuste + salida})
            print(f'{metodo} · {f}: {ajuste:.2f} s ajuste', flush=True)
    muestras['despues_mediciones'] = actividad_gpu()
    detalle = pd.DataFrame(filas)
    columnas = ['mapa E (s/imagen)', 'ajuste (s/imagen)', 'inferencia (s/imagen)', 'total (s/imagen)']
    resumen = detalle.groupby('método')[columnas].median()
    resumen['parámetros entrenables'] = [parametros_siren, parametros_siren + parametros_unet]
    detalle.to_csv(destino / 'por_imagen.csv', index=False)
    resumen.to_csv(destino / 'resumen.csv')
    meta = {'semilla': SEMILLA, 'dispositivo': str(DEVICE),
            'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None,
            'actividad_gpu_porcentaje': muestras, 'parametros_unet': parametros_unet,
            'iteraciones': 1000, 'archivo_checkpoint': ruta.name}
    (destino / 'config.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(resumen.to_string(), flush=True)
    print('Actividad GPU puntual (%):', muestras, flush=True)
    del unet, ck
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
