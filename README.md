# EritroFase

**Corrección del fondo en imágenes de fase de eritrocitos mediante redes neuronales informadas por la física**

Proyecto final · SI3014 Redes Neuronales y Aprendizaje Profundo · Universidad EAFIT · Grupo 1
Integrantes: Juan José Escobar, Emmanuel Hernández, Mateo Lopera.

El [notebook ejecutado](EritroFase.ipynb) contiene la arquitectura, el entrenamiento y
los resultados. La red condicional de Fourier cumplió el criterio prefijado en
prueba por **0,019 niveles equivalentes** de RMS lento frente a Zernike, con un cambio
mediano absoluto de altura celular del **2,91 %**. En el día reservado para
generalización, Zernike dejó ligeramente menos fondo residual.

## Estructura

```
DeepLearning-Final/
├── EritroFase.ipynb          # Notebook principal y autocontenido: todo el código está aquí; se entrega con sus salidas
├── README.md
├── requirements.txt
├── data/
│   ├── raw/RBCs_Holograms/   # Dataset de Castañeda et al. (2024). NO se versiona
│   │   ├── Holograms/{Training,Validation}/
│   │   └── Phase/{Training,Validation}/
│   └── selection/            # Lista de imágenes elegidas, partición y máscaras de referencia
├── docs/references/          # Artículos de referencia (PDF)
├── reference_code/INR/       # Copia sin modificar del código guía (el notebook la incorpora en la sección 4.4)
└── results/
    ├── figures/              # Figuras por etapa (E1_..., E2_..., ...)
    ├── tables/               # Tablas de métricas (CSV)
    └── runs/                 # Una carpeta por corrida: config, curvas, métricas
```

## Datos

El repositorio contiene los 30 recortes congelados de `data/selection/`. Con ellos se
pueden reproducir el preprocesamiento, las líneas base, el entrenamiento y la evaluación.
La partición es de 12 recortes para entrenamiento, 5 para validación, 5 para prueba
y 8 de otro día para generalización.

Para repetir también la auditoría del dataset completo, descargue el ZIP de Castañeda,
Trujillo y Doblas (2024). Descomprima `Holograms` y `Phase` dentro de
`data/raw/RBCs_Holograms/`. Cada una debe contener `Training` y `Validation`.
[Artículo de datos](https://doi.org/10.1016/j.dib.2024.110424) ·
[Descarga directa del ZIP en OSF (2,69 GB)](https://osf.io/download/cp2kn/?view_only=e8d752fda65e456f896b72a91aa8db51).

## Reproducir el notebook

Desde la raíz del repositorio, cree un entorno con Python 3.12 e instale PyTorch
para su hardware. Estos comandos corresponden al entorno CUDA 13.0 usado para la
corrida final. Para otros dispositivos, consulte el
[selector de instalación de PyTorch](https://pytorch.org/get-started/locally/).

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 EritroFase.ipynb
```

El notebook detecta los datos completos si están presentes. Con los recortes versionados,
usa la evidencia archivada para el inventario del dataset y vuelve a ejecutar los
experimentos. En PowerShell, `$env:ERITROFASE_DATOS = 'completo'` exige la presencia del
dataset y `$env:ERITROFASE_DATOS = 'seleccion'` fuerza el modo de 30 recortes.
`$env:ERITROFASE_RECALCULAR = '0'` habilita la reutilización de la caché local.
Por defecto se recalculan los pasos disponibles.

La configuración elegida, las curvas y los resultados por imagen están en
[`results/runs/eritrofase_final/`](results/runs/eritrofase_final/). El criterio de éxito,
fijado en la sección 6.5, exige una mediana de cambio absoluto de altura celular de
hasta 5 % y compara el fondo lento residual con la mejor línea base que respeta ese límite.
