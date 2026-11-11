# EritroFase

**Corrección del fondo en imágenes de fase de eritrocitos mediante redes neuronales informadas por la física**

Proyecto final · SI3014 Redes Neuronales y Aprendizaje Profundo · Universidad EAFIT · Grupo 1
Integrantes: Juan José Escobar, Emmanuel Hernández, Mateo Lopera.

La entrega principal está en [EritroFase.ipynb](EritroFase.ipynb). Su modelo C
usa un codificador convolucional y una SIREN modulada con FiLM. La sección 11
incluye una comparación con el [experimento complementario de Fourier](experiments/EritroFase-Fourier.ipynb).

## Estructura

```
DeepLearning-Final/
├── EritroFase.ipynb          # Notebook principal y autocontenido: todo el código está aquí; se entrega con sus salidas
├── README.md
├── requirements.txt
├── experiments/EritroFase-Fourier.ipynb
├── data/
│   ├── raw/RBCs_Holograms/   # Dataset de Castañeda et al. (2024). NO se versiona
│   │   ├── Holograms/{Training,Validation}/
│   │   └── Phase/{Training,Validation}/
│   └── selection/            # Lista de imágenes elegidas, partición y máscaras de referencia
├── docs/references/          # Artículos de referencia (PDF)
├── reference_code/INR/       # Copia sin modificar del código guía (el notebook la incorpora en la sección 4.4)
└── results/
    ├── fourier/              # Tablas, figuras y configuración del experimento de Fourier
    ├── figures/              # Figuras por etapa (E1_..., E2_..., ...)
    ├── tables/               # Tablas de métricas (CSV)
    └── runs/                 # Una carpeta por corrida: config, curvas, métricas
```

## Datos

El notebook principal necesita el dataset de Castañeda, Trujillo y Doblas (2024)
para repetir la auditoría y ampliar los recortes de entrenamiento. Descomprima
`Holograms` y `Phase` dentro de `data/raw/RBCs_Holograms/`. Cada carpeta debe
contener `Training` y `Validation`. [Artículo de datos](https://doi.org/10.1016/j.dib.2024.110424) ·
[Descarga del ZIP en OSF (2,69 GB)](https://osf.io/download/cp2kn/?view_only=e8d752fda65e456f896b72a91aa8db51).

El experimento de Fourier usa los 30 recortes versionados de `data/selection/`.

## Entorno

1. Cree un entorno virtual con Python 3.12 o 3.13.
2. Instale [PyTorch para su hardware](https://pytorch.org/get-started/locally/). Para
   la corrida de Fourier en Windows con RTX 5090 se verificó PyTorch 2.14.1 con CUDA 13.0:
   `python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130`.
3. Instale las demás dependencias con `python -m pip install -r requirements.txt`.
4. Abra el notebook desde Jupyter y ejecute las celdas en orden. Para repetir Fourier
   desde la terminal en la raíz del proyecto, use:

   ```sh
   python -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 experiments/EritroFase-Fourier.ipynb
   ```

## Comparación de modelos

La variante de Fourier aprende una corrección suave sobre Zernike. Su entrenamiento
usa 12 recortes. El modelo C de la entrega aprende el fondo con 120 recortes de las
mismas 12 adquisiciones. Ambos se evaluaron sobre los mismos cinco recortes de prueba
y ocho del día reservado. La comparación es descriptiva. Los resultados de Fourier
se guardan en [`results/fourier/`](results/fourier/) y la selección del modelo C
está documentada en la sección 8.7 del [notebook principal](EritroFase.ipynb).
