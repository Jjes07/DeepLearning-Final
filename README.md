# EritroFase

**Corrección del fondo en imágenes de fase de eritrocitos mediante redes neuronales informadas por la física**

Proyecto final · SI3014 Redes Neuronales y Aprendizaje Profundo · Universidad EAFIT · Grupo 1
Integrantes: Juan José Escobar, Emmanuel Hernández, Mateo Lopera.

La entrega principal está en [EritroFase.ipynb](EritroFase.ipynb). Su modelo C
usa un codificador convolucional y una SIREN modulada con FiLM. La sección 11
incluye una comparación con el [experimento complementario de Fourier](experiments/EritroFase-Fourier.ipynb)
y con el [experimento complementario de Emmanuel](experiments/EritroFase-Emmanuel.ipynb).

Entregables: [informe](docs/final-report/EritroFase_Informe.pdf) y
[presentación de la sustentación](docs/final-report/EritroFase_Sustentacion.pptx).
Ambos se actualizaron el 12/11/2026 para incluir el
[experimento de control](#experimento-de-control-posterior-a-la-entrega) y precisar las
afirmaciones sobre velocidad y suavidad; las versiones entregadas el 11/11/2026 están en
el historial de git (commit `06e7fd8`). La fuente del informe está en
[`docs/final-report/src/`](docs/final-report/src/EritroFase_Informe.html).

## Estructura

```
DeepLearning-Final/
├── EritroFase.ipynb          # Notebook principal y autocontenido: todo el código está aquí; se entrega con sus salidas
├── README.md
├── requirements.txt
├── experiments/
│   ├── EritroFase-Fourier.ipynb
│   ├── EritroFase-Control.ipynb   # Control posterior a la entrega: la misma pérdida física sobre una base polinomial
│   ├── EritroFase-Emmanuel.ipynb  # Versión paralela (PR #13); solo lectura, no se re-ejecuta desde aquí
│   └── emmanuel/              # Scripts, CSV, curvas e informe de sus experimentos (sección 11.3)
├── data/
│   ├── raw/RBCs_Holograms/   # Dataset de Castañeda et al. (2024). NO se versiona
│   │   ├── Holograms/{Training,Validation}/
│   │   └── Phase/{Training,Validation}/
│   └── selection/            # Lista de imágenes elegidas, partición y máscaras de referencia
├── docs/
│   ├── final-report/         # Entregables: informe (PDF, 3 páginas) y presentación de la sustentación (PPTX)
│   │   └── src/              # Fuente HTML del informe y sus figuras (el PDF se imprime con Chrome)
│   └── references/           # Artículos de referencia (PDF)
├── reference_code/INR/       # Copia sin modificar del código guía (el notebook la incorpora en la sección 4.4)
└── results/
    ├── fourier/              # Tablas, figuras y configuración del experimento de Fourier
    ├── control/              # Tablas y figuras del experimento de control
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

El experimento de Emmanuel ajusta por imagen la misma SIREN que la variante A de la
entrega, con otra pérdida física (recorte por fracción). Su notebook se conserva tal
como se ejecutó, con notas de integración agregadas, y no debe volver a ejecutarse
desde `experiments/` porque escribiría en `results/` con los nombres de la entrega.
La comparación está en la sección 11.3 del notebook principal.

## Experimento de control (posterior a la entrega)

[`experiments/EritroFase-Control.ipynb`](experiments/EritroFase-Control.ipynb) ajusta un
polinomio de grado 1 a 12 con la misma pérdida física del modelo final (Definiciones 17
y 18, hiperparámetros de `results/runs/E4_C_w1_ls0.001/config.json`) y lo evalúa con las
métricas de la entrega sobre las mismas imágenes. Separa dos cosas que la entrega cambia
a la vez: la familia de funciones del fondo y la pérdida.

- Es posterior a la entrega del 11/11/2026 y no es una evaluación ciega: prueba y
  generalización ya se habían examinado. Reporta todos los grados.
- Solo necesita los 30 recortes de `data/selection/`; no usa PyTorch ni GPU. Lee las
  funciones de preprocesamiento, máscaras y métricas del notebook principal y comprueba
  que reproduce sus tablas publicadas.
- Con M1 y M2, la curva del control pasa por los puntos de A y queda por debajo de los
  de C. A cambio, absorbe células cortadas por el borde del recorte, que M2 no evalúa
  (68 de 104). Esa medida no está calculada para A, C ni el Zernike.

Tablas y figuras en [`results/control/`](results/control/). Para repetirlo desde la raíz
del proyecto (unos minutos en CPU):

```sh
python -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1800 experiments/EritroFase-Control.ipynb
```
