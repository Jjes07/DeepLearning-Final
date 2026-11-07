# EritroFase

**Corrección del fondo en imágenes de fase de eritrocitos mediante redes neuronales informadas por la física**

Proyecto final · SI3014 Redes Neuronales y Aprendizaje Profundo · Universidad EAFIT · Grupo 1
Integrantes: Juan José Escobar, Emmanuel Hernández, Mateo Lopera.

## Estructura

```
DeepLearning-Final/
├── EritroFase.ipynb          # Notebook principal: se entrega con todas sus salidas
├── README.md
├── requirements.txt
├── configs/                  # Configuraciones (YAML) de cada experimento
├── data/
│   ├── raw/RBCs_Holograms/   # Dataset de Castañeda et al. (2024). NO se versiona
│   │   ├── Holograms/{Training,Validation}/
│   │   └── Phase/{Training,Validation}/
│   └── selection/            # Lista de imágenes elegidas, partición y máscaras de referencia
├── src/eritrofase/           # Código reutilizable (datos, modelos, pérdidas, métricas, figuras)
├── reference_code/INR/       # Copia sin modificar del código guía del profesor
└── results/
    ├── figures/              # Figuras por etapa (E1_..., E2_..., ...)
    ├── tables/               # Tablas de métricas (CSV)
    └── runs/                 # Una carpeta por corrida: config, curvas, métricas
```

## Datos

Dataset: R. Castañeda, C. Trujillo y A. Doblas (2024), *A human erythrocytes hologram dataset for
learning-based model training*, Data in Brief 54, 110424. doi:10.1016/j.dib.2024.110424.
Descarga: https://osf.io/8p7ba/ (DOI 10.17605/OSF.IO/8P7BA).

Descomprimir de modo que queden `data/raw/RBCs_Holograms/Phase/Training` y
`data/raw/RBCs_Holograms/Phase/Validation`. El nombre original de la carpeta tiene un espacio
(`RBCs Holograms`): renombrarla a `RBCs_Holograms`.

## Entorno

1. Crear un entorno virtual con Python 3.11–3.13.
2. Instalar PyTorch según el hardware (ver `requirements.txt`).
3. `pip install -r requirements.txt`
4. Ejecutar `EritroFase.ipynb` en orden, de arriba abajo.
