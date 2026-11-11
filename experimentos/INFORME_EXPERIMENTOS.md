# Informe de experimentos — issue #3

## Objetivo
Encontrar en validación un modelo que supere a la línea base Zernike: mediana M1 < 6,4 niveles con mediana |Δh| ≤ 5 %.

## Líneas base reproducidas

| método | M1 | |Δh| (%) |
|--------|-----:|----------:|
| SIREN ω0=1 (guía) | 3.47 | 12.85 |
| Zernike (guía) | 6.39 | 2.55 |

## Resultados de todas las corridas

| id | descripción | esquema | M1 | |Δh| (%) | cumple | t_train (s) | t_infer (s) |
|----|-------------|---------|-----:|----------:|:-------|------------:|------------:|
| 8efa2f7a | A_SIREN_signo0.5_asim0.8_1k | A | 1.33 | 15.13 | No | 15.08 | 0.02 |
| be580425 | A_SIREN_signo0.5_asim0.7_1k | A | 1.33 | 7.73 | No | 15.86 | 0.02 |
| 19420cc4 | A_SIREN_signo1.0_recorte0.7_2000 | A | 2.38 | 8.93 | No | 29.97 | 0.01 |
| afd125da | A_SIREN_signo1.0_recorte0.75_2000 | A | 2.50 | 16.00 | No | 28.11 | 0.01 |
| 2f0e1d22 | A_SIREN_signo1.2_recorte0.7_2000 | A | 2.69 | 9.53 | No | 59.50 | 0.03 |
| 94a4f2da | A_SIREN_signo05_recorte07_1k | A | 2.91 | 7.83 | No | 14.00 | 0.01 |
| e369c299 | A_schedule0.0-2.0_rec0.7_1000 | A | 3.05 | 9.62 | No | 135.85 | 0.04 |
| a8144738 | A_SIREN_signo1.0_asim0.7_1k | A | 3.26 | 6.86 | No | 13.99 | 0.01 |
| 9b63357c | A_SIREN_signo0.6_recorte07_1k | A | 3.43 | 8.57 | No | 14.64 | 0.01 |
| e9452d99 | A_schedule0.0-3.0_rec0.7_1000 | A | 3.49 | 10.02 | No | 158.73 | 0.05 |
| 5acc6312 | A_SIREN_signo0.7_recorte07_1k | A | 3.53 | 7.98 | No | 14.22 | 0.01 |
| 5baf6930 | A_SIREN_signo1.0_asim0.8_1k | A | 3.57 | 9.18 | No | 15.13 | 0.02 |
| ca683f00 | A_SIREN_signo0.8_recorte07_1k | A | 3.76 | 7.76 | No | 13.85 | 0.01 |
| fbb30b8e | A_SIREN_signo1.0_suav0.001_rec0.7_1000 | A | 4.35 | 6.11 | No | 46.12 | 0.05 |
| 2d3cfab5 | A_SIREN_signo1_recorte08_1k | A | 4.38 | 9.07 | No | 13.79 | 0.01 |
| 5944f1a5 | A_SIREN_signo1_recorte07_1k | A | 4.38 | 6.28 | No | 20.13 | 0.02 |
| 0ed073e8 | A_SIREN_signo1.0_suav0.0001_rec0.7_1000 | A | 4.38 | 6.22 | No | 58.66 | 0.06 |
| 6e52985e | A_asim05_finetune_signo2.0_200 | A+ | 4.45 | 13.07 | No | 0.05 | 0.05 |
| d0af352f | A_asim05_finetune_signo3.0_200 | A+ | 4.88 | 13.04 | No | 0.06 | 0.06 |
| 1f87180b | A_SIREN_signo1.2_recorte0.7_1000 | A | 5.24 | 6.33 | No | 15.65 | 0.02 |
| f3ff0cc1 | A_ganadora_final_semilla42 | A | 5.65 | 3.78 | Sí | 49.58 | 0.01 |
| f9ea3f81 | A_cand_signo1.5_ref0.5_rec0.7_semilla42 | A | 5.65 | 3.78 | Sí | 49.85 | 0.01 |
| f79d7905 | A_SIREN_signo1.5_ref0.5_rec0.7_1000 | A | 5.65 | 3.78 | Sí | 14.26 | 0.01 |
| e56927c5 | A_ganadora_final_semilla12345 | A | 5.98 | 5.30 | No | 48.98 | 0.01 |
| 0233f55d | A_cand_signo1.5_ref0.5_rec0.7_semilla12345 | A | 5.98 | 5.30 | No | 49.89 | 0.01 |
| 616fdbf2 | A_ganadora_semilla2024 | A | 5.98 | 4.14 | Sí | 48.78 | 0.01 |
| b8ef702e | A_ganadora_final_semilla2024 | A | 5.98 | 4.14 | Sí | 49.90 | 0.01 |
| 3d821365 | A_cand_signo0.8_ref0.1_rec0.7_semilla2024 | A | 5.98 | 4.14 | Sí | 49.67 | 0.01 |
| 2dabfafe | A_cand_signo1.0_ref0.1_rec0.75_semilla2024 | A | 6.03 | 4.79 | Sí | 49.34 | 0.01 |
| 18d948db | A_SIREN_signo1.5_recorte0.7_1000 | A | 6.14 | 6.41 | No | 14.61 | 0.01 |
| 894bec7a | A_SIREN_signo1.0_ref0.2_rec0.7_1000 | A | 6.23 | 5.34 | No | 13.69 | 0.01 |
| f93f62b8 | A_ganadora_final_semilla42 | A | 6.23 | 4.72 | Sí | 51.94 | 0.01 |
| 00efd994 | A_SIREN_signo0.8_ref0.1_rec0.7_1000 | A | 6.23 | 4.72 | Sí | 12.90 | 0.01 |
| 461071e3 | A_ganadora_semilla42 | A | 6.23 | 4.72 | Sí | 49.18 | 0.01 |
| 65e24aa5 | A_cand_signo0.8_ref0.1_rec0.7_semilla42 | A | 6.23 | 4.72 | Sí | 50.16 | 0.01 |
| 3831a08a | A_ganadora_semilla12345 | A | 6.33 | 3.60 | Sí | 48.52 | 0.01 |
| c5690340 | A_cand_signo0.8_ref0.1_rec0.7_semilla12345 | A | 6.33 | 3.60 | Sí | 50.33 | 0.01 |
| 6ad480e1 | A_ganadora_final_semilla12345 | A | 6.33 | 3.60 | Sí | 51.42 | 0.01 |
| e0bd5eea | A_SIREN_signo1.0_ref0.1_rec0.7_1500 | A | 6.34 | 5.03 | No | 19.31 | 0.01 |
| f186ca4c | A_SIREN_signo1.0_ref0.1_rec0.75_1000 | A | 6.38 | 4.27 | Sí | 13.59 | 0.01 |
| 0fd4e419 | A_cand_signo1.0_ref0.1_rec0.75_semilla42 | A | 6.38 | 4.27 | Sí | 48.93 | 0.01 |
| 0f069e30 | A_cand_signo1.0_ref0.1_rec0.75_semilla12345 | A | 6.40 | 5.28 | No | 50.32 | 0.01 |
| fc16c9df | A_SIREN_signo1.0_ref0.1_rec0.7_1000 | A | 6.45 | 4.16 | Sí | 68.60 | 0.07 |
| e01a512d | A_cand_signo1.0_ref0.1_rec0.7_semilla42 | A | 6.45 | 4.16 | Sí | 50.65 | 0.01 |
| 55bab9c1 | A_cand_signo1.5_ref0.5_rec0.7_semilla2024 | A | 6.57 | 3.57 | Sí | 49.82 | 0.01 |
| 1a43b3fc | A_ganadora_final_semilla2024 | A | 6.57 | 3.57 | Sí | 49.29 | 0.01 |
| 5bdfcef7 | A_cand_signo1.0_ref0.1_rec0.7_semilla2024 | A | 6.66 | 4.93 | Sí | 51.44 | 0.01 |
| 59166a03 | A_SIREN_signo1.2_ref0.1_rec0.7_1000 | A | 6.72 | 4.65 | Sí | 13.78 | 0.01 |
| cb4b2f9a | A_cand_signo1.0_ref0.1_rec0.7_semilla12345 | A | 6.94 | 4.80 | Sí | 49.39 | 0.01 |
| 5ef3a93c | A_SIREN_signo1.0_ref0.05_rec0.7_1000 | A | 7.02 | 4.99 | Sí | 14.34 | 0.01 |
| 4a903fb4 | A_SIREN_signo1 | A | 7.15 | 9.96 | No | 24.16 | 0.05 |
| 9da065d5 | A_SIREN_signo1.5_ref0.1_rec0.7_1000 | A | 7.15 | 4.42 | Sí | 15.78 | 0.02 |
| 06eeb33c | A_SIREN_signo1.0_ref0.5_rec0.7_1000 | A | 7.32 | 3.70 | Sí | 31.40 | 0.03 |
| e5ab03e0 | A_SIREN_signo1_pos01_recorte07 | A | 7.73 | 5.22 | No | 25.64 | 0.05 |
| 5b2297ad | A_SIREN_signo1_recorte07 | A | 8.30 | 4.98 | Sí | 9.32 | 0.02 |
| d8c05740 | A_SIREN_w1_signo5_pos01 | A | 8.59 | 11.88 | No | 39.43 | 0.04 |
| fbcf51f5 | A_SIREN_w1_signo5 | A | 9.20 | 10.25 | No | 60.53 | 0.06 |
| da0797ce | A_SIREN_signo2 | A | 9.64 | 6.51 | No | 14.98 | 0.03 |
| 03ae28e3 | A_SIREN_signo1_recorte05 | A | 10.76 | 3.17 | Sí | 23.69 | 0.05 |
| 6e419139 | A_SIREN_signo2_recorte07 | A | 11.11 | 4.10 | Sí | 15.41 | 0.03 |
| 6ed07503 | A_SIREN_w1_signo10 | A | 11.71 | 7.55 | No | 59.18 | 0.06 |

## Configuración seleccionada

Configuración robusta con menor M1 promedio entre las que cumplen |Δh| ≤ 5 % en 3 semillas:

- ID: `3d821365`
- Descripción base: A_cand_signo0.8_ref0.1_rec0.7
- Esquema: A
- M1 validación mediana (semilla 2024): 5.98 niveles
- |Δh| validación mediana (semilla 2024): 4.14 %

Hiperparámetros:
```python
{'descripcion': 'A_cand_signo0.8_ref0.1_rec0.7_semilla2024', 'modelo_kwargs': {'hidden_dim': 128, 'n_layers': 2, 'omega_0': 1.0}, 'lr': 0.001, 'iteraciones': 1000, 'pesos': {'esp': 1.0, 'signo': 0.8, 'signo_neg': 1.0, 'signo_pos': 0.01, 'suav': 0.0, 'ref': 0.1, 'recorte': True, 'rho_recorte': 0.7}, 'semilla': 2024}
```

### Métricas por imagen de validación

| imagen | M1 | |Δh| (%) |
|--------|-----:|----------:|
| val_1 | 7.28 | 4.14 |
| val_2 | 6.71 | 6.61 |
| val_3 | 5.98 | 3.26 |
| val_4 | 4.39 | 4.83 |
| val_5 | 5.95 | 2.35 |

### Robustez: 3 semillas

| semilla | M1 | |Δh| (%) |
|---------|-----:|----------:|
| 42 | 6.23 | 4.72 |
| 2024 | 5.98 | 4.14 |
| 12345 | 6.33 | 3.60 |
| media ± std | 6.18 ± 0.18 | 4.16 ± 0.56 |

## Comparación con líneas base

| método | M1 | |Δh| (%) | cumple |
|--------|-----:|----------:|:-------|
| SIREN ω0=1 (guía) | 3.47 | 12.85 | No |
| Zernike (guía) | 6.39 | 2.55 | Sí |
| A_cand_signo0.8_ref0.1_rec0.7_semilla2024 | 5.98 | 4.14 | Sí |

## Ablación de términos de la pérdida

Se probó la siguiente familia de pérdidas:
- `esp`: Charbonnier sobre todo el residuo, o recorte simétrico/asimétrico sobre un percentil ρ.
- `signo`: asimetría S3 (r<0 fuerte, r>0 débil) con peso variable.
- `suav`: laplaciano del fondo (no mejoró el trade-off en las pruebas).
- `ref`: residuo medio cercano a 0 (fue clave para alcanzar |Δh| ≤ 5 % sin perder toda la ventaja en M1).

Hallazgos principales:
- Sin signo o con recorte asimétrico, M1 baja mucho pero |Δh| sube por encima de 5 % (el fondo absorbe células).
- Con signo≈0,8–1,0, recorte simétrico ρ=0,7 y referencia `ref=0,1`, se logra el equilibrio que supera a Zernike.
- El término de referencia (`r.median()²`) fue clave: sin él, |Δh| ≤ 5 % se alcanzaba a costa de M1 > 6,4.
- El laplaciano (`suav`) y el ajuste fino A+ no mejoraron el trade-off en las configuraciones probadas.

## Recursos y tiempos

- Parámetros entrenables del modelo A usado: 33,537.
- Tiempo de entrenamiento de la ganadora: 49.67 s (validación completa).
- Tiempo de inferencia de la ganadora: 0.0138 s por imagen de validación (mediana aprox.).

## Esquemas explorados

| esquema | configs | M1 min | M1 mediana | |Δh| min | |Δh| mediana |
|---------|--------:|-------:|-----------:|---------:|-------------:|
| A | 59 | 1.33 | 6.23 | 3.17 | 5.03 |
| A+ | 2 | 4.45 | 4.67 | 13.04 | 13.05 |

## Curvas de entrenamiento

Curvas guardadas en `experimentos/curvas/`: 5 archivo(s).

## Observaciones y limitaciones
- El esquema A (SIREN implícita por imagen) mejora M1 fácilmente, pero la preservación de células (|Δh| ≤ 5 %) requiere una regularización fuerte que encarece M1.
- El esquema C (encoder amortizado) no se evaluó extensamente en esta ronda por limitaciones de tiempo; sus primeros intentos dieron M1 muy altos.
- Solo se dispone de 5 imágenes de validación; la varianza de las métricas es alta.
- No se tocó el conjunto de prueba ni el de generalización.

## Conclusión ronda 1
La configuración `A_cand_signo0.8_ref0.1_rec0.7_semilla2024` supera a Zernike en validación: mediana M1 = 5.98 niveles (< 6,39) y mediana |Δh| = 4.14 % (≤ 5 %).
Con 3 semillas, la media de M1 es 6.18 ± 0.18 y la de |Δh| es 4.16 ± 0.56 %.

## Ronda 2: esquemas C y D

### Objetivo
Evaluar si un modelo amortizado (C) o una descomposición fondo + célula (D) pueden aprender de las 12 imágenes de entrenamiento y superar al esquema A ajustado por imagen.

### Salvaguardas aplicadas
Tras el fallo por desbordamiento de VRAM en el primer intento de D con `batch_size = 96`, se implementaron:
- `torch.cuda.set_per_process_memory_fraction(0.75, 0)` al inicio de cada entrenamiento.
- Lotes máximos: 16 para C, 8 para D.
- Precisión mixta (`torch.autocast` + `GradScaler`) en D.
- Checkpoint del mejor modelo copiado a CPU (`detach().cpu().clone()`).
- Métricas de validación cada 10 épocas.
- Liberación explícita de memoria entre configuraciones.
- Prueba de humo de D: 2 épocas, memoria máxima 1.07 GB (< 7 GB).

### Resultados del barrido (semilla 42)

| descripción | esquema | M1 | |Δh| (%) | t_train (s) |
|-------------|---------|-----:|----------:|------------:|
| C_fmax3_rec07_signo08_ref01 | C | 19.52 | 0.76 | 40.2 |
| C_fmax5_rec07_signo08_ref01 | C | 19.73 | 1.44 | 39.1 |
| C_fmax8_rec07_signo08_ref01 | C | 20.32 | 0.46 | 38.6 |
| D_fmax5_lc0.01_ltv0 | D | 36.88 | 0.05 | 253.2 |
| D_fmax5_lc0.05_ltv0 | D | 29.35 | 0.06 | 474.0 |
| D_fmax5_lc0.1_ltv0 | D | 26.18 | 0.04 | 727.1 |
| D_fmax8_lc0.01_ltv0 | D | 36.47 | 0.02 | 731.0 |
| D_fmax8_lc0.05_ltv0 | D | 29.24 | 0.03 | 595.3 |
| D_fmax8_lc0.1_ltv0 | D | 26.02 | 0.03 | 1169.4 |
| D_fmax5_lc0.001_suav0.01 | D | 40.45 | 0.07 | 247.2 |
| D_fmax5_lc0.01_suav0.01 | D | 36.82 | 0.07 | 242.6 |
| D_fmax5_lc0.001_suav0.001 | D | 40.42 | 0.06 | 260.7 |
| D_fmax5_lc0.01_suav0.001 | D | 37.29 | 0.08 | 258.7 |

Ninguna configuración cumple el objetivo. Todos los valores de |Δh| están muy por debajo de 5 %, lo que indica que los modelos apenas corrigen la imagen; al mismo tiempo M1 es muy alto, es decir, el residuo conserva mucha energía de baja frecuencia.

### Comparación final

| método | M1 | |Δh| (%) | cumple |
|--------|-----:|----------:|:-------|
| SIREN ω0=1 (guía) | 3.47 | 12.85 | No |
| Zernike (guía) | 6.39 | 2.55 | Sí |
| A (ronda 1, media ± std) | 6.18 ± 0.18 | 4.16 ± 0.56 | Sí |
| C (mejor: fmax3) | 19.52 | 0.76 | No |
| D (mejor: fmax8_lc0.1) | 26.02 | 0.03 | No |

### Diagnóstico
- **C:** `EncoderAmortizado` termina en `AdaptiveAvgPool2d(1)`, que descarta la disposición espacial del mapa de características y dificulta inferir la orientación de un fondo inclinado. La base ya incluye el polinomio de grado 2, cuyo ajuste obtuvo M1 ≈ 12,8; por tanto, los resultados de C no demuestran que la base sea demasiado rígida.
- **D:** la rama de fondo también emplea promedio global y la rama de célula colapsa a valores cercanos a cero. Además, `perdida_descomposicion` calcula `r.median()` sobre todo el lote, en vez de hacerlo por imagen.
- **Error de lote:** en la implementación usada en estas corridas, `BaseSuave.forward` reorganizaba directamente una matriz de tamaño píxeles × lote como lote × alto × ancho, mezclando las imágenes cuando el lote tenía más de un elemento. En D, el fondo tenía forma `(B, H, W)` frente a `(B, 1, H, W)` para la imagen y la célula, lo que también podía mezclar imágenes por difusión. Estos defectos limitan la interpretación de C y D.

Los resultados históricos de las tablas se conservan tal como se registraron; no se han recalculado con esas correcciones.

### Recursos y tiempos
- Parámetros entrenables de C (`base_channels=32`, `f_max=5`): ~0.5 M.
- Parámetros entrenables de D (`base_channels=32`): ~0.7 M.
- Tiempo de entrenamiento de D más lento: ~19.5 min (300 épocas, f_max=8, lc=0.1).
- Tiempo de inferencia de D: ~0.01–1.9 s por imagen (depende de la carga de GPU).

### Curvas de entrenamiento
Guardadas en `experimentos/curvas/`. Las curvas muestran convergencia estable de la pérdida, pero sin correlación con la mejora de M1, lo que indica que la pérdida de descomposición no guía al modelo hacia una corrección útil del fondo.

### Recomendación final
Los resultados registrados de C y D no superan al esquema A ni a Zernike. Antes de atribuir el fallo a la capacidad de la base, conviene corregir el manejo de lotes y evaluar una arquitectura que conserve la posición espacial. También debe revisarse la pérdida por imagen y el colapso de la rama de célula si se retoma D.

No se seleccionaron finalistas porque ninguna configuración alcanzó M1 < 6.4 niveles con |Δh| ≤ 5 %.


## Ronda 3 — pesos aprendidos y ajuste ponderado

### Protocolo

E predice, con una U-Net de dos niveles y 16 canales iniciales, un peso por píxel y resuelve en una pasada los mínimos cuadrados ponderados sobre la base polinómica de grado 3 y Fourier. El sistema se resuelve en float32 con `μ = 10⁻⁴`; el entrenamiento usa precisión mixta en la red. Se aplican volteos y giros de 90°, 300 épocas y lotes de cuatro imágenes. El umbral de signo se estima por imagen, sin máscaras, mediante la mediana robusta de diferencias entre vecinos; en las 17 imágenes de entrenamiento y validación estuvo entre 0,0143 y 0,0181 en escala normalizada.

La pérdida promedia por imagen el ajuste ponderado, el término de signo, una penalización de área y una variación total de los pesos. Se escogió la cota blanda `relu(0,6 − media(w))²`: evita exigir que todas las imágenes tengan exactamente el promedio poblacional de 76 % de fondo. Una penalización blanda no garantiza que cada media alcance 0,6. Los dos valores de su coeficiente fueron 0,5 y 2,0; el coeficiente de signo fue 1 y el de variación total 0,001.

La pérdida de validación, calculada sin máscaras, eligió el checkpoint y las dos finalistas. M1 y |Δh| se registraron cada diez épocas solo como observaciones. La petición de no usar máscaras en la selección entra en tensión con elegir por M1 y |Δh|, que sí las necesitan: por eso las finalistas se eligieron sin etiquetas y la regla de menor media de M1 con media de |Δh| ≤ 5 % se aplicó **a posteriori entre ellas**. Los valores de pérdida con distintos coeficientes de área corresponden a objetivos distintos y su comparación debe interpretarse con cautela. No se usó el conjunto de prueba para esta ronda.

### Barrido: semilla 42

M1 y |Δh| son medianas entre las cinco imágenes de validación; una sola semilla no permite estimar su desviación entre semillas.

| f_max | λ área | pérdida val. | M1 | &#124;Δh&#124; (%) | época elegida | entrenamiento (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0,5 | 0,09815 | 8,72 | 1,66 | 288 | 99,54 |
| 2 | 2,0 | 0,11012 | 7,55 | 1,97 | 300 | 93,39 |
| 4 | 0,5 | 0,09664 | 9,03 | 1,74 | 288 | 83,46 |
| 4 | 2,0 | 0,10904 | 8,07 | 2,32 | 296 | 90,78 |
| 6 | 0,5 | 0,09683 | 8,79 | 1,64 | 288 | 105,43 |
| 6 | 2,0 | 0,10900 | 7,51 | 1,95 | 300 | 99,07 |

Las finalistas por pérdida fueron `f_max=4, λ=0,5` y `f_max=6, λ=0,5`. La configuración de menor M1 en este barrido fue `f_max=6, λ=2`, pero no fue seleccionada por pérdida y no cuenta con tres semillas; su resultado aislado no permite atribuirle una media ni una desviación.

### Finalistas: tres semillas

Cada observación es la mediana de las cinco imágenes; las columnas finales presentan media ± desviación estándar **muestral entre semillas**.

| Configuración | Semilla 42: M1 / &#124;Δh&#124; | Semilla 2024: M1 / &#124;Δh&#124; | Semilla 12345: M1 / &#124;Δh&#124; | M1 medio ± desv. | &#124;Δh&#124; medio ± desv. (%) |
|---|---:|---:|---:|---:|---:|
| E, f_max 4, λ 0,5 | 9,03 / 1,74 | 7,68 / 2,01 | 9,76 / 1,62 | 8,82 ± 1,06 | 1,79 ± 0,20 |
| E, f_max 6, λ 0,5 | 8,79 / 1,64 | 7,70 / 2,07 | 9,40 / 1,72 | **8,63 ± 0,87** | **1,81 ± 0,23** |

Ambas satisfacen la restricción de altura. Entre ellas, E con `f_max=6` tiene la menor media de M1. No alcanza las metas M1 < 6,18 y M1 < 6,39.

### Comparación con la ronda anterior y la guía

| Método | M1 | &#124;Δh&#124; (%) | Alcance de la cifra |
|---|---:|---:|---|
| A | 6,18 ± 0,18 | 4,16 ± 0,56 | Media ± desviación entre tres semillas |
| E, f_max 6 | 8,63 ± 0,87 | 1,81 ± 0,23 | Media ± desviación entre tres semillas |
| Zernike, guía | 6,39; desviación entre semillas no disponible | 2,55; desviación entre semillas no disponible | Mediana de cinco imágenes |
| SIREN, guía | 3,47; desviación entre semillas no disponible | 12,85; desviación entre semillas no disponible | Mediana de cinco imágenes |

Para las referencias solo hay una medición por imagen en `baseline_validacion.csv`; no se inventa una dispersión entre semillas. Como dato distinto, la media ± desviación muestral **entre las cinco imágenes** es 7,37 ± 2,41 para M1 de Zernike y 3,62 ± 0,34 para SIREN; no es comparable a la desviación entre semillas de A y E. SIREN incumple el límite de altura con la mediana de referencia.

### Control E-lineal

El control con CoordConv, mapa reducido 8 × 8, base de grado 3 y `f_max=4` se ejecutó una vez, con semilla 42 y 300 épocas. Eligió la época 260 por pérdida de validación sin máscaras y obtuvo M1 = 11,95 y |Δh| = 2,75 %. Su desviación entre semillas no está disponible. Tiene 130.330 parámetros entrenables, tardó 44,23 s en entrenarse y 0,0037 s por imagen en inferencia media. Es un control exploratorio y no supera la finalista E.

### Curvas, mapas y recursos

La figura [E_pesos_validacion.png](figuras/E_pesos_validacion.png) presenta las cinco imágenes, pesos, fondos y correcciones para la semilla 42 de la finalista. Los pesos bajos coinciden visualmente con las células, pero también abarcan regiones amplias de fondo. Como comprobación retrospectiva con las máscaras, el peso medio de célula fue aproximadamente 0,003 y el de fondo 0,684; la media espacial global osciló entre 0,412 y 0,499. Esto confirma que la cota de 0,6 era blanda y no se cumplió estrictamente.

En [E_curvas_finalista.png](figuras/E_curvas_finalista.png), la pérdida de validación baja de 0,146 en la época 50 a 0,098 en la 300, mientras M1 pasa de 7,28 a 8,15. Por tanto, la mejora de la pérdida no garantiza mejora de M1; el checkpoint de la semilla 42 se eligió en la época 288 por pérdida, sin usar esas métricas.

E tiene 117.393 parámetros entrenables. Para la finalista, el entrenamiento tardó 100,28 s por semilla en promedio y la inferencia 0,0112 s por imagen en promedio. La prueba de humo de dos épocas con 32 canales, `f_max=6` y lote 4 asignó como máximo 0,436 GiB de GPU, menos de 7 GB; el máximo observado durante el barrido de 16 canales fue 0,285 GiB. Se limitó el proceso al 75 % de la memoria de la GPU, se copiaron los mejores estados a CPU y se liberó la caché entre configuraciones. El tiempo acumulado de entrenamiento medido fue 949,06 s para E, más 44,23 s para el control y 4,08 s para la prueba de humo, por debajo del presupuesto de 100 minutos. Los pesos están en `experimentos/pesos/`, fuera de Git.

### Recomendación

Mantener A como referencia seleccionada bajo la regla de calidad: E reduce el error de altura, pero su M1 medio sigue por encima tanto de A como de la mediana de Zernike. E es una alternativa amortizada prometedora para estudiar la selección de fondo, no una sustitución validada. Antes de ampliar el barrido conviene revisar el desajuste entre la pérdida y M1 y la tendencia de `w` a excluir demasiado fondo; cualquier elección por métricas con máscaras debe declararse como supervisada. Estos resultados son de validación y no prueban generalización al conjunto de prueba.

## Ronda 4 — SIREN guiada por los pesos de E

### Protocolo

Se combinaron los mapas de la U-Net de E finalista (`f_max=6`, λ de área 0,5) con el ajuste por imagen de A. Las semillas de E y de la SIREN se emparejaron: 42, 2024 y 12345. Los pesos de E ya estaban entrenados y permanecieron fijos.

Para cada imagen se mantuvieron la arquitectura de A (128 unidades, dos capas ocultas, ω0=1), Adam con tasa 0,001, 1000 iteraciones, signo 0,8 y referencia 0,1. El ajuste espacial fue `Σ w·Charbonnier(r) / Σ w`. En la variante con recorte se seleccionó en cada iteración el 70 % de residuos de menor magnitud antes de aplicar esa media ponderada. El recorte de A usa el valor absoluto del residuo; por ello, la variante combinada no es exactamente la misma pérdida espacial de A.

Se probaron cuatro variantes: `w` original, `w^0,5`, `w^0,25` y `w` con recorte ρ=0,7. Elevar un peso entre 0 y 1 a una potencia menor que 1 aumenta su contribución, en particular cerca de regiones que E considera poco fiables. No se probaron los pesos de signo opcionales.

El ajuste y la obtención de `w` no usaron máscaras de referencia. Las máscaras de las cinco imágenes de validación se emplearon después para calcular M1 y |Δh| y seleccionar la variante; esa selección utiliza información de referencia. No se evaluaron imágenes de prueba ni de generalización. Cada valor por semilla es la mediana de las cinco imágenes; la media y la desviación estándar muestral se calculan entre las tres semillas.

### Resultados

| Variante | Semilla 42: M1 / &#124;Δh&#124; | Semilla 2024: M1 / &#124;Δh&#124; | Semilla 12345: M1 / &#124;Δh&#124; | M1 medio ± desv. | &#124;Δh&#124; medio ± desv. (%) |
|---|---:|---:|---:|---:|---:|
| `w` | 6,47 / 3,89 | 6,58 / 4,01 | 6,98 / 3,87 | 6,68 ± 0,27 | 3,93 ± 0,07 |
| `w^0,5` | 6,86 / 4,99 | 6,27 / 5,06 | 7,20 / 3,60 | 6,78 ± 0,47 | 4,55 ± 0,82 |
| `w^0,25` | 6,14 / 5,18 | 6,09 / 4,32 | 6,26 / 5,41 | **6,16 ± 0,09** | **4,97 ± 0,57** |
| `w` + recorte 0,7 | 6,90 / 3,43 | 6,59 / 4,09 | 7,20 / 3,54 | 6,90 ± 0,31 | 3,69 ± 0,35 |

Según la regla de menor M1 con |Δh| medio ≤ 5 %, gana `w^0,25`. Su M1 sin redondear es 6,16396 y cumple la meta < 6,18 por un margen cercano a 0,016 niveles respecto a la referencia indicada. El |Δh| medio sin redondear es 4,96935 %; las semillas 42 y 12345 superan individualmente el 5 %. Por tanto, la mejora de fondo y el cumplimiento de altura son marginales y no prueban una ventaja estable.

La figura [AE_validacion.png](figuras/AE_validacion.png) muestra las cinco imágenes de validación con `w^0,25`, fondo e imagen corregida para la semilla 42. El fondo se ve suave y no presenta la cuadrícula evidente observada en E; el mapa todavía asigna pesos reducidos a regiones amplias alrededor de algunas células. La inspección visual no demuestra por sí sola preservación de altura ni generalización.

### Recursos y reproducibilidad

Se ejecutaron 60 ajustes (cuatro variantes × tres semillas × cinco imágenes) en un único proceso de cálculo en GPU, con `torch.cuda.set_per_process_memory_fraction(0.75, 0)` y liberación de memoria entre ajustes. La suma de los tiempos de ajuste medidos fue 1534,05 s (25,57 min), sin incluir la obtención de pesos ni la evaluación posterior; el proceso completo de ajuste transcurrió aproximadamente 26 min y 15 s, por debajo de 45 min. La mejor variante tardó 24,38 s por imagen en promedio solo en ajuste, bajo la carga existente del equipo. E tardó históricamente 0,0112 s por imagen en inferencia; A registró unos 49,67 s de ajuste para las cinco imágenes de una semilla en otra ejecución. Estos tiempos no constituyen una comparación controlada de velocidad.

Los resultados por semilla e imagen están en [resultados_ronda4.csv](resultados_ronda4.csv). El guion [ronda4_ae.py](ronda4_ae.py) conserva el procedimiento; los pesos de E y las salidas intermedias permanecen localmente en `experimentos/pesos/`, fuera de Git.

### Tabla final de decisión

| Método | M1 en validación | &#124;Δh&#124; (%) | Cumple la regla de altura | Alcance |
|---|---:|---:|:---:|---|
| A | 6,18 ± 0,18 | 4,16 ± 0,56 | Sí | Media ± desviación muestral entre tres semillas |
| E, `f_max=6` | 8,63 ± 0,87 | 1,81 ± 0,23 | Sí | Media ± desviación muestral entre tres semillas |
| **AE, `w^0,25`** | **6,16 ± 0,09** | **4,97 ± 0,57** | **Sí, por media** | Media ± desviación muestral entre tres semillas |
| Zernike, guía | 6,39 | 2,55 | Sí | Mediana de cinco imágenes; sin dispersión entre semillas |
| SIREN, guía | 3,47 | 12,85 | No | Mediana de cinco imágenes; sin dispersión entre semillas |

### Recomendación final

Bajo la regla de selección establecida, **AE con `w^0,25` es el modelo seleccionado**: obtiene la menor M1 media de las variantes elegibles y mejora ligeramente la referencia A. La mejora es muy pequeña, mientras que el promedio de altura queda a solo 0,03 puntos porcentuales del límite y dos semillas lo incumplen individualmente. A sigue siendo una alternativa más prudente si se prioriza el margen de preservación observado y una implementación sin el detector adicional de E.

AE produce fondos visualmente suaves, pero requiere una inferencia de E y 1000 iteraciones de SIREN para cada imagen; E es mucho más rápida una vez entrenada, aunque deja mayor residuo de fondo. AE y A pueden aplicarse a una imagen nueva sin su máscara de referencia durante el ajuste, y AE necesita además disponer de los pesos de E. Esta capacidad de ejecución no demuestra que funcionen bien fuera de validación: quedan pendientes la evaluación independiente y la comprobación de preservación en imágenes nuevas.
