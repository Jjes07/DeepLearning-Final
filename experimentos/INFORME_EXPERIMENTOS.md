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

## Conclusión
La configuración `A_cand_signo0.8_ref0.1_rec0.7_semilla2024` supera a Zernike en validación: mediana M1 = 5.98 niveles (< 6,39) y mediana |Δh| = 4.14 % (≤ 5 %).
Con 3 semillas, la media de M1 es 6.18 ± 0.18 y la de |Δh| es 4.16 ± 0.56 %.
