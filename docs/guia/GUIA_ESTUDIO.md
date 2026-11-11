# Guía de estudio — Corrección del fondo en imágenes de fase de eritrocitos

**Curso:** SI3014 · Redes Neuronales y Aprendizaje Profundo · Universidad EAFIT  
**Proyecto:** EritroFase — Corrección del fondo en imágenes de fase de eritrocitos mediante redes neuronales informadas por la física  
**Fuentes principales:** `EritroFase.ipynb` (secciones 1, 3, 4, 5, 6), `docs/references/*.pdf`, `reference_code/INR/*.py`.  
**Advertencia:** todo número y afirmación de esta guía proviene de esas fuentes. Lo que aún no está en ellas está marcado como *pendiente de verificar*.

---

## Índice

1. [El problema en una página](#1-el-problema-en-una-página)
2. [Fase óptica, envolvimiento y desenvolvimiento](#2-fase-óptica-envolvimiento-y-desenvolvimiento)
3. [El dataset](#3-el-dataset)
4. [Modelo de formación y no identificabilidad](#4-modelo-de-formación-y-no-identificabilidad)
5. [Representaciones neuronales implícitas: SIREN y Fourier](#5-representaciones-neuronales-implícitas-siren-y-fourier)
6. [¿Qué significa "informada por la física"?](#6-qué-significa-informada-por-la-física)
7. [Funciones de pérdida](#7-funciones-de-pérdida)
8. [Líneas base y resultados de validación](#8-líneas-base-y-resultados-de-validación)
9. [Métricas M1 y M2](#9-métricas-m1-y-m2)
10. [Glosario](#10-glosario)
11. [Banco de preguntas probables](#11-banco-de-preguntas-probables)

---

## 1. El problema en una página

### 1.1 ¿Qué es una imagen de fase?

Una cámara registra **intensidad** de luz (cuánta luz llega), no la fase. Pero cuando la luz atraviesa un objeto transparente, lo que cambia es el **retraso de la onda**, es decir, su fase. La microscopía holográfica digital (DHM) recupera esa fase indirectamente haciendo interferir la onda que pasó por la muestra con una onda de referencia.

> **Fase de una muestra transparente (Proposición 1):**  
> $$\varphi(x,y)=\frac{2\pi\,(n_s-n_m)\,t(x,y)}{\lambda}$$  
> donde $n_s$ es el índice de refracción de la muestra, $n_m$ el del medio, $t$ el espesor y $\lambda$ la longitud de onda.

**Ejemplo numérico.** Para una célula con $n_s-n_m\approx 0.05$, $t\approx 2\,\mu\text{m}$ y $\lambda=532\,\text{nm}$:  
$$\varphi\approx\frac{2\pi\cdot 0.05\cdot 2\,\mu\text{m}}{0.532\,\mu\text{m}}\approx 1.18\,\text{rad}.$$

### 1.2 ¿Por qué los eritrocitos se ven en fase y no en intensidad?

Son casi transparentes: absorben muy poca luz. Lo que modifican es el camino óptico, y por tanto la fase. En la imagen de fase aparecen como relieves; en intensidad serían prácticamente invisibles.

### 1.3 ¿Qué es el fondo y por qué estorba?

La fase reconstruida no es cero fuera de las células. Quedan:
- inclinaciones residuales del haz de referencia,
- curvatura y aberraciones del sistema óptico,
- un desplazamiento constante arbitrario.

Eso forma el **fondo** $b(x,y)$, que se suma a la señal celular. Como las medidas cuantitativas (altura, volumen, masa seca) se calculan respecto al nivel del fondo, cualquier fondo inclinado o curvo introduce sesgo.

### 1.4 ¿Qué significa "fondo negro preservando la célula"?

Significa estimar el fondo $\hat b$ y restarlo para que la región sin células quede en cero (negro en la visualización), **sin alterar** los valores de fase dentro de las células. La dificultad central es que la descomposición no es única: si no se impone ningún supuesto, $\hat b=u$ reproduce la imagen perfectamente y borra las células.

---

## 2. Fase óptica, envolvimiento y desenvolvimiento

### 2.1 La fase como ángulo

La fase es un ángulo. Solo se conoce módulo $2\pi$: el arco tangente devuelve un valor en $(-\pi,\pi]$. Si la fase real supera ese intervalo, la imagen muestra saltos artificiales de $2\pi$.

> **Fase envuelta:** $W(\varphi)=\varphi-2\pi\,\text{round}(\varphi/2\pi)\in(-\pi,\pi]$.

### 2.2 Relación entre niveles de gris y radianes

Los PNG del dataset son de 8 bits, valores 0–255. La hipótesis de trabajo, consistente con los datos, es:

$$\varphi_{\text{env}}=\frac{2\pi\,g}{255}-\pi.$$

Por tanto:

$$1\,\text{nivel de gris}\approx\frac{2\pi}{255}\approx 0.0246\,\text{rad}.$$

**Ejemplo.** Un salto de 255 niveles en la imagen original equivale a un salto de $2\pi$ rad ($\approx 6.28$ rad) en la fase envuelta.

### 2.3 Desenvolvimiento

El desenvolvimiento suma o resta múltiplos enteros de $2\pi$ en cada píxel para reconstruir una fase continua. El notebook usa `skimage.restoration.unwrap_phase`, que implementa el algoritmo de Herráez et al. (2002). Desenvuelve primero los píxeles más confiables siguiendo un camino no continuo.

**Ejemplo.** En una imagen de generalización, el perfil original salta entre $\sim 250$ y $\sim 0$ niveles en el fondo; tras desenvolver, el perfil es continuo y la mancha blanca desaparece.

### 2.4 ¿Por qué `unwrap_phase` y no SPINR?

El código guía incluye `spinr_unwrap`, un desenvolvimiento con SIREN. En el ejemplo sintético de `main.py`:

- `unwrap_phase` recupera la fase verdadera con RMS = 0.0000 rad y tarda 0.0033 s.
- SPINR da RMS = 2.15 rad (deja un escalón de $\sim 2\pi$) y tarda 8.76 s, unas **2.650 veces más**.

En imágenes reales, la diferencia SPINR − `unwrap_phase` tiene RMS de 1.50 y 2.88 rad, y SPINR deja saltos residuales donde `unwrap_phase` no. Por eso se eligió `unwrap_phase`.

### 2.5 Reversibilidad

Desenvolver no altera el dato medido: la fase envuelta se recupera exactamente de la desenvuelta. El riesgo es contar mal las vueltas; ese riesgo se controla contando los saltos $>\pi$ que quedan tras desenvolver.

---

## 3. El dataset

### 3.1 Origen

Castañeda, Trujillo y Doblas (2024), *A human erythrocytes hologram dataset for learning-based model training*, *Data in Brief* 54, 110424. Datos públicos en OSF (doi 10.17605/OSF.IO/8P7BA).

| Característica | Valor |
|---|---|
| Sistema | DHM fuera de eje, telecéntrico, Mach–Zehnder en transmisión |
| Iluminación | Láser de 532 nm |
| Objetivo | 40× / 0.65 NA |
| Cámara | 1920×1200 píxeles de 5.86 µm |
| Recortes | Parches de 256×256 píxeles |
| Pares holograma–fase | 36.864 |

### 3.2 Estructura real de los archivos

- 36.864 archivos en `Phase/` y otros tantos en `Holograms/`.
- 12.288 recortes distintos; cada uno tiene 3 variantes: base, `-A` y `-B`.
- 192 adquisiciones (marca de tiempo `fecha__hora`).
- 4 campos `X` por adquisición.
- 16 recortes `Y` por campo.

### 3.3 Volteos, rotaciones y cuadrícula

- `-A` es volteo vertical exacto de la base; `-B` es volteo horizontal exacto.
- Los 16 recortes de un campo forman un mosaico de 1024×1024; el 90 % de los campos es continuo.
- Los 4 campos de una adquisición son la **misma escena rotada** 90°, 180° y 270°. La correlación máxima entre campos de la misma adquisición está entre 0.60 y 0.86; entre adquisiciones distintas no pasa de 0.24.

### 3.4 Fuga en la partición original

La partición original `Training`/`Validation` comparte contenido:

- 5.837 de 12.288 recortes (47.5 %) aparecen en ambas carpetas.
- El 100 % de los campos y adquisiciones aparece en ambas carpetas.

Eso invalida la partición original como medida de generalización.

### 3.5 Partición del proyecto

- **Unidad de independiente = adquisición.** Dos recortes solo son independientes si vienen de adquisiciones distintas.
- **Desarrollo:** 22 imágenes del 29/01/2021, todas de adquisiciones distintas.
  - Entrenamiento: 12
  - Validación: 5
  - Prueba: 5
- **Generalización:** 8 imágenes del 28/01/2021, otro día de medición.
- Verificación: 0 adquisiciones compartidas entre conjuntos; prueba de contenido todos contra todos; ningún par supera el umbral de 0.999.

### 3.6 Unidades: niveles de gris equivalentes

Como el dataset no tiene calibración explícita, se trabaja en **niveles de gris equivalentes** $u$:

$$u=-\frac{255}{2\pi}\,\mathcal{U}\!\left(\frac{2\pi\,g}{255}-\pi\right),$$

con $\mathcal{U}$ el desenvolvimiento. El signo negativo corrige la convención del dataset, en la que las células quedan por debajo del fondo tras desenvolver. La conversión a radianes queda condicionada a la hipótesis de escala lineal.

---

## 4. Modelo de formación y no identificabilidad

### 4.1 Modelo aditivo

Tras el preprocesamiento, cada imagen $u$ cumple:

$$u(x,y)=c(x,y)+b(x,y)+\varepsilon(x,y),$$

- $c$: contribución de las células.
- $b$: fondo (aberraciones + constante).
- $\varepsilon$: fluctuación que no es ni célula ni fondo lento.

El objetivo es estimar $\hat b$ y entregar $u_{\text{corr}}=u-\hat b$.

### 4.2 No identificabilidad (Proposición 3)

Para cualquier función $h$, el par $(c+h,\; b-h)$ explica $u$ exactamente igual que $(c,b)$. En particular, $\hat b=u$ reproduce la imagen sin error y elimina todas las células. Sin supuestos adicionales el problema no tiene solución única.

### 4.3 Supuestos físicos S1–S4

| Supuesto | Formulación | Justificación | Evidencia en los datos |
|---|---|---|---|
| **S1 · Fondo suave** | $b$ es de baja frecuencia | Las aberraciones vienen de componentes ópticos grandes frente a una célula | Se cumple en parte: la componente lenta es del mismo orden que una textura no suave |
| **S2 · Muestra esparsa** | $c=0$ en la mayoría de los píxeles | Pocas células por campo | Las células ocupan ~24 % del área |
| **S3 · Signo** | $c\ge 0$ | $n_{\text{célula}}>n_{\text{medio}}$ (Proposición 1) | Con la convención de signo del preprocesamiento, las células quedan positivas |
| **S4 · Referencia** | Mediana del fondo corregido = 0 | La fase solo está definida salvo una constante | Se impone en la pérdida |

S1–S4 son los que rompen la no identificabilidad y seleccionan una solución con sentido físico.

---

## 5. Representaciones neuronales implícitas: SIREN y Fourier

### 5.1 ¿Qué es una representación neuronal implícita?

Una imagen se representa como una función continua $f_\theta:[-1,1]^2\to\mathbb{R}$ dada por una red neuronal. La **entrada son las coordenadas** $(x,y)$ de un píxel; la salida es el valor en ese píxel. Para reconstruir la imagen se evalúa la red en todas las coordenadas.

### 5.2 SIREN (Sitzmann et al., 2020)

> **SIREN:** cada capa oculta aplica $h_{l+1}=\sin\big(\omega_0\,(W_l h_l+b_l)\big)$ y la salida es lineal.

**Inicialización especial:**
- Primera capa: $W_0\sim\mathcal{U}(-1/n,\;1/n)$.
- Otras capas: $W_l\sim\mathcal{U}\big(-\sqrt{6/n}/\omega_0,\;\sqrt{6/n}/\omega_0\big)$.

Esta inicialización mantiene la distribución de activaciones estable a través de las capas. Además, la derivada de una SIREN es otra SIREN (la derivada del seno es un coseno, que es un seno desfasado), lo que la hace útil para supervisar derivadas.

**Papel de $\omega_0$.** Controla la frecuencia máxima de la primera capa. Con coordenadas en $[-1,1]$ y 2 entradas, el argumento del seno varía a lo más $2\omega_0$ radianes de un extremo al otro.

| $\omega_0$ | Variación máxima | Periodos a lo ancho | Uso en el proyecto |
|---|---|---|---|
| 1 | 2 rad | 0.32 | `LowFreqBackgroundNetwork` del código guía (fondo) |
| 10 | 20 rad | 3.2 | `ImplicitPhaseNetwork` del código guía |
| 30 | 60 rad | 9.5 | Valor usado por Sitzmann et al. para detalle fino |

> **Matiz:** $\omega_0$ no es un límite duro. Las capas siguientes componen senos y pueden generar frecuencias más altas. La suavidad depende también de la profundidad y del entrenamiento.

### 5.3 Características de Fourier (Tancik et al., 2020)

Antes del MLP, las coordenadas $\mathbf v=(x,y)$ se transforman en:

$$\gamma(\mathbf v)=\big[\cos(2\pi B\mathbf v),\;\sin(2\pi B\mathbf v)\big],$$

con $B_{ij}\sim\mathcal{N}(0,\sigma^2)$.

**Papel de $\sigma$.** Es el ancho de banda de la distribución de frecuencias. Con $\sigma$ pequeño la red subajusta (salida suave); con $\sigma$ grande sobreajusta. Tancik et al. muestran que importa más el ancho que la forma exacta de la distribución.

### 5.4 ¿En qué se parecen y en qué no?

| Aspecto | SIREN | Fourier features |
|---|---|---|
| Activación | Seno en todas las capas | ReLU/tanh en el MLP; seno solo en el embedding |
| Inicialización | Crítica y específica | El embedding es fijo o poco entrenable |
| Control de frecuencia | $\omega_0$ + profundidad | $\sigma$ del embedding |
| Derivadas | Derivada = otra SIREN | Más complicado |
| Uso típico | Representar señales y sus derivadas | Mejorar MLPs para alta frecuencia |

### 5.5 ¿Por qué controlar el ancho de banda equivale a imponer suavidad al fondo?

Una red con bajo $\omega_0$ o bajo $\sigma$ solo puede representar funciones que cambian lentamente en el dominio. Si forzamos al fondo a ser una red de baja frecuencia, la red no tiene grados de libertad para ajustar las células, que son estructuras pequeñas y de alto contraste. Así la arquitectura implementa el supuesto S1 (fondo suave).

### 5.6 Arquitectura elegida para el modelo final

<!-- PENDIENTE: modelo -->

---

## 6. ¿Qué significa "informada por la física"?

### 6.1 Lo que significa aquí

En este proyecto, "informada por la física" significa que la función de pérdida se construye a partir del modelo de formación de la imagen y de los supuestos físicos S1–S4:

- El modelo aditivo $u=c+b+\varepsilon$ justifica restar el fondo propuesto y evaluar el residuo.
- S2 (muestra esparsa) justifica una pérdida de esparsidad sobre el residuo.
- S1 (fondo suave) justifica una penalización de curvatura sobre el fondo.
- S3 (signo) justifica penalizar que el fondo quede por encima de la imagen.
- S4 (referencia) justifica que el fondo corregido tenga mediana cero.

**No hay etiquetas:** no existen versiones "limpias" de las imágenes. La red se entrena comparando su salida con la imagen observada, a través de un modelo físico.

### 6.2 Lo que NO significa

No significa resolver ecuaciones de propagación de ondas, ecuaciones de Maxwell o el problema directo de la holografía. No se modela cómo se forma el holograma; solo se usa el modelo aditivo de la fase ya reconstruida.

---

## 7. Funciones de pérdida

### 7.1 De los supuestos a los términos

Sea $b_\theta$ el fondo propuesto y $r_\theta=u-b_\theta$ el residuo. Se usan estos términos:

| Término | Pérdida | Supuesto |
|---|---|---|
| Esparsidad (Charbonnier) | $\mathcal L_{\text{esp}}=\frac1N\sum_i \rho_\epsilon(r_i)$, $\rho_\epsilon(x)=\sqrt{x^2+\epsilon^2}$ | S2 |
| Esparsidad recortada | Media de los $\lfloor\rho N\rfloor$ valores más pequeños de $|r_i|$ | S2 (código guía) |
| Suavidad | $\mathcal L_{\text{suav}}=\frac1N\sum_i (\Delta b_\theta)_i^2$ | S1 |
| Signo | $\mathcal L_{\text{sig}}=\frac1N\sum_i \max(0,-r_i-\kappa\hat\sigma)^2$ | S3 |
| Coeficientes | $\|\mathbf a\|_1$ si el fondo es una base | S1 (Tang et al.; Huang y Cao) |

La pérdida total es una combinación ponderada; los pesos se eligen en validación.

### 7.2 Charbonnier y L1 frente a L2 (Proposición 4)

> **Proposición 4.** Para un fondo constante $b$:  
> - $\arg\min_b \sum_i (u_i-b)^2 = \text{media}(u)$.  
> - $\arg\min_b \sum_i |u_i-b| = \text{mediana}(u)$.

**Ejemplo.** 90 píxeles de fondo en 100 y 10 píxeles de célula en 200:
- Media: $(90\cdot 100+10\cdot 200)/100=110$. El fondo sale sesgado 10 niveles.
- Mediana: 100. Es robusta mientras las células ocupen menos del 50 %.

En el proyecto las células ocupan ~24 %, así que L1 (y su aproximación derivable Charbonnier) es más robusta que L2.

### 7.3 L1 recortada del código guía

`compensate_background` usa una pérdida que promedia solo una fracción $\rho$ de los residuos más pequeños. Eso ignora explícitamente los residuos grandes, que suelen ser células. El problema es que, si la red es muy flexible, aun así puede absorber parte de las células, como se ve en los resultados de la SIREN del código guía.

### 7.4 Término de signo

Penaliza que el fondo propuesto quede claramente por encima de la imagen. Si $r_i=u_i-b_i$ es muy negativo, significa que $b_i$ está por encima de la medida, lo que contradice S3 (las células deben estar por encima del fondo).

### 7.5 Suavidad con laplaciano

Penaliza la curvatura del fondo. Si $b$ es suave, su laplaciano es pequeño. Esto implementa S1 sin restringirse a una base polinómica o de Zernike.

---

## 8. Líneas base y resultados de validación

### 8.1 Líneas base definidas

| ID | Método | Fondo estimado | Parámetros libres |
|---|---|---|---|
| L0 | Pistón | $\text{mediana}(u)$ | 1 |
| L1 | Polinomio robusto grado $d\in\{1,2,3\}$ | $\sum_{i+j\le d} c_{ij}x^i y^j$ | 3, 6, 10 |
| L2 | Zernike robusto | 15 polinomios de Zernike (hasta orden 4), pesos robustos | 15 |
| L3 | SIREN de baja frecuencia | `LowFreqBackgroundNetwork` (128 neuronas, 2 capas, $\omega_0=1$) | 33.537 |

### 8.2 Resultados en validación (sección 6.4 del notebook)

Medianas sobre 5 imágenes de validación:

| Método | RMS lento residual (niv. eq.) | $\|\Delta h\|$ (%) |
|---|---|---|
| Pistón (L0) | 22.6 | 0.00 |
| Polinomio grado 1 (L1) | 15.8 | 0.26 |
| Polinomio grado 2 | 12.8 | 0.98 |
| Polinomio grado 3 | 7.9 | 1.37 |
| Zernike robusto (L2) | 6.4 | 2.55 |
| SIREN $\omega_0=1$ (L3) | **3.5** | **12.80** |

**Observaciones clave:**
- Cuanto más flexible el método, menos fondo lento deja, pero más altura le quita a las células.
- El Zernike preserva la célula ($\|\Delta h\|<5$ %) y deja 6.4 niveles de fondo lento.
- La SIREN deja solo 3.5 niveles de fondo lento, pero absorbe ~13 % de la altura celular.
- El grado del polinomio elegido es 3 (cumple la regla de parsimonia fijada).

### 8.3 Compromiso fondo–célula

El espacio que el modelo final debe ganar está **entre el Zernike y la SIREN del código guía**: dejar tan poco fondo lento como la SIREN sin su absorción. Las restricciones físicas de la sección 7 apuntan precisamente a eso.

### 8.4 Entrenamiento y resultados del modelo propuesto

<!-- PENDIENTE: modelo -->

---

## 9. Métricas M1 y M2

### 9.1 Notación

- $u$: imagen preprocesada.
- $v=u-\hat b$: imagen corregida.
- $C$, $B$: máscaras de referencia de célula y fondo.
- Todo en niveles de gris equivalentes.

### 9.2 M1: fondo lento residual

> **Definición 13.** Sea $G$ un núcleo gaussiano de $\sigma_s=8$ píxeles. Se define  
> $$\tilde v=\frac{G*(v\,\mathbb{1}_B)}{G*\mathbb{1}_B},\qquad \mathrm{RMS}_{\text{lento}}=\sqrt{\text{media}_{p\in B'}\,\tilde v(p)^2},$$  
> con $B'=\{p\in B:(G*\mathbb{1}_B)(p)>0.5\}$.

Es el promedio local del residuo usando **solo** píxeles de fondo. El suavizado elimina casi todo el ruido blanco: para $\sigma=14$ niveles y $\sigma_s=8$, el piso teórico es $\approx 14/(2\sqrt{\pi}\cdot 8)\approx 0.49$ niveles. Lo que queda por encima es fondo lento sin corregir.

**Intuición.** M1 responde: ¿el fondo entre células queda plano y cerca de cero, a escala de ~8 píxeles?

### 9.3 M2: altura local y su cambio

> **Definición 14.** Para cada célula evaluable $C_j$ con su anillo de fondo $R_j$:  
> $$h_j(w)=\text{media}_{C_j}w-\text{media}_{R_j}w,\qquad \Delta h_j=\frac{h_j(v)-h_j(u)}{h_j(u)}.$$

Por imagen se reporta la mediana de $\Delta h_j$ y de $|\Delta h_j|$. Valor ideal: 0.

### 9.4 ¿Por qué local y no global? (Proposición 5)

> **Proposición 5.** Si el error del fondo es afín en $C_j\cup R_j$, $e(\mathbf x)=e_0+\mathbf g\cdot\mathbf x$, entonces  
> $$h_j(u-\hat b)=h_j(u-b)-\mathbf g\cdot(\bar{\mathbf x}_{C_j}-\bar{\mathbf x}_{R_j}).$$  
> Si el anillo rodea simétricamente a la célula, $\bar{\mathbf x}_{R_j}=\bar{\mathbf x}_{C_j}$ y la altura local no depende del error afín.

**Ejemplo numérico.** Fondo cuadrático $b=a\,r^2$ centrado en una célula de radio 20 px, anillo de 26 a 36 px. Si el fondo sube 100 niveles del centro al borde de la imagen (128 px), $a\approx 0.0061$. La corrección legítima cambia la altura en $\approx 4.8$ niveles, un ~3 % de una célula de 150 niveles. De ahí el umbral $|\Delta h|<5$ %.

### 9.5 Qué no miden

- M1 no distingue si un residuo plano es ruido o fondo; solo mide la componente lenta.
- M2 no detecta una absorción proporcional de toda la célula si el fondo también cambia proporcionalmente en el anillo.
- Ninguna mide la verdad absoluta, porque no existe verdad conocida en las imágenes reales.

---

## 10. Glosario

1. **Aberración:** deformación del frente de onda introducida por el sistema óptico; se manifiesta como fondo en la imagen de fase.
2. **Adquisición:** marca de tiempo `fecha__hora` que agrupa 4 campos del dataset; unidad independiente.
3. **Ángulo de fase:** desplazamiento de la onda medido en radianes; determina el retraso óptico.
4. **Anillo de fondo:** región de 6 a 16 píxeles alrededor de una célula, usado en M2.
5. **Campo:** conjunto de 16 recortes con mismo `X` en una adquisición.
6. **Charbonnier:** función $\sqrt{x^2+\epsilon^2}$, aproximación derivable de $|x|$.
7. **Célula evaluable:** componente conexa de la máscara de referencia con área ≥ 600 px, lejos del borde y con anillo válido.
8. **DHM (microscopía holográfica digital):** técnica que registra el patrón de interferencia entre objeto y referencia para recuperar fase.
9. **Desenvolvimiento:** proceso de recuperar una fase continua sumando múltiplos de $2\pi$.
10. **Envolvimiento:** representación de la fase en $(-\pi,\pi]$, que produce saltos de $2\pi$.
11. **Eritrocito:** glóbulo rojo; objeto transparente estudiado en el dataset.
12. **Esparsidad:** propiedad de que la mayoría de los píxeles tienen valor cero (aquí, fuera de las células).
13. **Fase cuantitativa (QPI):** imagen que mide el retraso de fase de forma numérica.
14. **Fondo:** contribución no celular a la fase (aberraciones + constante).
15. **Fondo lento residual:** componente suave del fondo que queda sin corregir, medida por M1.
16. **Fourier features:** embedding senoidal de las coordenadas antes de un MLP.
17. **Fuga de información:** compartir muestras entre entrenamiento y evaluación.
18. **Holograma:** patrón de interferencia registrado por la cámara.
19. **Índice de refracción:** velocidad relativa de la luz en un medio; $n_s>n_m$ para una célula.
20. **Laplaciano:** operador de segunda derivada; penaliza curvatura.
21. **L1 recortada:** pérdida que ignora los residuos más grandes.
22. **Máscara de referencia:** división célula/fondo usada para evaluar, independiente del modelo.
23. **Mediana robusta:** estimador central menos sensible a valores atípicos que la media.
24. **Nivel de gris equivalente:** unidad de trabajo que conserva la escala 0–255 del PNG.
25. **No identificabilidad:** imposibilidad de separar $c$ y $b$ sin supuestos adicionales.
26. **Objetivo:** onda que pasó por la muestra, en contraste con la onda de referencia.
27. **Pérdida Charbonnier:** ver *Charbonnier*.
28. **Recorte:** imagen PNG de 256×256 del dataset.
29. **Representación neuronal implícita (INR):** red que mapea coordenadas a valores de imagen.
30. **SIREN:** red con activación seno e inicialización especial.
31. **Signo de la célula:** convención de que la célula es positiva tras el preprocesamiento.
32. **SPINR:** desenvolvimiento con SIREN incluido en el código guía.
33. **Suavidad:** propiedad del fondo de variar lentamente en el espacio.
34. **Transporte de intensidad:** técnica alternativa de QPI sin interferometría (mencionada en referencias).
35. **Variante:** versión `-A` o `-B` de un recorte, que son volteos exactos.
36. **Verdad conocida:** valor real de la fase; no existe para las imágenes reales del proyecto.
37. **Volumen celular:** propiedad biológica estimable a partir de la fase.
38. **Zernike:** familia de polinomios ortogonales usados para modelar aberraciones ópticas.

---

## 11. Banco de preguntas probables

### Problema

1. **¿Por qué no basta una imagen de intensidad para estudiar eritrocitos?**  
   Porque son casi transparentes; absorben muy poca luz. Lo que modifican es la fase.

2. **¿Qué información da la fase que no da la intensidad?**  
   El retraso óptico, que depende del espesor y del índice de refracción; permite estimar volumen, masa seca y morfología.

3. **¿Por qué el fondo es un problema?**  
   Las medidas cuantitativas se calculan respecto al nivel del fondo; un fondo inclinado o curvo introduce sesgo espacial.

4. **¿Qué significa "fondo negro preservando la célula"?**  
   Que la región sin células quede en cero, sin cambiar los valores de fase dentro de las células.

5. **¿Por qué no se puede simplemente restar la mediana global?**  
   Eso corrige solo el nivel medio (pistón), no la inclinación ni la curvatura del fondo.

### Arquitectura

6. **¿Qué es una representación neuronal implícita?**  
   Una red que recibe coordenadas $(x,y)$ y devuelve el valor de la imagen en ese punto; se evalúa en todos los píxeles.

7. **¿Qué es SIREN y en qué se diferencia de un MLP común?**  
   Usa seno como activación en todas las capas y tiene una inicialización especial que estabiliza las activaciones.

8. **¿Qué papel juega $\omega_0$ en SIREN?**  
   Controla la frecuencia máxima de la primera capa. $\omega_0=1$ solo puede variar 2 rad a lo ancho de la imagen; $\omega_0=30$ permite detalle fino.

9. **¿Qué son las características de Fourier?**  
   Un embedding $\gamma(\mathbf v)=[\cos(2\pi B\mathbf v),\sin(2\pi B\mathbf v)]$ antes del MLP, con $B\sim\mathcal{N}(0,\sigma^2)$.

10. **¿Qué controla $\sigma$ en Fourier features?**  
    El ancho de banda. $\sigma$ pequeño subajusta (salida suave); $\sigma$ grande sobreajusta.

11. **¿Por qué se usan redes de baja frecuencia para el fondo?**  
    Porque el fondo se supone suave (S1); una red de baja frecuencia no puede absorber detalles celulares.

12. **¿SIREN y Fourier features hacen lo mismo?**  
    No. SIREN pone senos en todas las capas; Fourier features solo en la entrada. La inicialización de SIREN es crítica; en Fourier features importa más $\sigma$.

13. **¿Por qué la derivada de una SIREN es otra SIREN?**  
    Porque la derivada de $\sin$ es $\cos$, que es un $\sin$ desfasado $\pi/2$.

14. **¿Qué ventaja tiene SIREN para problemas con derivadas?**  
    Puede supervisarse directamente con gradientes o laplacianos de la función.

### Datos

15. **¿De dónde viene el dataset?**  
    Castañeda, Trujillo y Doblas (2024); 36.864 pares holograma–fase de eritrocitos humanos.

16. **¿Qué contiene realmente el dataset?**  
    12.288 recortes distintos de 256×256, cada uno en 3 variantes (base, -A, -B), de 192 adquisiciones.

17. **¿Qué son `-A` y `-B`?**  
    Volteos exactos de la imagen base: `-A` vertical y `-B` horizontal.

18. **¿Por qué los 4 campos de una adquisición no son independientes?**  
    Son la misma escena rotada 90°, 180° y 270°.

19. **¿Cuál era el problema de la partición original?**  
    Compartía recortes, campos y adquisiciones entre `Training` y `Validation`; el 47.5 % de los recortes aparecía en ambas.

20. **¿Cómo se hizo la partición del proyecto?**  
    Una imagen por adquisición; días disjuntos entre desarrollo y generalización; prueba de contenido todos contra todos.

21. **¿Por qué se desenvuelve la fase antes de modelar?**  
    Porque los datos están envueltos; sin desenvolver ni el fondo es suave ni la célula se comporta como suma al fondo.

22. **¿En qué unidades se trabaja?**  
    Niveles de gris equivalentes. La conversión a radianes ($1\approx 0.0246$ rad) depende de la hipótesis de escala lineal.

23. **¿Por qué se invierte el signo en el preprocesamiento?**  
    Porque en el dataset las células quedan por debajo del fondo; se invierte para que sean positivas, como predice la física.

### Entrenamiento

24. **¿Qué significa "informada por la física" en este proyecto?**  
    Que la pérdida sale del modelo aditivo y de los supuestos S1–S4, no de etiquetas.

25. **¿Qué NO significa "informada por la física"?**  
    No significa resolver ecuaciones de propagación de ondas o el problema directo del holograma.

26. **¿Por qué no hay etiquetas?**  
    No existen versiones "limpias" de las imágenes reales.

27. **¿Qué supuestos rompen la no identificabilidad?**  
    S1 (fondo suave), S2 (muestra esparsa), S3 (signo positivo de la célula), S4 (mediana del fondo en cero).

28. **¿Por qué se prefiere L1 (Charbonnier) sobre L2?**  
    L1 es robusta a los valores atípicos (las células); la media de L2 se sesga hacia las células.

29. **¿Para qué sirve la L1 recortada?**  
    Para ignorar explícitamente los residuos más grandes, que suelen ser células.

30. **¿Qué penaliza el término de signo?**  
    Que el fondo propuesto quede por encima de la imagen, lo que contradice que la célula sea positiva.

31. **¿Qué penaliza el término de suavidad?**  
    La curvatura del fondo (laplaciano), implementando el supuesto S1.

### Resultados

32. **¿Cuáles son las líneas base y qué resultados dan?**  
    Pistón, polinomio robusto (grado 3 elegido), Zernike robusto y SIREN $\omega_0=1$. En validación, Zernike da 6.4 niveles de fondo lento y 2.55 % de cambio de altura; SIREN da 3.5 niveles pero 12.80 %.

33. **¿Qué muestra el compromiso fondo–célula?**  
    A mayor flexibilidad del método, menos fondo lento residual pero más absorción de la célula.

34. **¿Qué métricas son principales y por qué?**  
    M1 (fondo lento residual) y M2 (cambio local de altura). M1 mide si el fondo queda plano; M2 mide si la célula se preserva.

35. **¿Por qué M2 es local y no global?**  
    Porque un error afín del fondo se cancela al comparar la célula con un anillo de fondo simétrico (Proposición 5).

36. **¿Qué no miden M1 y M2?**  
    No distinguen ruido de fondo lento, ni detectan absorción proporcional, ni dan verdad absoluta.

### Generalización

37. **¿Qué imágenes se usan para generalización?**  
    8 imágenes de otro día (28/01/2021), con envolvimiento en el fondo y células de apariencia distinta.

38. **¿Por qué solo 8 imágenes de generalización?**  
    Porque vienen de 3 escenas nada más; los resultados deben interpretarse con esa limitación.

39. **¿Cómo se verifica que no haya fuga de información?**  
    Adquisiciones únicas por conjunto, días disjuntos, y prueba de contenido todos contra todos con umbral 0.999.

### Conclusiones

40. **¿Por qué deep learning si un polinomio funciona?**  
    El polinomio de grado 3 deja 7.9 niveles de fondo lento y el Zernike 6.4; la SIREN del código guía deja 3.5 pero absorbe la célula. El objetivo del modelo es ganar el espacio entre Zernike y SIREN: menos fondo que el primero, sin la absorción de la segunda. Si el modelo no lo logra, el polinomio/Zernike serían preferibles por simplicidad y velocidad.

---

## Referencias

1. R. Castañeda, C. Trujillo y A. Doblas (2024). A human erythrocytes hologram dataset for learning-based model training. *Data in Brief*, 54, 110424.
2. D. Xu, L. Cao y H. Sun (2025). Adaptive aberration-corrected quantitative phase microscopy via physics-informed deep learning. *Advanced Photonics*, 7(6), 066013.
3. J. Tang et al. (2023). Phase aberration compensation via a self-supervised sparse constraint network in digital holographic microscopy. *Optics and Lasers in Engineering*, 168, 107671.
4. Z. Huang y L. Cao (2023). Phase aberration separation for holographic microscopy by alternating direction sparse optimization. *Optics Express*, 31(8), 12520–12533.
5. V. Sitzmann et al. (2020). Implicit Neural Representations with Periodic Activation Functions. arXiv:2006.09661.
6. M. Tancik et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. arXiv:2006.10739.
7. Código guía del curso (`main.py`, `siren.py`, `reliability.py`, `losses.py`, `unwrapper.py`), en `reference_code/INR/`.
