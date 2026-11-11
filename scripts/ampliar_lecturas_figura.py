"""Añade en EritroFase.ipynb las celdas markdown de lectura de 9.1b y 10.1b y actualiza el párrafo 12.1
con las cifras de generalización. Las cifras provienen de la ejecución de scripts/ampliar_generalizacion_figura.py.
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "EritroFase.ipynb"

LECTURA_9_1B = """**Lectura de la figura (9.1b).** Medianas en prueba (5 imágenes):

| Método | M1 (niveles eq.) | \\|Δh\\| mediana (%) |
|---|---|---|
| Pistón | 18,18 | 0,00 |
| Polinomio g3 | 10,37 | 1,19 |
| Zernike (guía) | 6,49 | 2,94 |
| SIREN ω0=1 (guía) | 4,22 | 12,27 |
| A (propuesto) | 7,10 | 4,81 |

- En M1, A queda entre el Zernike y el polinomio de grado 3, sin superar al Zernike.
- En \\|Δh\\|, A queda por encima del Zernike y del polinomio de grado 3, que son los métodos que mejor preservan la célula en esta muestra. Es consistente con el resultado de 9.1: el criterio de medianas no se cumple.
- La SIREN deja el fondo más bajo, pero su \\|Δh\\| (cerca de 12 %) indica que absorbe parte de la célula.
"""

LECTURA_10_1B = """**Lectura de la salida (10.1b).** Medianas en prueba y en el otro día (8 imágenes; \\|Δh\\| usa las 7 con células evaluables):

| Método | M1 prueba | M1 otro día | \\|Δh\\| prueba (%) | \\|Δh\\| otro día (%) |
|---|---|---|---|---|
| Pistón | 18,18 | 20,53 | 0,00 | 0,00 |
| Polinomio g3 | 10,37 | 9,29 | 1,19 | 2,01 |
| Zernike (guía) | 6,49 | 5,98 | 2,94 | 4,60 |
| SIREN ω0=1 (guía) | 4,22 | 3,78 | 12,27 | 12,52 |
| A (propuesto) | 7,10 | 11,79 | 4,81 | 2,07 |

- **Orden de los métodos.** En M1 se mantienen los extremos: la SIREN es la de menor M1 y el Pistón la de mayor en ambos conjuntos. El orden intermedio cambia: A pasa del 3.º al 4.º lugar, por detrás del polinomio de grado 3 y del Zernike. En \\|Δh\\| se invierte la posición de A y el Zernike.
- **A frente a Zernike.** En M1, A no supera al Zernike en el otro día (11,79 frente a 5,98 niveles). En \\|Δh\\| sí lo supera (2,07 % frente a 4,60 %), pero en prueba ocurría lo contrario (4,81 % frente a 2,94 %). Con 8 imágenes, 3 escenas y pocas células evaluables por imagen, esta diferencia no es concluyente.
- Los resultados son consistentes con que el orden intermedio de los métodos dependa del conjunto de imágenes; no bastan para afirmar que A supere a Zernike en generalización.
"""

VIEJO_12_1 = ("En el otro día, la mediana de M1 es 11,79 y la de |Δh| es 2,07 %. "
              "Minimizar solo M1 podría absorber parte de la célula.")
NUEVO_12_1 = ("En el otro día (8 imágenes), la SIREN sigue con el M1 más bajo (3,78 niveles) y el Pistón con el más alto (20,53), "
              "pero A obtiene 11,79 niveles, por detrás del Zernike (5,98) y del polinomio de grado 3 (9,29). "
              "Su |Δh| mediana es 2,07 % frente a 4,60 % del Zernike, lo contrario que en prueba. "
              "El orden intermedio no es estable entre conjuntos y A no supera al Zernike en M1 en ninguno de los dos. "
              "Minimizar solo M1 podría absorber parte de la célula.")


def posicion(nb, condicion) -> int:
    pos = [i for i, c in enumerate(nb.cells) if condicion(c)]
    assert len(pos) == 1, f"posición no única: {pos}"
    return pos[0]


def main() -> None:
    nb = nbformat.read(RUTA, as_version=4)
    assert not any("Lectura de la figura (9.1b)" in c.source for c in nb.cells), "lectura 9.1b ya existe"
    assert not any("(10.1b)" in c.source for c in nb.cells), "lectura 10.1b ya existe"

    i91b = posicion(nb, lambda c: c.cell_type == "code" and c.source.startswith("# 9.1b"))
    nb.cells.insert(i91b + 1, new_markdown_cell(LECTURA_9_1B))

    i101b = posicion(nb, lambda c: c.cell_type == "code" and c.source.startswith("# 10.1b"))
    nb.cells.insert(i101b + 1, new_markdown_cell(LECTURA_10_1B))

    i12 = posicion(nb, lambda c: c.get("id") == "4067c53b")
    texto = nb.cells[i12].source
    assert texto.count(VIEJO_12_1) == 1, "el párrafo 12.1 no coincide"
    nb.cells[i12].source = texto.replace(VIEJO_12_1, NUEVO_12_1)

    nbformat.validate(nb)
    nbformat.write(nb, RUTA)
    print(f"Celdas totales: {len(nb.cells)}")


if __name__ == "__main__":
    main()
