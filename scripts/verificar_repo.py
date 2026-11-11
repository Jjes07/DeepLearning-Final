"""Verifica que el repositorio cumpla las reglas del curso antes de integrar cambios."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat

LIMITE_MB = 20
EXTENSIONES_PROHIBIDAS = {".pt", ".pth", ".ckpt"}


def tamano_mb(ruta: Path) -> float:
    return ruta.stat().st_size / (1024 * 1024)


def archivos_versionados(raiz: Path) -> list[Path]:
    """Devuelve los archivos que git está rastreando, excluyendo submódulos."""
    resultado = subprocess.run(
        ["git", "ls-files"],
        cwd=raiz,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return [raiz / linea for linea in resultado.stdout.splitlines() if linea]


def validar_notebook(ruta: Path) -> list[str]:
    """Valida la estructura del notebook y busca celdas de error."""
    errores: list[str] = []

    try:
        with ruta.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as exc:  # noqa: BLE001
        return [f"No se pudo leer el notebook {ruta}: {exc}"]

    try:
        nbformat.validate(nb)
    except nbformat.ValidationError as exc:
        errores.append(f"El notebook {ruta} no es válido según nbformat: {exc}")

    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "code":
            continue
        for salida in celda.get("outputs", []):
            if salida.output_type == "error":
                ename = salida.get("ename", "Error")
                evalue = salida.get("evalue", "")
                errores.append(
                    f"Celda de código {i + 1} en {ruta} tiene una salida de error: "
                    f"{ename}: {evalue}"
                )

    return errores


def validar_archivos(raiz: Path) -> list[str]:
    """Revisa archivos prohibidos o demasiado grandes bajo control de versiones."""
    errores: list[str] = []

    for ruta in archivos_versionados(raiz):
        if not ruta.exists():
            continue

        relativa = ruta.relative_to(raiz).as_posix()

        if relativa.startswith("data/raw/"):
            errores.append(
                f"Archivo prohibido bajo data/raw/: {relativa}. "
                "Los datos crudos no se versionan."
            )

        if ruta.suffix.lower() in EXTENSIONES_PROHIBIDAS:
            errores.append(
                f"Archivo prohibido ({ruta.suffix}): {relativa}. "
                "Los pesos de modelos no se versionan."
            )

        if tamano_mb(ruta) > LIMITE_MB:
            errores.append(
                f"Archivo demasiado grande ({tamano_mb(ruta):.1f} MB): {relativa}. "
                f"El límite es {LIMITE_MB} MB."
            )

    return errores


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verifica el notebook y los archivos versionados del repositorio."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Raíz del repositorio (por defecto: directorio actual).",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=None,
        help="Ruta del notebook a validar (por defecto: busca *.ipynb en la raíz).",
    )
    args = parser.parse_args()

    raiz = args.repo.resolve()

    if args.notebook is None:
        notebooks = sorted(raiz.glob("*.ipynb"))
        if not notebooks:
            print("No se encontró ningún notebook en la raíz del repositorio.")
            return 1
        notebook = notebooks[0]
    else:
        notebook = args.notebook.resolve()

    print(f"Verificando repositorio: {raiz}")
    print(f"Notebook: {notebook.relative_to(raiz)}")

    errores = validar_notebook(notebook) + validar_archivos(raiz)

    if errores:
        print("\nSe encontraron problemas:")
        for error in errores:
            print(f"  - {error}")
        return 1

    print("\nTodas las verificaciones pasaron.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
