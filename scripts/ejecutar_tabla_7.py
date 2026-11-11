"""Ejecuta la tabla de alternativas tras añadir el resumen del presupuesto."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
ruta = ROOT / "EritroFase.ipynb"
nb = nbformat.read(ruta, as_version=4)
celda = next(c for c in nb.cells if c.cell_type == "code" and c.source.startswith("# 7.1 Resultados documentados"))
preparacion = nbformat.v4.new_code_cell("""
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
TAB_DIR = Path.cwd() / "results" / "tables"
def guardar_tabla(df, nombre, index=False):
    df.to_csv(TAB_DIR / f"{nombre}.csv", index=index)
""")
temporal = nbformat.v4.new_notebook(cells=[preparacion, nbformat.v4.new_code_cell(celda.source)])
temporal.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
NotebookClient(temporal, timeout=-1, resources={"metadata": {"path": str(ROOT)}}).execute()
celda.outputs = temporal.cells[1].outputs
nbformat.validate(nb)
nbformat.write(nb, ruta)
print("Tabla 7.1 ejecutada y guardada.")
