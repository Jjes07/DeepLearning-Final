"""Integra las secciones experimentales conservando el resto del cuaderno."""
from pathlib import Path
import importlib.util
import nbformat

RAIZ = Path(__file__).resolve().parent.parent


def cargar(nombre):
    ruta = RAIZ / "scripts" / f"celdas_{nombre}.py"
    spec = importlib.util.spec_from_file_location(f"celdas_{nombre}", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return getattr(modulo, f"CELDAS_{nombre.upper()}")


def main():
    ruta = RAIZ / "EritroFase.ipynb"
    nb = nbformat.read(ruta, as_version=4)
    inicio = [i for i, c in enumerate(nb.cells)
              if c.cell_type == "markdown" and c.source.startswith("## 7. Arquitectura")]
    fin = [i for i, c in enumerate(nb.cells)
           if c.cell_type == "markdown" and c.source.startswith("## 13. Referencias")]
    if len(inicio) != 1 or len(fin) != 1 or inicio[0] >= fin[0]:
        raise ValueError("No se encontraron los límites únicos de las secciones 7–12")
    nb.cells[inicio[0]:fin[0]] = cargar("7_8") + cargar("tiempos") + cargar("9_12")
    guia = nb.cells[1]
    guia.source = guia.source.replace(
        "**La ejecución que se entrega se hace con `RECALCULAR = True` y *Restart & Run All***, de modo que todas las salidas provienen de una sola ejecución completa.",
        "En esta ejecución se reutiliza la caché existente de las secciones 2–6; las secciones 7–12 se calculan y guardan por separado. Una caché cargada no implica un nuevo entrenamiento.")
    config = nb.cells[10]
    config.source = config.source.replace(
        "#   True  -> se recalcula todo. OBLIGATORIO en la ejecución final que se entrega.",
        "#   True  -> se recalculan los pasos guardados en caché.")
    config.source = config.source.replace(
        "#   False -> se reutiliza lo guardado en results/cache/ (modo desarrollo, para retomar rápido).",
        "#   False -> se reutiliza lo guardado en caché cuando existe.")
    registro = nb.cells[2]
    registro.source = registro.source.replace(
        "### 0.5 Registro de decisiones",
        "| H7–H10 | Arquitectura, entrenamiento, prueba y otro día | Criterios y contraste en las secciones indicadas | Ver salidas y lecturas correspondientes | 7–10 |\n\n### 0.5 Registro de decisiones")
    registro.source = registro.source.replace(
        "### 0.6 Decisiones revisables",
        "| Modelo A por imagen, pérdida recortada y semilla elegida en validación | C, D, E y AE | Compromiso entre fondo, preservación y costo; comparación y límites en las salidas | 7–8 |\n\n### 0.6 Decisiones revisables")
    registro.source = registro.source.replace(
        "Por definir: A (por imagen), B (U-Net) o C (coeficientes o hiperred)",
        "A, ajuste por imagen")
    registro.source = registro.source.replace(
        "| 9 | Pérdidas y sus pesos (supuestos S1–S4) | Por definir |",
        "| 9 | Pérdidas y sus pesos (supuestos S1–S4) | Recorte 0,7; signo 0,8; referencia 0,1 |")
    registro.source = registro.source.replace(
        "| 10 | Normalización de la entrada | Por definir | Pendiente desde 3.11 |",
        "| 10 | Normalización de la entrada | Niveles equivalentes sin normalizar | Reproduce el experimento A |")
    nbformat.validate(nb)
    nbformat.write(nb, ruta)
    print(f"Integradas {len(nb.cells)} celdas en {ruta.name}")


if __name__ == "__main__":
    main()
