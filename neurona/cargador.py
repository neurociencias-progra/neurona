"""WORKER 1 — Leer el archivo. Solo eso.

No calcula nada, no dibuja nada, no sabe qué es una tasa de disparo. Su único
trabajo es convertir un CSV del laboratorio en números que Python entienda.

No importa a ningún otro worker: esa es la regla.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# El archivo del laboratorio tiene 31 columnas de metadatos; de la 31 en adelante
# vienen los tiempos de espiga, y cada ensayo tiene un número distinto.
N_COLUMNAS_METADATOS = 31


class Cargador:
    """Lee un CSV de ensayos del laboratorio, que NO es rectangular.

    Cada fila es un ensayo: primero sus metadatos, después sus tiempos de espiga.
    Como cada ensayo tiene un número distinto de espigas, las filas tienen
    longitudes distintas y `pd.read_csv` no sirve tal cual.

    Args:
        ruta: camino al archivo `.csv`.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        if not self.ruta.exists():
            raise FileNotFoundError(f"No encuentro el archivo: {self.ruta}")

    def __repr__(self) -> str:
        return f"Cargador({self.ruta.name!r})"

    def metadatos(self) -> np.ndarray:
        """Las 31 columnas rectangulares → arreglo (n_ensayos, 31)."""
        return np.loadtxt(
            self.ruta,
            delimiter=",",
            usecols=range(N_COLUMNAS_METADATOS),
            skiprows=self._filas_de_encabezado(),
        )

    def tiempos_de_espiga(self) -> list[np.ndarray]:
        """La zona irregular → una lista con un arreglo de tiempos por ensayo.

        ⚠ Las líneas vacías se convierten en ensayos VACÍOS, no se saltan. Si se
        saltaran, la fila i de las espigas dejaría de corresponder a la fila i de
        los metadatos, y cada espiga quedaría etiquetada con la clase del ensayo
        equivocado. Es un error silencioso, que son los peores.
        """
        trenes: list[np.ndarray] = []
        with open(self.ruta, "r", encoding="utf-8") as f:
            for i, linea in enumerate(f):
                if i < self._filas_de_encabezado():
                    continue
                if not linea.strip():
                    trenes.append(np.array([]))
                    continue
                campos = linea.strip().split(",")
                trenes.append(
                    np.array([float(x) for x in campos[N_COLUMNAS_METADATOS:]])
                )
        return trenes

    def _filas_de_encabezado(self) -> int:
        """¿El archivo trae una fila de títulos? Se lo preguntamos al archivo.

        Algunos archivos del laboratorio la traen y otros no. En vez de adivinar,
        intentamos convertir la primera celda a número: si funciona son datos, si
        lanza `ValueError` es texto. Es el `try/except` de la Clase 3 tomando una
        decisión por nosotros.
        """
        with open(self.ruta, "r", encoding="utf-8") as f:
            primera = f.readline().split(",")[0]
        try:
            float(primera)
            return 0
        except ValueError:
            return 1
