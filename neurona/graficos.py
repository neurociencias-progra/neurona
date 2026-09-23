"""WORKER 3 — Dibujar. Solo eso.

No lee archivos, no calcula tasas. Recibe datos ya listos y los pinta.

No importa a ningún otro worker: esa es la regla.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class Graficador:
    """Dibuja las dos figuras clásicas de una neurona: el ráster y las tasas.

    Args:
        colores: nombre de un mapa de color de Matplotlib para distinguir clases.
    """

    def __init__(self, colores: str = "tab20"):
        self.colores = colores

    def __repr__(self) -> str:
        return f"Graficador(colores={self.colores!r})"

    def _paleta(self, n: int):
        mapa = plt.get_cmap(self.colores)
        return [mapa(i % mapa.N) for i in range(n)]

    def raster(self, lista_de_tiempos, etiquetas=None, titulo: str = "Ráster", ax=None):
        """Un punto por espiga, una fila por ensayo.

        Args:
            lista_de_tiempos: una entrada por ensayo, con sus tiempos de espiga.
            etiquetas: opcional, la clase de cada ensayo. Si se da, los ensayos se
                agrupan por clase y cada clase recibe un color.
            titulo: el título de la figura.
            ax: un eje de Matplotlib donde dibujar. Si es None, se crea uno.

        Returns:
            El eje donde se dibujó, por si quieres seguir retocándolo.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if etiquetas is None:
            ax.eventplot(lista_de_tiempos, colors="black", linelengths=0.8)
        else:
            etiquetas = np.asarray(etiquetas)
            clases = np.unique(etiquetas)
            paleta = self._paleta(len(clases))
            orden, colores_fila = [], []
            for color, clase in zip(paleta, clases):
                for i in np.where(etiquetas == clase)[0]:
                    orden.append(lista_de_tiempos[i])
                    colores_fila.append(color)
            ax.eventplot(orden, colors=colores_fila, linelengths=0.8)

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Ensayo")
        ax.set_title(titulo)
        return ax

    def curvas(self, centros, curvas_por_clase: dict, titulo: str = "Tasa de disparo", ax=None):
        """Una curva de tasa por clase, todas en el mismo lienzo.

        Args:
            centros: el eje temporal, compartido por todas las curvas.
            curvas_por_clase: diccionario `clase -> arreglo de tasas`.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        paleta = self._paleta(len(curvas_por_clase))
        for color, (clase, tasas) in zip(paleta, sorted(curvas_por_clase.items())):
            ax.plot(centros, tasas, color=color, label=f"clase {clase}")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Tasa de disparo (Hz)")
        ax.set_title(titulo)
        ax.legend(fontsize=8, ncol=2)
        return ax
