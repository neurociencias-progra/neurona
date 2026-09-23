"""Un tren de espigas: los disparos de una neurona en un ensayo.

Esto es lo que construimos en la Clase 1 (POO 2), ahora viviendo en un archivo en
vez de en una celda. Es un *objeto de datos*: no hace trabajo, solo guarda unos
tiempos y sabe comportarse como un ciudadano de Python.
"""

from __future__ import annotations

import numpy as np


class TrenDeEspigas:
    """Los tiempos de disparo de una neurona durante un ensayo.

    Args:
        tiempos: lista o arreglo con los instantes de cada espiga, en segundos.
        nombre: etiqueta para identificarlo (por ejemplo, "ensayo 7").

    Raises:
        ValueError: si `tiempos` no es una lista ni un arreglo de NumPy.
    """

    def __init__(self, tiempos, nombre: str = "neurona"):
        if not isinstance(tiempos, (list, np.ndarray)):
            raise ValueError(
                f"tiempos debe ser una lista o un arreglo de NumPy, no {type(tiempos).__name__}"
            )
        self.nombre = nombre
        # Protegido (un guion bajo): nadie debería tocarlo desde fuera.
        # Ordenado desde el nacimiento, para no tener que ordenarlo nunca más.
        self._tiempos = np.sort(np.asarray(tiempos, dtype=float))

    # --- representación -------------------------------------------------
    def __repr__(self) -> str:
        if len(self) == 0:
            return f"TrenDeEspigas({self.nombre!r}, 0 espigas)"
        return (
            f"TrenDeEspigas({self.nombre!r}, {len(self)} espigas, "
            f"{self._tiempos[0]:.2f}–{self._tiempos[-1]:.2f} s)"
        )

    # --- protocolo de Python --------------------------------------------
    def __len__(self) -> int:
        """`len(tren)` devuelve cuántas espigas hay."""
        return len(self._tiempos)

    def __add__(self, otro: "TrenDeEspigas") -> "TrenDeEspigas":
        """`a + b` junta dos trenes en uno nuevo, sin tocar ninguno de los dos."""
        if not isinstance(otro, TrenDeEspigas):
            return NotImplemented
        return TrenDeEspigas(
            np.concatenate([self._tiempos, otro._tiempos]),
            nombre=f"{self.nombre}+{otro.nombre}",
        )

    def __lt__(self, otro: "TrenDeEspigas") -> bool:
        """`a < b` compara cuántas espigas tiene cada uno — habilita `sorted()`."""
        if not isinstance(otro, TrenDeEspigas):
            return NotImplemented
        return len(self) < len(otro)

    def __eq__(self, otro) -> bool:
        if not isinstance(otro, TrenDeEspigas):
            return NotImplemented
        return np.array_equal(self._tiempos, otro._tiempos)

    # --- lo que sabe hacer ----------------------------------------------
    @property
    def tiempos(self) -> np.ndarray:
        """Una *copia* de los tiempos, para que nadie modifique el original."""
        return self._tiempos.copy()

    def tasa_media(self, t_inicio: float, t_fin: float) -> float:
        """Espigas por segundo dentro de la ventana [t_inicio, t_fin).

        Devuelve 0.0 si la ventana está vacía o invertida.
        """
        if t_fin <= t_inicio:
            return 0.0
        dentro = (self._tiempos >= t_inicio) & (self._tiempos < t_fin)
        return float(dentro.sum() / (t_fin - t_inicio))

    @staticmethod
    def hz(n_espigas: int, segundos: float) -> float:
        """Convierte un conteo y una duración en hercios.

        Es estático porque no necesita ningún tren concreto: es aritmética.
        """
        if segundos <= 0:
            raise ValueError("la duración tiene que ser positiva")
        return n_espigas / segundos
