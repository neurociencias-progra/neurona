"""EL ORQUESTADOR — la clase `Neurona`.

Los tres workers (`Cargador`, `CalculadorDeTasas`, `Graficador`) hacen cada uno
una cosa y **no se conocen entre sí**. Esta clase es la única que los conoce a los
tres: los crea, les pasa datos de uno a otro, y ofrece hacia fuera una interfaz
simple.

Ese reparto —muchos workers pequeños, un orquestador que los coordina— es el mismo
que usa el pipeline de registro neuronal en tiempo real del laboratorio, donde nueve
nodos procesan señal y un `NeuralPipelineManager` los conecta. Aquí está a escala
de una clase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .cargador import Cargador
from .graficos import Graficador
from .tasas import CalculadorDeTasas
from .tren import TrenDeEspigas

# Columnas del CSV que este código usa por nombre en vez de por número.
# Salieron de la «arqueología de datos» de la clase de tasa de disparo.
COL_ENSAYO = 0
COL_CLASE = 1
COL_ACIERTO = 4


class Neurona:
    """Una neurona registrada, con todos sus ensayos.

    Ejemplo:
        >>> n = Neurona("datos/neu_400_RR034075_002_OCPSLA_10_00_VPCizq.csv")
        >>> len(n)
        140
        >>> n.raster()
        >>> n.tasa_por_clase()

    Args:
        ruta_csv: el archivo de la sesión.
        ventana: anchura de la ventana para las tasas, en segundos.
        paso: resolución temporal de las tasas, en segundos.
    """

    def __init__(self, ruta_csv, ventana: float = 0.2, paso: float = 0.02):
        # El orquestador crea a sus workers. Ellos no se crean entre sí.
        self._cargador = Cargador(ruta_csv)
        self._calculador = CalculadorDeTasas(ventana=ventana, paso=paso)
        self._graficador = Graficador()

        # …y conecta la salida de uno con la entrada del siguiente.
        self.metadatos = self._cargador.metadatos()
        tiempos = self._cargador.tiempos_de_espiga()

        self.nombre = Path(ruta_csv).stem
        self.trenes = [
            TrenDeEspigas(t, nombre=f"ensayo {int(self.metadatos[i, COL_ENSAYO])}")
            for i, t in enumerate(tiempos)
        ]

    # --- representación -------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Neurona({self.nombre!r}: {len(self)} ensayos, "
            f"{self.n_espigas()} espigas, {len(self.clases())} clases)"
        )

    def __len__(self) -> int:
        """`len(neurona)` son sus ensayos.

        Sale gratis porque una Neurona TIENE una lista de trenes (composición):
        no hereda de nada, contiene cosas.
        """
        return len(self.trenes)

    def __getitem__(self, i: int) -> TrenDeEspigas:
        """`neurona[7]` devuelve el tren del ensayo 7 — y hace la clase iterable."""
        return self.trenes[i]

    # --- lo que sabe de sí misma ----------------------------------------
    def n_espigas(self) -> int:
        """Total de espigas en toda la sesión."""
        return sum(len(t) for t in self.trenes)

    def clases(self) -> np.ndarray:
        """Las clases de estímulo presentes, ordenadas."""
        return np.unique(self.metadatos[:, COL_CLASE]).astype(int)

    def aciertos(self) -> np.ndarray:
        """Vector booleano: ¿acertó el mono en cada ensayo?"""
        return self.metadatos[:, COL_ACIERTO].astype(bool)

    def desempeno(self) -> float:
        """Fracción de ensayos acertados."""
        return float(self.aciertos().mean())

    def _ventana_temporal(self) -> tuple[float, float]:
        """El rango de tiempo que cubren las espigas, redondeado hacia fuera."""
        todos = [t.tiempos for t in self.trenes if len(t) > 0]
        if not todos:
            return 0.0, 1.0
        juntos = np.concatenate(todos)
        return float(np.floor(juntos.min())), float(np.ceil(juntos.max()))

    # --- las dos figuras --------------------------------------------------
    def raster(self, por_clase: bool = True, ax=None):
        """Dibuja el ráster de toda la sesión.

        Args:
            por_clase: si es True, agrupa los ensayos por clase y los colorea.
        """
        etiquetas = self.metadatos[:, COL_CLASE].astype(int) if por_clase else None
        return self._graficador.raster(
            [t.tiempos for t in self.trenes],
            etiquetas=etiquetas,
            titulo=f"Ráster — {self.nombre}",
            ax=ax,
        )

    def tasa_por_clase(self, ax=None):
        """Dibuja una curva de tasa de disparo por cada clase de estímulo."""
        t_min, t_max = self._ventana_temporal()
        clase_de = self.metadatos[:, COL_CLASE].astype(int)
        curvas = {}
        for clase in self.clases():
            de_esta_clase = [
                self.trenes[i].tiempos for i in np.where(clase_de == clase)[0]
            ]
            centros, media = self._calculador.curva_media(de_esta_clase, t_min, t_max)
            curvas[int(clase)] = media
        return self._graficador.curvas(
            centros, curvas, titulo=f"Tasa por clase — {self.nombre}", ax=ax
        )
