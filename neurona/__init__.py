"""Análisis de una neurona registrada en corteza premotora ventral.

Uso rápido:
    >>> from neurona import Neurona
    >>> n = Neurona("datos/neu_400_RR034075_002_OCPSLA_10_00_VPCizq.csv")
    >>> n.raster()
"""

from .cargador import Cargador
from .graficos import Graficador
from .neurona import Neurona
from .tasas import CalculadorDeTasas
from .tren import TrenDeEspigas

__all__ = ["Neurona", "Cargador", "CalculadorDeTasas", "Graficador", "TrenDeEspigas"]
