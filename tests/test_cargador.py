"""Pruebas del Cargador — el worker que lee el archivo."""

from pathlib import Path

import numpy as np
import pytest

from neurona import Cargador

CSV = Path(__file__).parent.parent / "datos" / "neu_400_RR034075_002_OCPSLA_10_00_VPCizq.csv"


def test_carga_los_140_ensayos():
    """El archivo del laboratorio tiene 140 ensayos. Ni 139 ni 141."""
    c = Cargador(CSV)
    assert c.metadatos().shape == (140, 31)


def test_hay_un_tren_por_ensayo():
    """La fila i de las espigas DEBE corresponder a la fila i de los metadatos.

    Si esta prueba falla, cada espiga quedaría etiquetada con la clase del ensayo
    equivocado, y todos los análisis posteriores estarían mal sin avisar.
    """
    c = Cargador(CSV)
    assert len(c.tiempos_de_espiga()) == len(c.metadatos())


def test_los_ensayos_tienen_distinto_numero_de_espigas():
    """El archivo es irregular: si todos midieran lo mismo, algo se truncó."""
    largos = [len(t) for t in Cargador(CSV).tiempos_de_espiga()]
    assert len(set(largos)) > 1


def test_un_archivo_que_no_existe_avisa_de_inmediato():
    with pytest.raises(FileNotFoundError):
        Cargador("no_existe_este_archivo.csv")
