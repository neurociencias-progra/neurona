"""Pruebas del CalculadorDeTasas y del TrenDeEspigas."""

import numpy as np
import pytest

from neurona import CalculadorDeTasas, TrenDeEspigas


def test_diez_espigas_en_dos_segundos_son_cinco_hercios():
    """La aritmética más básica: si esto falla, nada de lo demás importa."""
    assert TrenDeEspigas.hz(10, 2) == 5.0


def test_la_tasa_en_una_ventana_cuenta_solo_lo_de_dentro():
    tren = TrenDeEspigas([0.1, 0.5, 0.9, 5.0])   # tres dentro de [0, 1), una fuera
    assert tren.tasa_media(0.0, 1.0) == 3.0


def test_una_ventana_invertida_no_revienta_y_devuelve_cero():
    """Preferimos 0.0 a una excepción: es un caso borde, no un error del usuario."""
    assert TrenDeEspigas([1.0, 2.0]).tasa_media(2.0, 1.0) == 0.0


def test_el_tren_nace_ordenado():
    tren = TrenDeEspigas([3.0, 1.0, 2.0])
    assert np.array_equal(tren.tiempos, [1.0, 2.0, 3.0])


def test_sumar_dos_trenes_no_modifica_ninguno():
    """`a + b` devuelve algo nuevo: no muta a los sumandos. Como en matemáticas."""
    a, b = TrenDeEspigas([1.0, 2.0]), TrenDeEspigas([3.0])
    c = a + b
    assert len(c) == 3 and len(a) == 2 and len(b) == 1


def test_los_tiempos_que_devuelve_son_una_copia():
    """Si devolviera el arreglo interno, cualquiera podría corromper el tren."""
    tren = TrenDeEspigas([1.0, 2.0])
    tren.tiempos[0] = 99.0
    assert tren.tiempos[0] == 1.0


def test_tiempos_invalidos_se_rechazan_al_construir():
    with pytest.raises(ValueError):
        TrenDeEspigas("esto no es una lista de tiempos")


def test_la_curva_tiene_un_valor_por_paso():
    calc = CalculadorDeTasas(ventana=0.2, paso=0.1)
    centros, tasas = calc.curva(np.array([0.5, 0.6]), 0.0, 1.0)
    assert len(centros) == len(tasas) == 10
