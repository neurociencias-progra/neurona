"""Clase 3 — Del CSV a una base de datos.

Un CSV irregular no cabe en una tabla: las filas tienen largos distintos. La
solución no es forzarlo, es usar **dos tablas**:

    ensayos (140 filas)   una fila por ensayo, con sus metadatos
    espigas (~20 000)     una fila por ESPIGA, con el ensayo al que pertenece

A eso se le llama «formato largo», y es como las bases de datos guardan cosas de
largo variable. Con las dos tablas y un JOIN se responde cualquier pregunta.

Uso:
    python scripts/construir_bd.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from neurona import Cargador  # noqa: E402

CSV = Path(__file__).parent.parent / "datos" / "neu_400_RR034075_002_OCPSLA_10_00_VPCizq.csv"
BD = Path(__file__).parent.parent / "neurona.sqlite"

# Nombres de las columnas que sí sabemos qué son (arqueología de la clase de tasa).
# Las que no, se quedan como col5, col6… y así se documentan: honestamente.
NOMBRES = {0: "ensayo", 1: "clase", 2: "boton_presionado", 3: "boton_correcto", 4: "acierto"}


def main() -> None:
    cargador = Cargador(CSV)
    meta = cargador.metadatos()
    tiempos = cargador.tiempos_de_espiga()

    # --- tabla 1: un renglón por ensayo ---
    columnas = [NOMBRES.get(j, f"col{j}") for j in range(meta.shape[1])]
    ensayos = pd.DataFrame(meta, columns=columnas)
    ensayos["ensayo"] = ensayos["ensayo"].astype(int)
    ensayos["clase"] = ensayos["clase"].astype(int)
    ensayos["acierto"] = ensayos["acierto"].astype(int)

    # --- tabla 2: un renglón por ESPIGA ---
    filas = [
        {"ensayo": int(meta[i, 0]), "t": float(t)}
        for i, ts in enumerate(tiempos)
        for t in ts
    ]
    espigas = pd.DataFrame(filas)

    with sqlite3.connect(BD) as con:
        ensayos.to_sql("ensayos", con, if_exists="replace", index=False)
        espigas.to_sql("espigas", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_espigas_ensayo ON espigas(ensayo)")

    print(f"✔ {BD.name}")
    print(f"   ensayos: {len(ensayos):>6} filas × {len(ensayos.columns)} columnas")
    print(f"   espigas: {len(espigas):>6} filas × {len(espigas.columns)} columnas")
    print("\nPruébala:")
    print('   SELECT clase, COUNT(*), AVG(acierto) FROM ensayos GROUP BY clase;')
    print("\n⚠ El .sqlite NO se sube a git (está en .gitignore): se regenera con este")
    print("  script. Lo que se versiona es la receta, no el pastel.")


if __name__ == "__main__":
    main()
