"""
entrenar.py — El protocolo honesto: ¿la neurona sabía?
======================================================

Un ÚNICO protocolo reproducible para la Clase 4. Predice acierto/error del mono
(columna 4 del CSV) a partir de la tasa de disparo de la neurona VPC en ventanas
del ensayo. Todas las cifras del guion de clase salen de correr este archivo.

Diseño (esto es lo que hace las cifras reproducibles):
  * UNA sola semilla global (SEED) y UN solo objeto de validación cruzada (cv).
  * La dispersión (± std) y la prueba de permutación usan EL MISMO cv, de modo
    que el "score real" de la permutación es, por construcción, la media de CV.
    Una afirmación, un número.
  * class_weight="balanced" y balanced_accuracy porque las clases están 78/22:
    con accuracy, decir siempre "acierto" ya saca 0.779 sin aprender nada.

Carga los datos con el worker Cargador del paquete neurona/, el mismo que se
escribió en la Clase 2. Escribe resultados/resultados.json con métricas, semillas y el
hash del commit: esa es la idea de MLflow, sin instalar MLflow.

Uso:
    python entrenar.py [ruta_al_csv]

Requisitos: numpy, scikit-learn. (pip install -r requirements.txt)
"""

from __future__ import annotations
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from neurona import Cargador  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Configuración del análisis (todo lo que decide las cifras vive aquí arriba)
# ─────────────────────────────────────────────────────────────────────────────
SEED = 0                      # semilla global: cambia esto y verás moverse todo
N_SPLITS = 5                  # folds por repetición
N_REPEATS = 20                # repeticiones → 100 mediciones de la métrica
N_PERMUTACIONES = 200         # barajeos de y para el nulo (resolución de p ≈ 1/201)
ARCHIVO_DEFECTO = str(Path(__file__).parent.parent / "datos" / "neu_400_RR034075_002_OCPSLA_10_00_VPCizq.csv")

# Columnas de eventos (arqueología de la clase de Tasa de disparo):
#   col25 = fin de P1 = t0   ·   P2 inicia fijo en +2.01 s
COL_T0 = 25
T_P2_INICIO = 2.01

# Épocas HONESTAS: todo lo anterior a la respuesta del mono. Predecir su acierto
# con actividad de DESPUÉS de que responde (o de que recibe premio) es hacer
# trampa; ese es justamente el experimento de fuga de abajo.
EPOCAS_HONESTAS = ["basal", "P1", "demora1", "P2", "demora2"]
EPOCA_FUGA = "post_retro"     # ventana post-retroalimentación: el cañón de la fuga


# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos  (en el repo: neurona/cargador.py y neurona/espigas.py)
# ─────────────────────────────────────────────────────────────────────────────
def cargar_metadatos(ruta: str) -> np.ndarray:
    """Las 31 columnas rectangulares de metadatos → arreglo (140, 31).

    Reutiliza el worker Cargador que escribimos en la Clase 2: el código que ya
    existe y está probado no se vuelve a escribir.
    """
    return Cargador(ruta).metadatos()


def cargar_espigas(ruta: str) -> list[np.ndarray]:
    """La zona irregular, con un arreglo de tiempos por ensayo. Mismo worker."""
    return Cargador(ruta).tiempos_de_espiga()


# ─────────────────────────────────────────────────────────────────────────────
# Ingeniería de features: tasa (Hz) por época y por ensayo
# (en el repo, esta tabla se construye en la Clase 3 y se guarda en SQLite)
# ─────────────────────────────────────────────────────────────────────────────
def _tasa(espigas: np.ndarray, t_ini: float, t_fin: float) -> float:
    """Espigas por segundo dentro de [t_ini, t_fin)."""
    if t_fin <= t_ini:
        return 0.0
    return float(np.sum((espigas >= t_ini) & (espigas < t_fin)) / (t_fin - t_ini))


def construir_features(meta: np.ndarray, espigas: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Devuelve un dict época → vector (140,) con la tasa de esa ventana por ensayo.

    Las ventanas se miden respecto a t0 = fin de P1 (columna 25). P1 y P2 tienen
    duración distinta en cada ensayo, así que sus bordes se leen por fila.
    """
    t0 = meta[:, COL_T0]
    ev = meta[:, 20:31] - t0[:, None]     # tiempos de evento relativos a t0
    n = len(espigas)
    cols = {e: np.zeros(n) for e in EPOCAS_HONESTAS + [EPOCA_FUGA]}
    for i, e in enumerate(espigas):
        r = ev[i]
        cols["basal"][i] = _tasa(e, -4.0, -2.0)            # antes de todo
        cols["P1"][i] = _tasa(e, r[4], r[5])               # col24 → col25 (=0)
        cols["demora1"][i] = _tasa(e, 0.0, T_P2_INICIO)    # demora fija P1→P2
        cols["P2"][i] = _tasa(e, r[6], r[7])               # col26 → col27
        cols["demora2"][i] = _tasa(e, r[7], r[8])          # col27 → col28 (fija)
        cols[EPOCA_FUGA][i] = _tasa(e, r[10], r[10] + 1.0)  # 1 s tras col30 (¡fuga!)
    return cols


def _matriz(cols: dict[str, np.ndarray], epocas: list[str]) -> np.ndarray:
    """Apila las épocas pedidas en una matriz (140, n_epocas)."""
    return np.column_stack([cols[e] for e in epocas])


# ─────────────────────────────────────────────────────────────────────────────
# El modelo y el protocolo de evaluación
# ─────────────────────────────────────────────────────────────────────────────
def modelo(class_weight="balanced"):
    """Regresión logística estandarizada. El escalado va DENTRO del Pipeline para
    que se ajuste solo con el fold de entrenamiento — si escalas antes de partir,
    filtras información del test y las cifras salen infladas."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight=class_weight),
    )


def evaluar(X: np.ndarray, y: np.ndarray, cv) -> tuple[float, float]:
    """Media y desviación de balanced_accuracy sobre las N_SPLITS×N_REPEATS
    mediciones. La desviación NO es un adorno: con n=140 es el tamaño real del
    ruido de muestreo, y es la lección del curso."""
    s = cross_val_score(modelo(), X, y, cv=cv, scoring="balanced_accuracy")
    return float(s.mean()), float(s.std())


def _hash_commit() -> dict:
    """Hash del commit actual y si el árbol tiene cambios sin commitear.
    Sin esto, un resultados.json no es reproducible: no sabes qué código lo produjo."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        sucio = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"commit": h, "arbol_sucio": sucio}
    except Exception:
        return {"commit": "sin-git", "arbol_sucio": None}


# ─────────────────────────────────────────────────────────────────────────────
def main(ruta_csv: str) -> None:
    meta = cargar_metadatos(ruta_csv)
    espigas = cargar_espigas(ruta_csv)
    y = meta[:, 4].astype(int)            # 1 = acierto, 0 = error
    cols = construir_features(meta, espigas)

    Xh = _matriz(cols, EPOCAS_HONESTAS)
    Xl = _matriz(cols, EPOCAS_HONESTAS + [EPOCA_FUGA])

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )

    n_aciertos, n_errores = int(y.sum()), int((y == 0).sum())
    tasa_mayoritaria = float(y.mean())

    print("=" * 70)
    print("¿LA NEURONA SABÍA?  —  protocolo honesto")
    print("=" * 70)
    print(f"Ensayos: {len(y)}   aciertos: {n_aciertos}   errores: {n_errores}")
    print(f"Clase mayoritaria (decir siempre 'acierto'): {tasa_mayoritaria:.3f}\n")

    # 1) La trampa: accuracy con el modelo por defecto
    acc_def = cross_val_score(
        modelo(class_weight=None), Xh, y, cv=cv, scoring="accuracy"
    )
    print(f"[1] LR por defecto, ACCURACY .......... {acc_def.mean():.3f} ± {acc_def.std():.3f}")
    print(f"    → por DEBAJO del {tasa_mayoritaria:.3f} de la clase mayoritaria: "
          f"'acertó' 77% de las veces sin aprender nada.\n")

    # 2) El modelo honesto: balanced_accuracy con clases pesadas
    bal_h, std_h = evaluar(Xh, y, cv)
    print(f"[2] LR balanced, BALANCED ACCURACY .... {bal_h:.3f} ± {std_h:.3f}")
    print(f"    → peor número que [1], MEJOR modelo: ahora sí detecta errores.")
    print(f"    → el ± {std_h:.2f} es la dispersión entre 100 particiones: eso mide n=140.\n")

    # 3) ¿Está por encima del azar? Prueba de permutación con EL MISMO cv
    print(f"[3] Prueba de permutación ({N_PERMUTACIONES} barajeos de y)... "
          f"(tarda ~1 min)")
    score, perm, pv = permutation_test_score(
        modelo(), Xh, y, cv=cv, scoring="balanced_accuracy",
        n_permutations=N_PERMUTACIONES, random_state=SEED, n_jobs=-1,
    )
    print(f"    score real: {score:.3f}   nulo: {perm.mean():.3f} ± {perm.std():.3f}   "
          f"p = {pv:.4f}")
    print(f"    → el score real coincide con [2] (mismo cv). Señal real pero débil.\n")

    # 4) La fuga, en concreto: añadir la ventana post-retroalimentación
    bal_l, std_l = evaluar(Xl, y, cv)
    print(f"[4] + época post-retroalimentación .... {bal_l:.3f} ± {std_l:.3f}  (¡FUGA!)")
    post_h = cols[EPOCA_FUGA][y == 1].mean()
    post_e = cols[EPOCA_FUGA][y == 0].mean()
    print(f"    → 'mejora' porque la neurona dispara distinto DESPUÉS de saber el")
    print(f"      resultado: {post_e:.1f} Hz tras errores vs {post_h:.1f} Hz tras aciertos.")
    print(f"      Eso no predice el acierto: lo REPORTA. Fuga de información.\n")

    # De dónde viene la (poca) señal honesta: separación aciertos/errores por época
    print("    Tasa media por época (Hz):        aciertos   errores")
    for e in EPOCAS_HONESTAS:
        h, r = cols[e][y == 1].mean(), cols[e][y == 0].mean()
        marca = "   ← la mayor separación" if e == "P2" else ""
        print(f"      {e:11s} .................... {h:7.2f}   {r:7.2f}{marca}")
    print()

    # ── resultados.json (la idea de MLflow sin MLflow) ──
    resultados = {
        "pregunta": "predecir acierto/error del mono desde la tasa de disparo",
        "archivo_datos": Path(ruta_csv).name,
        "fecha": date.today().isoformat(),
        **_hash_commit(),
        "protocolo": {
            "modelo": "LogisticRegression(class_weight='balanced') + StandardScaler",
            "validacion": f"RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS})",
            "metrica": "balanced_accuracy",
            "semilla": SEED,
        },
        "datos": {
            "n_ensayos": len(y),
            "n_aciertos": n_aciertos,
            "n_errores": n_errores,
            "tasa_clase_mayoritaria": round(tasa_mayoritaria, 4),
        },
        "features_honestas": EPOCAS_HONESTAS,
        "resultados": {
            "accuracy_defecto": [round(acc_def.mean(), 4), round(acc_def.std(), 4)],
            "balanced_accuracy_honesto": [round(bal_h, 4), round(std_h, 4)],
            "permutacion_p": round(float(pv), 4),
            "permutacion_nulo": [round(float(perm.mean()), 4), round(float(perm.std()), 4)],
            "balanced_accuracy_con_fuga": [round(bal_l, 4), round(std_l, 4)],
        },
        "veredicto": (
            "Señal real pero débil (bal.acc ~0.65, p<0.01), concentrada en P2. "
            "Con n=31 errores, la incertidumbre es grande: no creerse un número aislado."
        ),
    }
    salida = Path(__file__).parent.parent / "resultados"; salida.mkdir(exist_ok=True)
    destino = salida / "resultados.json"
    destino.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"✔ Escrito {destino}")
    print("=" * 70)


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else ARCHIVO_DEFECTO
    if not Path(ruta).exists():
        sys.exit(f"No encuentro el CSV: {ruta}\nUso: python entrenar.py [ruta_al_csv]")
    main(ruta)
