# `Neurona` — el proyecto del módulo de programación

Análisis de una **neurona real** registrada en la corteza premotora ventral de un mono
durante una tarea de comparación de intervalos. 140 ensayos, 20 068 espigas, 14 clases de
estímulo.

> **Este es el repositorio de REFERENCIA: el proyecto terminado.** Si eres estudiante del
> curso, el tuyo es el que clonaste de la plantilla — este es para consultar cuando te
> atores, y para ver hacia dónde vamos.

---

## Arrancar en tres clics

**No hace falta la terminal.** En VSCode:

1. Abre esta carpeta (`Archivo → Abrir carpeta…`).
2. `Ctrl+Shift+P` (en Mac, `Cmd+Shift+P`) → escribe **`Python: Create Environment`**.
3. Elige **Venv** o **Conda** —te ofrecerá lo que tu máquina tenga— y marca
   **`requirements.txt`** cuando te lo pregunte.

VSCode crea el entorno, lo selecciona e instala todo. Ya está.

<details>
<summary>Si prefieres hacerlo a mano en la terminal</summary>

| | crear | activar |
|---|---|---|
| **venv · Windows** | `python -m venv .venv` | `.venv\Scripts\activate` |
| **venv · macOS/Linux** | `python3 -m venv .venv` | `source .venv/bin/activate` |
| **conda · cualquiera** | `conda create -n neurona python=3.12` | `conda activate neurona` |

Los cuatro caminos terminan igual:

```bash
pip install -r requirements.txt
```

> ⚠ **Windows:** si al activar sale *«running scripts is disabled on this system»*, corre
> una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` en PowerShell. No pide
> administrador. O usa el camino de los tres clics y te lo ahorras.
</details>

---

## Qué hace

```python
from neurona import Neurona

n = Neurona("datos/neu_400_RR034075_002_OCPSLA_10_00_VPCizq.csv")

n                    # Neurona('neu_400_…': 140 ensayos, 20068 espigas, 14 clases)
len(n)               # 140
n[7]                 # TrenDeEspigas('ensayo 8', 160 espigas, -6.34–8.75 s)
n.desempeno()        # 0.7786  ← el mono acertó el 78 % de las veces

n.raster()           # el ráster de los 140 ensayos, agrupados y coloreados por clase
n.tasa_por_clase()   # una curva de tasa de disparo por cada clase de estímulo
```

---

## La idea: muchos *workers*, un orquestador

```
                        Neurona          ← el ORQUESTADOR
                  (los crea y los conecta)
                            │
        ┌───────────────────┼───────────────────┐
    Cargador        CalculadorDeTasas       Graficador
   lee el archivo   calcula hercios        dibuja figuras
```

Cada *worker* hace **una sola cosa** y **no importa a ningún otro**. El `Cargador` no sabe
qué es una tasa de disparo; el `Graficador` no sabe leer archivos. La única clase que los
conoce a los tres es `Neurona`, que les pasa datos de uno a otro.

¿Por qué molestarse? Porque cuando algo falla, sabes **dónde** mirar. Y porque puedes
cambiar el `Graficador` entero sin tocar una línea del `Cargador`.

Es el mismo reparto que usa el pipeline de registro neuronal en tiempo real del
laboratorio, donde nueve nodos procesan señal y un orquestador los conecta. Aquí está a
escala de una clase.

---

## Estructura

```
neurona/
├── neurona/                 el paquete
│   ├── tren.py                  TrenDeEspigas — el objeto de datos (Clase 1)
│   ├── cargador.py              WORKER 1 · leer el archivo
│   ├── tasas.py                 WORKER 2 · calcular hercios
│   ├── graficos.py              WORKER 3 · dibujar
│   └── neurona.py               ORQUESTADOR · la clase Neurona
├── tests/                   las pruebas (pytest)
├── scripts/
│   ├── construir_bd.py          Clase 3 · CSV → SQLite (dos tablas)
│   └── entrenar.py              Clase 4 · el modelo, y resultados.json
├── docs/
│   ├── diccionario-de-datos.md  qué significa cada columna (Clase 3)
│   └── reporte.md               tus respuestas a las cajas 💬
├── datos/                   el CSV de la sesión
└── resultados/              resultados.json — qué salió, y con qué código
```

---

## Las pruebas

```bash
pytest
```

Doce pruebas, y ninguna es decorativa. La más importante:

```python
def test_hay_un_tren_por_ensayo():
    """La fila i de las espigas DEBE corresponder a la fila i de los metadatos."""
```

Si esa falla, cada espiga queda etiquetada con la clase del ensayo equivocado y **todos**
los análisis salen mal **sin avisar**. Un error ruidoso te detiene; uno silencioso te
publica.

---

## Los dos guiones

```bash
python scripts/construir_bd.py    # CSV irregular → dos tablas SQL
python scripts/entrenar.py        # ¿la neurona sabía? → resultados/resultados.json
```

El segundo tarda ~1 minuto: hace 200 permutaciones para comprobar si lo que encuentra es
señal o casualidad. Su salida va a `resultados/resultados.json`, **con el hash del commit
que la produjo**. Sin ese dato, un resultado no se puede reproducir: no sabes qué código
lo generó.

---

## Reglas de la casa

* **Lo regenerable no se versiona.** El `.sqlite` está en `.gitignore`; el script que lo
  construye, no. Se versiona la receta, no el pastel.
* **Los secretos nunca se suben.** El token de GitHub va en `.env`, que también está
  ignorado.
* **Cada cambio va con su mensaje.** `git commit -m "Closes #1 — el Cargador lee el CSV"`
  cierra el issue #1 solo.
* **Si usas una herramienta de IA, decláralo** en `docs/reporte.md`. Usarla está
  permitido; no decirlo, no. Y pídele que te explique un error o que revise tu código —
  no que resuelva el ejercicio por ti, porque entonces el que aprende es él.
