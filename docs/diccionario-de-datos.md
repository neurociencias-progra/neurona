# Diccionario de datos

> **Esto se llena en la Clase 3.** Las columnas que no sepamos qué son se marcan como
> desconocidas: **la honestidad es parte del diccionario**. Un «no sé» documentado vale
> más que una etiqueta inventada.

Archivo: `datos/neu_400_RR034075_002_OCPSLA_10_00_VPCizq.csv` · 140 filas (ensayos) ·
columnas 0–30 metadatos · columnas 31+ tiempos de espiga (número variable por fila).

| Columna | Nombre | Qué es | Unidades | Cómo lo sabemos |
|---|---|---|---|---|
| 0 | `ensayo` | Número de ensayo, 1 a 140 | — | va de 1 a 140 sin repetir |
| 1 | `clase` | Clase de estímulo | — | 14 valores, 10 ensayos cada uno |
| 2 | | | | |
| 3 | | | | |
| 4 | `acierto` | ¿Acertó el mono? | 0/1 | binaria; su media por clase va de 0,50 a 0,90 |
| … | | | | |

**Pista para empezar:** corre `df.nunique()` sobre las 31 columnas. Varias tienen **un
solo valor** en los 140 ensayos. Una columna con un solo valor **no lleva información** —
y descubrirlo tú es más útil que si te lo contaran.
