# ¿Se cerrará en menos de 24 horas? Predicción para solicitudes NYC 311

**Actividad:** ejercicio modular del tema 05 · Introducción a la IA y ML (CUGDL)
**Integrantes:** Abraham Dominguez Valdez, Angel, Cristo y Adrian _(TODO equipo: nombres completos)_

> Convenciones de este archivo: lo marcado como **TODO equipo** es texto que depende de los resultados
> reales o de decisiones del equipo. Las tablas numéricas salen de `reports/tablas.md`, que genera
> `train.py` (se copian y pegan; no se teclean a mano).

---

## 1. Objetivo, fuente de datos y target

**Objetivo.** Determinar qué modelo de clasificación predice mejor si una solicitud de NYC 311 se cerrará
en menos de 24 horas, imaginando que la predicción se hace **en el momento en que se registra la solicitud**
y que el modelo solo conoce lo que había pasado hasta esa fecha.

**Fuente.** NYC Open Data, conjunto _311 Service Requests from 2010 to Present_
(identificador `erm2-nwe9`), consultado con la API Socrata:
`https://data.cityofnewyork.us/resource/erm2-nwe9.json`.

**Muestra.** Solicitudes creadas entre `DATE_FROM` y `DATE_TO` de `src/config.py`
(_TODO equipo: copiar aquí el periodo final_) con `unique_key` terminada en 0, lo que da una muestra de ~10 %
(igual que en clase). _TODO equipo: número de filas tras los filtros, salida de `explore.py`._

**Target `P_24h`.**

```
Diff  = closed_date − created_date
P_24h = 1  si Diff < 24 h      (se cierra rápido)
        0  si Diff ≥ 24 h
```

Igual que en clase, se descartan las solicitudes cerradas en 150 segundos o menos (cierres automáticos o
espurios), las que tienen `Diff` negativo y las que no tienen `closed_date`
(`KEEP_UNCLOSED = False` en `config.py`; con `True` se conservarían como `P_24h = 0`).
_TODO equipo: cuántas filas se descartan por cada filtro (salida de `explore.py`) y el % de `P_24h = 1`._

**Variables disponibles al registrar la solicitud.** Solo se usan columnas que existen en el instante de la
captura: `hour` y `weekday` (derivadas de `created_date`), `complaint_type`, `borough`,
`open_data_channel_type` y `location_type`.
No se usan `closed_date` ni `Diff` (son el resultado que queremos predecir), ni `unique_key` (es un
identificador consecutivo que codifica el tiempo). Tampoco se descargan columnas que se llenan después del
registro (estatus, resolución, fechas de actualización). `load_xy` lanza un error si una variable prohibida
(`FORBIDDEN_FEATURES`) llegara a las predictoras.

## 2. Dependencias y reproducción

Requiere Python ≥ 3.10 y [uv](https://docs.astral.sh/uv/).

```bash
git clone <URL-del-repositorio>
cd <carpeta-del-repositorio>
uv sync                          # instala las dependencias de pyproject.toml

uv run python explore.py         # (opcional) perfil de los datos y vista previa de los cortes
uv run python train.py           # EXPERIMENTO COMPLETO  ← comando para reproducir
uv run pytest                    # (opcional) pruebas del target y de la validación temporal
```

Sin `uv`: `pip install numpy pandas scikit-learn requests matplotlib tabulate` y `python train.py`.

**Cómo se obtienen los datos.** No se suben al repositorio. La primera vez que se corre `explore.py` o
`train.py` se descargan de la API (unos segundos) y se guardan en `data/` como caché; las siguientes
ejecuciones usan esa caché. Opcionalmente se puede definir la variable de entorno `SOCRATA_APP_TOKEN`
para evitar límites de la API. La semilla es `SEED = 42`.

`train.py` deja en `reports/`: `tablas.md` (todas las tablas de este informe), `cortes.csv`,
`metricas_por_corte.csv`, `configuraciones_probadas.csv`, `resultado_final.json` y `matriz_confusion.png`.
Para correr solo algunos modelos: `uv run python train.py --models dummy baseline_tipo logreg rf`.

## 3. Validación temporal

```
|----------------------- desarrollo ------------------------|24 h|--- prueba final ---|
corte 1: [ entrenamiento ]24h[ validación 1 ]
corte 2: [      entrenamiento      ]24h[ validación 2 ]
corte 3: [            entrenamiento            ]24h[ validación 3 ]
```

- **Bloque de prueba final:** los últimos `TEST_DAYS = 7` días del periodo, que incluyen un fin de semana
  completo. Se usa **una sola vez**, al final, con el modelo ya elegido.
- **Tres cortes de validación** (`N_SPLITS = 3`) de `VAL_DAYS = 5` días cada uno, justo antes de la prueba.
  Es una validación _walk-forward_ con ventana **expansiva**: cada corte entrena con todo el pasado
  disponible y valida en los días siguientes. (Con `TRAIN_WINDOW_DAYS = n` pasa a ventana deslizante.)
  Se eligió expansiva porque el periodo es corto y recortar el pasado dejaría muy pocos datos al primer corte.
- **Cortes por fecha, no por número de filas.** El volumen de solicitudes cambia entre semana y fin de
  semana; cortar por días calendario hace que cada ventana sea interpretable. Es la misma idea que
  `TimeSeriesSplit(gap=...)` de scikit-learn, con fechas legibles. Los cortes se pasan tal cual a
  `GridSearchCV(cv=...)`, así que **todos los modelos y todas las configuraciones usan exactamente los mismos cortes**.

**Cómo verificamos que las etiquetas de entrenamiento ya se conocían.** `P_24h` solo se conoce cuando han
pasado 24 h desde la creación. Para un corte con fecha `T` (inicio de la validación), solo entrenan
solicitudes con `created_date ≤ T − 24 h`: todas tuvieron tiempo de cumplir sus 24 h, así que sabemos con
certeza si se cerraron antes (`P_24h = 1`) o no (`P_24h = 0`). Si entrenáramos con solicitudes creadas entre
`T − 24 h` y `T`, solo conoceríamos la etiqueta de las que ya se cerraron —las rápidas— y el entrenamiento
quedaría sesgado. El mismo hueco de 24 h separa el desarrollo de la prueba final.
`verify_split` (en `src/temporal_split.py`) comprueba esto en cada ejecución e imprime, por corte, la última
solicitud de entrenamiento + 24 h frente a la fecha del corte; `tests/test_temporal_split.py` además verifica que
detecta una fuga si se introduce a propósito.
El preprocesamiento (imputación, escalado, one-hot) va **dentro del `Pipeline`**, por lo que en cada corte
se ajusta únicamente con las filas de entrenamiento de ese corte.

### Tabla de cortes

> **TODO equipo:** pegar aquí la sección «Cortes temporales» de `reports/tablas.md`
> (fechas, número de solicitudes y % de `P_24h = 1` de cada corte, incluida la prueba final).

## 4. Modelos, hiperparámetros y justificación

Todos los modelos son un `Pipeline` (preprocesador + modelo). Además de los modelos se incluyen dos
**referencias** que no compiten por ser elegidas: sirven para saber cuánto aporta cada modelo.

| Modelo | Ventaja esperada con estos datos | Limitación que podría afectarlo | Hiperparámetros probados |
|---|---|---|---|
| Referencia: clase mayoritaria | Piso mínimo: qué se logra sin usar ninguna variable | AUC = 0.5 por construcción | — |
| Referencia: tasa por tipo de queja | Regla simple y explicable; muchos tipos de queja siguen el mismo proceso de atención | Ignora hora, día, borough y canal; no combina variables | `min_count` ∈ {5, 30} |
| Regresión logística | Rápida y estable con muchas columnas one-hot; coeficientes interpretables; la regularización limita el sobreajuste | Solo relaciones lineales; no captura interacciones (tipo × hora) ni la ciclicidad de la hora | `C` ∈ {0.01, 0.1, 1, 10} × `class_weight` ∈ {None, balanced} |
| Bosque aleatorio | Captura interacciones y no linealidades sin escalado | Árboles muy fragmentados por el one-hot; puede sobreajustar al periodo de entrenamiento; más lento | `max_depth` ∈ {8, 16, None} × `min_samples_leaf` ∈ {1, 5, 20}; 200 árboles |
| k vecinos | No supone una forma para la frontera | Con muchas columnas one-hot las distancias pierden sentido; lento al predecir | `n_neighbors` ∈ {15, 51, 101} × `weights` ∈ {uniform, distance} |
| SVM (RBF) | Frontera flexible; buen desempeño con dimensión alta y tamaño medio | Costo cuadrático en filas; sensible a `C`; AUC con `decision_function` | `C` ∈ {0.5, 2, 8} |

**Preprocesamiento (cambios respecto a clase).** Imputación por mediana en la variable numérica (`KNNImputer`
no tiene sentido con una sola columna numérica y sin faltantes), imputación constante `sin_dato` en las
categóricas, `StandardScaler` en `hour` y `OneHotEncoder(handle_unknown="ignore")`: en el futuro aparecen
tipos de queja no vistos en el entrenamiento y no debe fallar.
_TODO equipo: si cambian algo más (por ejemplo, codificar la hora de forma cíclica), explicarlo aquí._

## 5. Métrica principal

**Métrica principal: ROC AUC promedio entre los tres cortes.** Se fijó **antes** de comparar modelos.

- Mide qué tan bien el modelo **ordena** las solicitudes (rápidas antes que lentas) sin depender de un umbral.
- Es comparable entre cortes aunque cambie la proporción de cierres rápidos de una ventana a otra
  (la exactitud y el F1 sí dependen de esa proporción).
- Se reportan también exactitud, precisión, _recall_ y F1 (umbral 0.5) por corte, porque la matriz de
  confusión final sí depende de un umbral.
- El criterio de elección es la **media entre cortes**; a igualdad, menor desviación entre cortes.

> **TODO equipo:** confirmar con el balance real de `P_24h` (salida de `explore.py`). Si las clases resultaran
> muy desbalanceadas, discutir si conviene `f1` o promedio de precisión; en ese caso cambiar
> `PRIMARY_METRIC` en `config.py` **antes** de correr `train.py` y ajustar este texto.

## 6. Resultados

> **TODO equipo:** pegar las secciones de `reports/tablas.md` y escribir la interpretación.

1. **Métricas por corte** (tabla «Métrica principal por corte» y «Todas las métricas por corte»).
2. **Mejor configuración por modelo** (y, si se desea, la tabla completa de configuraciones probadas).
3. **Resultado final en el bloque de prueba** (modelo elegido frente a las referencias).
4. **Matriz de confusión** de `reports/matriz_confusion.png` y la explicación de cada error:
   - _Falso positivo:_ se predijo cierre en < 24 h, pero tardó 24 h o más.
   - _Falso negativo:_ se predijo cierre lento, pero se resolvió en menos de 24 h.
   _TODO equipo: decidir cuál de los dos errores es más costoso en este contexto y cuál es el balance
   precisión/recall deseable, con los números reales._

### Conclusión

> **TODO equipo.** Responder: ¿qué modelo se eligió y con qué configuración? ¿Supera claramente a la
> referencia por tipo de queja? ¿El resultado en la prueba final es consistente con el de los cortes
> (estabilidad)? ¿Qué variable o información adicional mejoraría la predicción?

### Limitaciones

- Una muestra de pocas semanas: no captura estacionalidad anual ni cambios de política de las agencias.
- Se descartan solicitudes sin `closed_date` y cierres de ≤ 150 s, lo que puede sesgar hacia casos que sí se resuelven.
- Los días festivos del periodo (_p. ej._ Año Nuevo, Día de Martin Luther King Jr.) pueden alterar tiempos de cierre.
- Las variables disponibles al registrar la solicitud son pocas: no hay texto de la queja ni ubicación detallada.

## 7. Contribuciones

> **TODO equipo:** cada integrante debe tener al menos una contribución sustantiva (código, diseño de la
> validación, análisis de resultados o documentación). Número de commits por persona:
> `git shortlog -sn --all` (o la pestaña _Insights → Contributors_ de GitHub).

| Integrante | Commits | Contribución principal |
|---|---|---|
| Abraham | _TODO_ | Configuración, carga de datos y construcción del target `P_24h` (`config.py`, `io.py`, `target.py`); documentación de datos, target y reproducción (secciones 1 y 2) |
| Adrian | _TODO_ | Diseño de la validación temporal: prueba final, cortes, hueco de 24 h y verificación anti-fuga (`temporal_split.py`); exploración de datos (`explore.py`); sección 3 |
| Cristo | _TODO_ | Preprocesamiento, modelos, referencias y rejillas de hiperparámetros con su justificación (`preprocessor.py`, `models.py`); generación de las tablas del informe (`report.py`); sección 4 |
| Angel | _TODO_ | Búsqueda de hiperparámetros por cortes, evaluación final y matriz de confusión (`evaluation.py`), flujo completo del experimento (`train.py`); métrica principal y análisis de resultados (secciones 5 y 6) |

## 8. Estructura del repositorio

```
train.py              punto de entrada del experimento
explore.py            perfil de los datos y vista previa de los cortes
src/
  config.py           parámetros: periodo, variables, ventanas de validación, métrica, semilla
  io.py               descarga con caché y armado de X, y (ordenados en el tiempo)
  target.py           construcción de P_24h, hour y weekday
  temporal_split.py   bloque de prueba, cortes con hueco de 24 h y verificación anti-fuga
  preprocessor.py     imputación, escalado y one-hot (se ajusta dentro del Pipeline)
  models.py           modelos, referencias y rejillas de hiperparámetros
  evaluation.py       búsqueda por cortes, evaluación final y matriz de confusión
  report.py           genera reports/tablas.md
tests/                pruebas del target (test_target), de la validación temporal (test_temporal_split) y de los modelos (test_models)
reports/              tablas y figuras generadas
```

## 9. Opcionales

_No implementados todavía._ Versionamiento de modelos con MLflow e interfaz con Streamlit o marimo.
