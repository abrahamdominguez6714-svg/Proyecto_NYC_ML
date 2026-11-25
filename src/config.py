"""Configuración central del experimento NYC 311 · P_24h.

Todo lo que cambia el resultado (periodo de datos, tamaño de las ventanas de
validación, métrica principal, semilla) vive aquí y SOLO aquí. Así el README
puede citar estos valores y cualquier compañero puede reproducir el experimento
sin buscar números "escondidos" dentro del código.
"""
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"          # caché de la descarga (NO se sube a GitHub)
REPORTS_DIR = ROOT / "reports"    # tablas, figuras y JSON que alimentan el README

SEED = 42

# ── 1. Datos: API Socrata de NYC Open Data (311 Service Requests) ────────────
API = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
COLUMNS = [
    "unique_key", "created_date", "closed_date", "complaint_type",
    "borough", "open_data_channel_type", "location_type",
]

# Periodo descargado. Para ampliar los datos (más cortes, más filas) basta con
# cambiar estas dos fechas: la caché lleva las fechas en el nombre, así que no
# choca con descargas anteriores.
DATE_FROM = "2025-01-01T00:00:00"   # incluida
DATE_TO = "2025-02-01T00:00:00"     # excluida

QUERY = {
    "$select": ",".join(COLUMNS),
    # unique_key terminada en 0  ->  muestra de ~10 % (igual que en clase)
    "$where": f"created_date >= '{DATE_FROM}' AND created_date < '{DATE_TO}' "
              "AND unique_key like '%0'",
    "$order": "created_date ASC, unique_key ASC",
}
CACHE = DATA_DIR / f"nyc311_{DATE_FROM[:10]}_{DATE_TO[:10]}.json"
PAGE_SIZE = 5000        # filas por petición a la API
MAX_ROWS = 200_000      # tope de seguridad: si se alcanza, NO se guarda una muestra truncada

# ── 2. Target ────────────────────────────────────────────────────────────────
TARGET = "P_24h"        # 1 = se cierra en menos de 24 h, 0 = tarda 24 h o más
TARGET_HOURS = 24
MIN_SECONDS = 150       # filtro de clase: se descartan cierres de <= 150 s (cierres automáticos/espurios)
# En clase, las solicitudes SIN closed_date se descartan en silencio (NaT > 150 s es False).
# Si se pone True se conservan con P_24h = 0 (la descarga es muy posterior al periodo,
# así que "sigue abierta" implica "no se cerró en 24 h"). Por defecto: comportamiento de clase.
KEEP_UNCLOSED = False

# ── 3. Variables ─────────────────────────────────────────────────────────────
# Solo variables CONOCIDAS en el instante en que se registra la solicitud.
TIME_COL = "created_date"
NUM_FEATURES = ["hour"]
CAT_FEATURES = ["complaint_type", "borough", "open_data_channel_type", "location_type", "weekday"]
FEATURES = NUM_FEATURES + CAT_FEATURES
# Columnas que jamás pueden entrar como predictoras (fuga de información o identificadores).
FORBIDDEN_FEATURES = {"closed_date", "Diff", TARGET, "unique_key"}

# ── 4. Validación temporal ───────────────────────────────────────────────────
TEST_DAYS = 7            # último bloque de días reservado para la prueba final
N_SPLITS = 3             # número de cortes entrenamiento/validación
VAL_DAYS = 5             # días de cada ventana de validación
GAP_HOURS = 24           # hueco entre entrenamiento y validación (>= TARGET_HOURS)
TRAIN_WINDOW_DAYS = None # None = ventana expansiva; un entero = ventana deslizante de ese tamaño

# ── 5. Selección de modelo ───────────────────────────────────────────────────
# Se define ANTES de comparar modelos (ver README, sección "Métrica principal").
# Opciones: "roc_auc", "f1", "recall", "precision", "accuracy".
PRIMARY_METRIC = "roc_auc"

# Modelos que corre train.py por defecto (los nombres están en src/models.py).
MODELS_TO_RUN = ["dummy", "baseline_tipo", "logreg", "rf", "knn", "svm"]
