"""Construcción del target P_24h (igual que en clase, pero sin modificar el DataFrame de entrada).

Definición:
    Diff  = closed_date - created_date
    P_24h = 1  si Diff < 24 h   (la solicitud se cierra en menos de un día)
            0  si Diff >= 24 h

Nota: en clase se escribía `np.where(Diff.dt.days > 0, 0, 1)`. Es EXACTAMENTE lo mismo
(`dt.days > 0` <=> Diff >= 24 h), pero comparar contra un Timedelta deja la regla a la vista.
"""
import pandas as pd

from src.config import KEEP_UNCLOSED, MIN_SECONDS, TARGET, TARGET_HOURS


def make_target(
    df: pd.DataFrame,
    min_seconds: int = MIN_SECONDS,
    target_hours: int = TARGET_HOURS,
    keep_unclosed: bool = KEEP_UNCLOSED,
) -> pd.DataFrame:
    """Devuelve una copia con Diff, P_24h, hour y weekday, ya filtrada.

    Filtros (iguales a los de clase):
      * se descartan cierres en <= `min_seconds` segundos (cierres automáticos o espurios);
      * se descartan Diff negativos (closed_date anterior a created_date: error de captura);
      * se descartan solicitudes sin closed_date, salvo que `keep_unclosed=True`.
    """
    out = df.copy()

    # Texto -> fecha
    out["created_date"] = pd.to_datetime(out["created_date"])
    out["closed_date"] = pd.to_datetime(out["closed_date"])

    # Duración hasta el cierre
    out["Diff"] = out["closed_date"] - out["created_date"]

    # Filtro de clase. Con closed_date nula, Diff es NaT y (NaT > 150 s) es False -> se descarta.
    keep = out["Diff"].dt.total_seconds() > min_seconds
    if keep_unclosed:
        keep = keep | out["closed_date"].isna()
    out = out.loc[keep].copy()

    # Target: (NaT < 24 h) es False -> las no cerradas quedan en 0 cuando keep_unclosed=True
    out[TARGET] = (out["Diff"] < pd.Timedelta(hours=target_hours)).astype(int)

    # Variables derivadas de created_date (se conocen al registrar la solicitud)
    out["hour"] = out["created_date"].dt.hour
    out["weekday"] = out["created_date"].dt.day_name()

    return out
